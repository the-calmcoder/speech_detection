# AI Voice Detection System

A deep learning system for detecting AI-generated speech. The system uses a fine-tuned HuBERT model to classify audio as human or AI-generated, with temporal segment analysis for detecting partial deepfakes.

---

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Model Architecture & Approach](#model-architecture--approach)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Third-Party Attribution](#third-party-attribution)

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- Docker (recommended for deployment)

### Local Setup

1. **Clone the repository**

```bash
git clone <repo-url>
cd speech_detection-main
```

2. **Create a virtual environment and install dependencies**

```bash
python -m venv env
source env/bin/activate        # Linux/macOS
env\Scripts\activate           # Windows
pip install --upgrade pip
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

3. **Set the API key**

```bash
export API_KEY="your-secret-key"       # Linux/macOS
set API_KEY=your-secret-key            # Windows
```

4. **Generate human baseline (first time only)**

Place human voice samples (.mp3/.wav/.flac) in `data/human/`, then run:

```bash
python humanity.py
```

This creates `human_baseline.json` used by the explainability engine.

5. **Train the classifier (first time only)**

Place AI voice samples in `data/ai/` and human samples in `data/human/`, then run:

```bash
python train.py --root_dir . --epochs 10
```

This saves `classifier_weights.pth`.

6. **Run the server**

```bash
python api.py
```

Server starts at `http://0.0.0.0:5000`.

### Docker Setup

```bash
docker build -t voice-detection .
docker run -e API_KEY="your-secret-key" -p 5000:5000 voice-detection
```

The Docker image pre-downloads HuBERT model weights during build to avoid cold starts. Gunicorn runs with 2 workers in production.

---

## Model Architecture & Approach

### Overview

The system uses a transfer learning approach built on Meta's HuBERT (Hidden-Unit BERT) speech representation model. HuBERT is a self-supervised model pre-trained on 960 hours of LibriSpeech audio, producing rich acoustic embeddings that capture speech patterns at a deep level.

### Architecture

```
Audio Input (any format)
    │
    ▼
┌──────────────────────────┐
│   Audio Preprocessing    │  Base64 decode → soundfile load → mono → resample 16kHz
│   (audio_processing.py)  │  → center truncation (10s max) → amplitude normalization
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│   Temporal Segmentation  │  3s windows, 2s hop → overlapping segments
│   (audio_processing.py)  │  Skipped for audio ≤ 4s
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│   HuBERT Base            │  facebook/hubert-base-ls960
│   (model_core.py)        │  12 transformer layers, 768-dim embeddings
│                          │  Frozen weights (no fine-tuning)
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│   Mean Pooling           │  Temporal mean across hidden states → 768-dim vector
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│   Binary Classifier      │  nn.Linear(768, 1) → Sigmoid
│   (classifier_weights.pth)│  Fine-tuned on AI vs human speech
└──────────────────────────┘
    │
    ▼
  Output: AI probability (0.0 – 1.0)
```

### Key Design Decisions

**Temporal Segment Analysis** — Audio is split into overlapping 3-second windows with a 2-second hop. Each segment is classified independently, and results are aggregated using a weighted average where higher-probability segments contribute more. This enables detection of partial deepfakes where only a portion of the audio is AI-generated.

**Center Truncation** — For latency optimization, audio longer than 10 seconds is truncated to a 10-second center slice before segmentation. The center is the most representative portion, avoiding intro silence and trailing noise.

**Batched Inference** — Segments are processed through HuBERT in batches of 8 instead of individually, reducing the number of forward passes and improving throughput.

**Weighted Aggregation** — The overall AI probability is computed as a weighted average of per-segment probabilities. Weights are `p + 0.1`, biasing toward segments with higher AI probability so that even a small spliced deepfake portion is surfaced rather than averaged away.

### Training

The classifier head (a single linear layer) is trained while HuBERT's weights remain frozen. Training uses `BCEWithLogitsLoss` and the Adam optimizer with a learning rate of 0.001. The training script (`train.py`) loads audio from `data/ai/` and `data/human/` directories.

---

## API Reference

### Authentication

All requests require the `X-API-Key` header matching the server's `API_KEY` environment variable.

### Endpoints

#### `GET /health`

Health check endpoint.

**Response:**
```json
{"status": "ok"}
```

#### `POST /api/voice-detection`

Classifies an audio clip as human or AI-generated.

**Request Body:**
```json
{
    "audioBase64": "<base64-encoded-audio>",
    "language": "English"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audioBase64` | string | Yes | Base64-encoded audio file (WAV, MP3, FLAC) |
| `language` | string | No | Language of the audio. Default: `"English"`. Supported: English, Tamil, Hindi, Malayalam, Telugu |

**Success Response (200):**
```json
{
    "status": "success",
    "classification": "AI_GENERATED",
    "confidenceScore": 0.9234
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"success"` |
| `classification` | string | `"AI_GENERATED"` or `"HUMAN"` |
| `confidenceScore` | float | Confidence in the classification (0.0 – 1.0) |

**Error Response (400):**
```json
{
    "error": "Missing audioBase64"
}
```

**Error Response (401):**
```json
{
    "error": "Unauthorized"
}
```

### Source Code Alignment

The API response fields map directly to source code in `api.py`:

| Response Field | Source |
|----------------|--------|
| `classification` | `"AI_GENERATED"` if `overall_probability >= 0.5`, else `"HUMAN"` |
| `confidenceScore` | `overall_probability` for AI, `1.0 - overall_probability` for human, rounded to 4 decimal places |

The inference pipeline called by the endpoint:
1. `preprocess_audio()` in `audio_processing.py` — decodes, resamples, truncates, normalizes
2. `infer_segments()` in `model_core.py` — segments, batches through HuBERT, aggregates
3. Classification threshold applied in `api.py` — probability ≥ 0.5 = AI

---

## Project Structure

```
speech_detection-main/
├── api.py                   # Flask API server with /health and /api/voice-detection endpoints
├── audio_processing.py      # Audio decoding, resampling, truncation, segmentation
├── model_core.py            # HuBERT inference engine with batched segment processing
├── explainability.py        # Generates human-readable explanations for classifications
├── humanity.py              # Builds human voice baseline profile from training data
├── train.py                 # Training script for the binary classifier head
├── classifier_weights.pth   # Trained classifier weights
├── human_baseline.json      # Statistical baseline of human voice features
├── requirements.txt         # Python dependencies
└── Dockerfile               # Production container with gunicorn
```

---

## Third-Party Attribution

| Library | Version | License | Usage |
|---------|---------|---------|-------|
| [HuBERT](https://huggingface.co/facebook/hubert-base-ls960) (Meta/Facebook AI) | base-ls960 | MIT | Pre-trained speech representation model used as the feature extractor backbone |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | 4.36.2 | Apache 2.0 | Model loading and feature extraction via `HubertModel` and `Wav2Vec2FeatureExtractor` |
| [PyTorch](https://pytorch.org/) | 2.1.0 | BSD-3-Clause | Deep learning framework for model inference and training |
| [Flask](https://flask.palletsprojects.com/) | 3.0.0 | BSD-3-Clause | Web framework for the REST API |
| [librosa](https://librosa.org/) | latest | ISC | Audio resampling via `librosa.resample` |
| [SoundFile](https://github.com/bastibe/python-soundfile) | latest | BSD-3-Clause | Fast audio file decoding via `sf.read` |
| [NumPy](https://numpy.org/) | <2 | BSD-3-Clause | Numerical operations for audio arrays and probability aggregation |
| [scikit-learn](https://scikit-learn.org/) | latest | BSD-3-Clause | Machine learning utilities |
| [Gunicorn](https://gunicorn.org/) | latest | MIT | Production WSGI server for deployment |

The HuBERT model (`facebook/hubert-base-ls960`) was developed by Meta AI and pre-trained on 960 hours of LibriSpeech data. Only the classifier head is fine-tuned on custom AI vs human speech data; the base model weights remain frozen.
