FROM python:3.10-slim

WORKDIR /app

# System deps for audio decoding
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Pre-download model weights (avoid cold starts)
RUN python - << 'EOF'
from transformers import HubertModel, Wav2Vec2FeatureExtractor
HubertModel.from_pretrained("facebook/hubert-base-ls960")
Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
EOF

EXPOSE 5000

# Production WSGI server
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "api:app"]
