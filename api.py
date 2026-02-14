import os
import base64
from flask import Flask, request, jsonify

from audio_processing import preprocess_audio
from model_core import AudioInferenceEngine
from explainability import ExplainabilityEngine

app = Flask(__name__)
app.json.sort_keys = False

# Load API Key from environment variable
API_KEY = os.getenv("API_KEY")
if API_KEY is None:
    raise RuntimeError("API_KEY environment variable is not set")

# Supported languages (case-insensitive handling)
SUPPORTED_LANGUAGES = {"english", "tamil", "hindi", "malayalam", "telugu"}

print("--- Loading Engines ---")
audio_inference_engine = AudioInferenceEngine()
explainability_engine = ExplainabilityEngine()
print("--- All Engines Ready ---")


def validate_api_key(req):
    api_key = req.headers.get("X-API-Key") or req.headers.get("x-api-key")
    if api_key != API_KEY:
        return False
    return True


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/api/voice-detection", methods=["POST"])
def voice_detection():
    if not validate_api_key(request):
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400

        if "audioBase64" not in data:
            return jsonify({"error": "Missing audioBase64"}), 400

        language = data.get("language", "English")
        if not isinstance(language, str):
            return jsonify({"error": "Invalid language format"}), 400

        language_normalized = language.strip().lower()
        if language_normalized not in SUPPORTED_LANGUAGES:
            return jsonify({
                "error": "Unsupported language",
                "supportedLanguages": ["English", "Tamil", "Hindi", "Malayalam", "Telugu"]
            }), 400

        audio_base64 = data["audioBase64"]

        waveform, sample_rate = preprocess_audio(audio_base64)
        ai_probability, _ = audio_inference_engine.infer(waveform, sample_rate)

        if ai_probability >= 0.5:
            classification = "AI_GENERATED"
            confidence = ai_probability
        else:
            classification = "HUMAN"
            confidence = 1.0 - ai_probability

        explanation_text, _ = explainability_engine.explain(
            waveform, sample_rate, classification, confidence
        )

        response_data = {
            "status": "success",
            "language": language_normalized.capitalize(),
            "classification": classification,
            "confidenceScore": round(confidence, 4),
            "explanation": explanation_text
        }

        return jsonify(response_data), 200

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        # Log internally (replace with structured logging in production)
        print(f"Internal Server Error: {str(e)}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500


if __name__ == "__main__":
    # For production, run with Gunicorn or another WSGI server:
    # gunicorn -w 2 -b 0.0.0.0:5000 api:app
    app.run(host="0.0.0.0", port=5000)
