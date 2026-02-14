import base64
import io
import librosa
import numpy as np

class AudioPreprocessingError(Exception):
    pass

def preprocess_audio(audio_base64, target_sample_rate=16000):
    try:
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            raise AudioPreprocessingError(f"Incorrect Base64 String: {str(e)}")

        try:
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sample_rate = librosa.load(audio_buffer, sr=None, mono=False)
        except Exception as e:
            raise AudioPreprocessingError(f"Failed to load MP3 audio: {str(e)}")

        if waveform.ndim > 1:
            waveform = librosa.to_mono(waveform)

        if sample_rate != target_sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
            sample_rate = target_sample_rate

        waveform = waveform.astype(np.float32)

        try:
            waveform, _ = librosa.effects.trim(waveform, top_db=20)
        except Exception as e:
            raise AudioPreprocessingError(f"Failed to trim silence: {str(e)}")

        try:
            waveform = librosa.util.normalize(waveform)
        except Exception as e:
            raise AudioPreprocessingError(f"Failed to normalize amplitude: {str(e)}")

        return waveform, sample_rate

    except AudioPreprocessingError:
        raise
    except Exception as e:
        raise AudioPreprocessingError(f"Unexpected error during preprocessing: {str(e)}")