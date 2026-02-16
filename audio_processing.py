import base64
import io
import librosa
import numpy as np
import soundfile as sf

MAX_INFERENCE_DURATION = 10.0

class AudioPreprocessingError(Exception):
    pass

def truncate_center(waveform, sample_rate, max_duration_sec):
    """Truncates the audio to the center segment if it exceeds max duration."""
    max_samples = int(max_duration_sec * sample_rate)
    total_samples = len(waveform)

    if total_samples <= max_samples:
        return waveform

    # Take from the center.
    start = (total_samples - max_samples) // 2
    return waveform[start : start + max_samples]

def segment_audio(waveform, sample_rate, window_sec=3.0, hop_sec=2.0, min_duration_sec=4.0):
    total_samples = len(waveform)
    total_duration = total_samples / sample_rate

    # Short clips don't need segmentation.
    if total_duration <= min_duration_sec:
        return [(waveform, 0.0, round(total_duration, 3))]

    window_samples = int(window_sec * sample_rate)
    hop_samples = int(hop_sec * sample_rate)

    segments = []
    start = 0

    while start < total_samples:
        end = min(start + window_samples, total_samples)
        segment = waveform[start:end]

        # Skip trailing segments under 1 second.
        if len(segment) < sample_rate:
            break

        start_time = round(start / sample_rate, 3)
        end_time = round(end / sample_rate, 3)
        segments.append((segment, start_time, end_time))

        start += hop_samples

    # Fallback if no segments produced.
    if not segments:
        return [(waveform, 0.0, round(total_duration, 3))]

    return segments

def preprocess_audio(audio_base64, target_sample_rate=16000):
    try:
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            raise AudioPreprocessingError(f"Incorrect Base64 String: {str(e)}")

        try:
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sample_rate = sf.read(audio_buffer, dtype="float32")
        except Exception as e:
            raise AudioPreprocessingError(f"Failed to load MP3 audio: {str(e)}")

        # Convert to mono if stereo
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)    

        if sample_rate != target_sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
            sample_rate = target_sample_rate

        waveform = waveform.astype(np.float32)

        # Truncate to center slice for speed on long audio.
        waveform = truncate_center(waveform, sample_rate, MAX_INFERENCE_DURATION)


#####THE BELOW NORMALIZATION STEPS ARE COMMENTED OUT AS PER SOURCE, BUT CAN BE RE-ENABLED IF NEEDED.#####
        # Optional: Trim silence (Commented out as per source)
        # try:
        #     waveform, _ = librosa.effects.trim(waveform, top_db=20)
        # except Exception as e:
        #     raise AudioPreprocessingError(f"Failed to trim silence: {str(e)}")

        # Optional: Librosa normalize (Commented out as per source)
        # try:
        #     waveform = librosa.util.normalize(waveform)
        # except Exception as e:
        #     raise AudioPreprocessingError(f"Failed to normalize amplitude: {str(e)}")

        ### NORMALIZE AMPLITUDE (Manual implementation)




        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        return waveform, sample_rate

    except AudioPreprocessingError:
        raise
    except Exception as e:
        raise AudioPreprocessingError(f"Unexpected error during preprocessing: {str(e)}")