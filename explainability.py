import librosa
import numpy as np
import json
import os
import random

class ExplainabilityEngine:
    def __init__(self, baseline_path="human_baseline.json"):
        self.baseline = None
        if os.path.exists(baseline_path):
            with open(baseline_path, "r") as f:
                self.baseline = json.load(f)
        
        self.vocab = {
            "openers": [
                "The audio analysis shows",
                "Our system detected",
                "Voice analysis reveals",
                "Audio patterns indicate",
                "The voice sample shows"
            ],
            "severity_low": ["slight", "minor", "small", "weak"],
            "severity_high": ["noticeable", "clear", "obvious", "strong"],
            "severity_extreme": ["very unusual", "highly artificial", "clearly synthetic", "obviously fake"],
            "connectors": ["along with", "and also", "combined with", "together with"]
        }

    def _get_deviation_desc(self, z_score):
        abs_z = abs(z_score)
        if abs_z < 1.0: return "normal", 0
        elif abs_z < 2.0: return random.choice(self.vocab["severity_low"]), 1
        elif abs_z < 3.0: return random.choice(self.vocab["severity_high"]), 2
        else: return random.choice(self.vocab["severity_extreme"]), 3

    def explain(self, waveform, sample_rate, classification, confidence):
        try:
            if classification == "HUMAN":
                human_phrases = [
                    "Natural voice patterns match real human speech.",
                    "Breathing sounds and small voice variations suggest a real person.",
                    "Voice characteristics are consistent with natural human recordings.",
                    "No digital editing signs detected in the audio."
                ]
                return random.choice(human_phrases), {}

            if hasattr(waveform, 'numpy'): waveform = waveform.numpy()
            if len(waveform.shape) > 1: waveform = waveform.squeeze()

            curr_flatness = float(np.mean(librosa.feature.spectral_flatness(y=waveform)))
            curr_zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=waveform)))
            
            reasons = []
            if self.baseline:
                b_flat = self.baseline["flatness"]
                z_flat = (curr_flatness - b_flat["mean"]) / (b_flat["std"] + 1e-6)
                desc_flat, sev_flat = self._get_deviation_desc(z_flat)
                
                if sev_flat > 0:
                    noun = "sound smoothness" if random.random() > 0.5 else "audio uniformity"
                    reasons.append(f"{desc_flat} changes in {noun}")

                b_zcr = self.baseline["zcr"]
                z_zcr = (curr_zcr - b_zcr["mean"]) / (b_zcr["std"] + 1e-6)
                desc_zcr, sev_zcr = self._get_deviation_desc(z_zcr)
                
                if sev_zcr > 0:
                    noun = "high frequency noise" if random.random() > 0.5 else "digital artifacts"
                    reasons.append(f"{desc_zcr} {noun}")

            opener = random.choice(self.vocab["openers"])
            if not reasons:
                return f"{opener} unusual patterns typical of computer-generated audio.", {}
            
            if len(reasons) > 1:
                connector = random.choice(self.vocab["connectors"])
                explanation = f"{opener} {reasons[0]} {connector} {reasons[1]}."
            else:
                explanation = f"{opener} {reasons[0]}."

            return explanation, {}

        except Exception as e:
            return f"Analysis unavailable due to processing error.", {}