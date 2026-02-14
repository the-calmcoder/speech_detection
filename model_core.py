import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import os
import numpy as np

class AudioInferenceEngine:
    def __init__(self, model_name="facebook/hubert-base-ls960", weights_path="classifier_weights.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Initializing Inference Engine on {self.device} ---")
        
        # 1. Load the Base Model (HuBERT)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # 2. Define the Classifier Structure (Must match training!)
        self.classifier = nn.Linear(768, 1).to(self.device)

        # 3. Load Your Trained Weights
        if os.path.exists(weights_path):
            print(f" Loading trained weights from {weights_path}")
            self.classifier.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            print(f" CRITICAL WARNING: {weights_path} not found! Model will predict randomly.")
        
        self.classifier.eval()

    def infer(self, waveform, sample_rate):
        """
        Input: waveform (numpy array or tensor), sample_rate (int)
        Output: probability (float 0.0-1.0), embedding (numpy array)
        """
        try:
            # Ensure proper shape (1D numpy array)
            if hasattr(waveform, 'numpy'):
                waveform = waveform.numpy()
            if len(waveform.shape) > 1:
                waveform = waveform.squeeze()

            # Preprocess for HuBERT
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Take mean of hidden states to get a single vector per clip
                embedding = outputs.last_hidden_state.mean(dim=1)
                
                # Classification
                logits = self.classifier(embedding)
                ai_probability = torch.sigmoid(logits).item()

            return ai_probability, embedding.cpu().numpy()

        except Exception as e:
            # Log and fail fast — do not return fabricated predictions
            print(f"Inference Error: {e}")
            raise RuntimeError(f"Inference failed: {str(e)}") from e
