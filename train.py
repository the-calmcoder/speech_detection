import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# --- GLOBAL INIT ---
global_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

class TrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        for param in self.hubert.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_values):
        outputs = self.hubert(input_values)
        embedding = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(embedding)
        return logits

class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        possible_paths = [os.path.join(root_dir, 'data'), root_dir]
        data_path = None
        for p in possible_paths:
            if os.path.exists(os.path.join(p, 'ai')) and os.path.exists(os.path.join(p, 'human')):
                data_path = p
                break
        
        if not data_path:
            return

        for label, subdir in [(1.0, 'ai'), (0.0, 'human')]:
            path = os.path.join(data_path, subdir)
            files = [f for f in os.listdir(path) if f.endswith(('.mp3', '.wav', '.flac'))]
            for file in files:
                try:
                    waveform, sr = librosa.load(os.path.join(path, file), sr=16000, mono=True)
                    tensor_wave = torch.tensor(waveform).squeeze()
                    self.data.append((tensor_wave, label))
                except Exception:
                    continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    waveforms = [item[0].numpy() for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    inputs = global_feature_extractor(
        waveforms, 
        sampling_rate=16000, 
        return_tensors="pt", 
        padding=True
    )
    return inputs.input_values, labels

def main(root_dir, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    dataset = AudioDataset(root_dir)
    if len(dataset) == 0:
        print("Error: No samples found.")
        return
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    model = TrainingModel().to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        for input_values, labels in dataloader:
            input_values = input_values.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_values)
            loss = criterion(logits.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.classifier.state_dict(), 'classifier_weights.pth')
    print("Training Complete. Weights saved to classifier_weights.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    main(args.root_dir, args.epochs)