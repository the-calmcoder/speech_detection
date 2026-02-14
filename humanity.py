import os
import librosa
import numpy as np
import json
import warnings

# Suppress Version Warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
# Use absolute path to avoid "Folder not found" errors
# Assuming you are running this from the 'Voice_ai' folder
current_dir = os.getcwd()
DATA_PATH = os.path.join(current_dir, "data", "human")
OUTPUT_FILE = "human_baseline.json"

def get_features(file_path):
    try:
        # Load audio (suppress warnings)
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        
        # --- FIX: USE KEYWORD ARGUMENTS (y=y) ---
        # Librosa 0.10+ requires 'y=' for all feature functions
        
        # 1. Zero Crossing Rate
        zcr_raw = librosa.feature.zero_crossing_rate(y=y)
        zcr = np.mean(zcr_raw)
        
        # 2. Spectral Flatness
        flatness_raw = librosa.feature.spectral_flatness(y=y)
        flatness = np.mean(flatness_raw)
        
        # 3. Spectral Centroid
        centroid_raw = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid = np.mean(centroid_raw)
        
        return [zcr, flatness, centroid]
    except Exception as e:
        print(f"⚠️ Failed to process {os.path.basename(file_path)}: {e}")
        return None

print(f"🧠 Profiling Human Baseline from: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print(f"❌ CRITICAL ERROR: The folder '{DATA_PATH}' does not exist.")
    print("   Make sure you are running this script from the 'Voice_ai' folder.")
    exit()

features_list = []
files = [f for f in os.listdir(DATA_PATH) if f.endswith(('.mp3', '.wav', '.flac'))]

print(f"   Found {len(files)} files. Analyzing...")

for f in files:
    feats = get_features(os.path.join(DATA_PATH, f))
    if feats:
        features_list.append(feats)

# Check if we actually got data
if len(features_list) == 0:
    print("\n❌ ERROR: No valid features extracted.")
    print("   This likely means your librosa version mismatch is still breaking things.")
    print("   But this script SHOULD have fixed it.")
    exit()

# Calculate Statistics
data_matrix = np.array(features_list)
means = np.mean(data_matrix, axis=0)
stds = np.std(data_matrix, axis=0)

profile = {
    "zcr": {"mean": float(means[0]), "std": float(stds[0])},
    "flatness": {"mean": float(means[1]), "std": float(stds[1])},
    "centroid": {"mean": float(means[2]), "std": float(stds[2])}
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(profile, f)

print(f"\n✅ Success! Baseline saved to {OUTPUT_FILE}")
print(json.dumps(profile, indent=2))