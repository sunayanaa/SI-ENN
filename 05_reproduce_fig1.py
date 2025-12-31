# 07_generate_fig1_violin_v2.py
# Version: 2.0 (Self-Contained Fix)
# Description: Defines Dataset class locally -> Calculates Stats -> Plots Violin.

import torch
from torch.utils.data import DataLoader, Subset, Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import librosa
from scipy.stats import entropy
from google.colab import drive

# 1. Setup & Connect
print("--- 1. Connecting and Locating Files ---")
drive.mount('/content/drive')
project_root = '/content/drive/MyDrive/ASVspoof_Project'

# Auto-locate paths
found_proto = None
found_flac = None
target_proto = 'ASVspoof2019.LA.cm.train.trn.txt'

for root, dirs, files in os.walk(project_root):
    if target_proto in files:
        found_proto = os.path.join(root, target_proto)
    if os.path.basename(root) == 'flac' and 'ASVspoof2019_LA_train' in os.path.dirname(root):
        found_flac = root

if not found_proto or not found_flac:
    # Fallback default
    found_proto = os.path.join(project_root, 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
    found_flac = os.path.join(project_root, 'data/LA/ASVspoof2019_LA_train/flac')

print(f"Protocol: {found_proto}")
print(f"Audio: {found_flac}")

# 2. Re-Define Dataset Class (Crucial Fix)
class ASVSpoof2019LADataset(Dataset):
    def __init__(self, flac_dir, protocol_file, max_len=64000):
        self.flac_dir = flac_dir
        self.max_len = max_len
        self.df = pd.read_csv(protocol_file, sep=' ', header=None, 
                              names=['Speaker', 'Filename', 'System', 'Null', 'Label'])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['Filename']
        label_str = row['Label']
        file_path = os.path.join(self.flac_dir, filename + '.flac')
        
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=16000)
        except Exception:
            audio = np.zeros(self.max_len)
        
        # Pad/Truncate
        if len(audio) < self.max_len:
            audio = np.pad(audio, (0, self.max_len - len(audio)), mode='constant')
        else:
            audio = audio[:self.max_len]
            
        # Compute Entropy Feature (H_r)
        # Use first 32000 samples (2s) for speed
        seg = audio[:32000]
        if len(seg) > 17:
            a = librosa.lpc(seg, order=16)
            res = np.convolve(seg, a, mode='same')
            spec = np.abs(np.fft.fft(res, n=512))**2
            psd = spec[:256] / (np.sum(spec[:256]) + 1e-10)
            ent = entropy(psd)
            h_r = ent / np.log(len(psd))
        else:
            h_r = 0.5
        
        return torch.tensor(audio, dtype=torch.float32), torch.tensor(0 if label_str == 'bonafide' else 1).long(), torch.tensor(h_r).float()

# 3. Analyze Data
print("\n--- 2. Extracting Features (2000 samples) ---")
dataset = ASVSpoof2019LADataset(flac_dir=found_flac, protocol_file=found_proto)

# Random subset for fair distribution
indices = np.random.choice(len(dataset), 2000, replace=False)
loader = DataLoader(Subset(dataset, indices), batch_size=32, shuffle=False, num_workers=2)

entropies = []