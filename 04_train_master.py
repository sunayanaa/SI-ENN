# 04_experiment_runner_v3_1_MASTER.py
# Version: 3.1 (Fixes SincConv Size Mismatch)
# Description: Fully working Training Loop with corrected tensor shapes.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import pandas as pd
import numpy as np
import librosa
import os
import time
from scipy.stats import entropy
from google.colab import drive

# 1. Mount Drive & Auto-Locate Paths
drive.mount('/content/drive')
project_root = '/content/drive/MyDrive/ASVspoof_Project'

print("\n--- 1. Auto-Locating Files ---")
found_proto = None
found_flac = None

target_proto = 'ASVspoof2019.LA.cm.train.trn.txt'
target_flac_dir = 'flac' 

# Quick search to verify paths
for root, dirs, files in os.walk(project_root):
    if target_proto in files:
        found_proto = os.path.join(root, target_proto)
    if os.path.basename(root) == 'flac' and 'ASVspoof2019_LA_train' in os.path.dirname(root):
        found_flac = root

if found_proto and found_flac:
    print(f"SUCCESS: Found Protocol: {found_proto}")
    print(f"SUCCESS: Found Audio Dir: {found_flac}")
else:
    # Fail-safe hardcode if auto-search misses (unlikely but safe)
    found_proto = os.path.join(project_root, 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
    found_flac = os.path.join(project_root, 'data/LA/ASVspoof2019_LA_train/flac')
    print("Using default paths.")

# --- 2. Dataset Class ---
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
            audio, sr = librosa.load(file_path, sr=16000)
        except Exception:
            audio = np.zeros(self.max_len)
        
        if len(audio) < self.max_len:
            audio = np.pad(audio, (0, self.max_len - len(audio)), mode='constant')
        else:
            audio = audio[:self.max_len]
            
        # Feature: Spectral Entropy of LP Residual
        seg = audio[:32000] # Use first 2s
        if len(seg) > 17:
            a = librosa.lpc(seg, order=16)
            res = np.convolve(seg, a, mode='same')
            spec = np.abs(np.fft.fft(res, n=512))**2
            psd = spec[:256] / (np.sum(spec[:256]) + 1e-10)
            ent = entropy(psd)
            h_r = ent / np.log(len(psd))
        else:
            h_r = 0.5
        
        return torch.tensor(audio, dtype=torch.float32), \
               torch.tensor(0 if label_str == 'bonafide' else 1).long(), \
               torch.tensor(h_r).float()

# --- 3. Model (Fixed v3.1) ---
class SincConv_fast(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super(SincConv_fast, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0: self.kernel_size += 1
        self.sample_rate = sample_rate
        
        self.low_hz_ = nn.Parameter(torch.Tensor(out_channels).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(out_channels).view(-1, 1))
        
        low_hz = 30 + (np.linspace(0, sample_rate/2 - 50, out_channels))
        band_hz = np.linspace(50, sample_rate/2 - 50, out_channels)
        self.low_hz_.data = torch.tensor(low_hz).float().view(-1, 1) / sample_rate
        self.band_hz_.data = torch.tensor(band_hz).float().view(-1, 1) / sample_rate
        
        # Window size is strictly integer division
        n_lin = torch.linspace(0, (self.kernel_size//2)-1, self.kernel_size//2)
        self.window_ = 0.54 - 0.46 * torch.cos(2 * 3.14159 * n_lin / self.kernel_size)
        self.register_buffer('window', self.window_.float())

    def forward(self, waveforms):
        low = 50 + torch.abs(self.low_hz_) * self.sample_rate
        high = torch.clamp(low + 50 + torch.abs(self.band_hz_) * self.sample_rate, 50, self.sample_rate/2)
        band = (high-low)[:,0]
        
        # FIX: Force integer range to match window size exactly
        n = torch.arange(0, self.kernel_size//2).to(waveforms.device).float()
        
        f_times_t_low = torch.matmul(low, (2*3.14159 * n / self.sample_rate).view(1,-1))
        f_times_t_high = torch.matmul(high, (2*3.14159 * n / self.sample_rate).view(1,-1))

        denom = 2*3.14159 * n / self.sample_rate
        denom[0] = 1.0 
        
        band_pass_left = ((torch.sin(f_times_t_high)-torch.sin(f_times_t_low)) / denom.view(1,-1)) * self.window.view(1,-1)
        band_pass_left[:, 0] = 2.0 * band 

        band_pass_center = 2*band.view(-1,1)
        band_pass_right = torch.flip(band_pass_left,dims=[1])
        filters = torch.cat([band_pass_left, band_pass_center, band_pass_right],dim=1)
        filters = filters.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, filters, stride=10, padding=self.kernel_size//2)

class SI_ENN(nn.Module):
    def __init__(self):
        super(SI_ENN, self).__init__()
        self.sinc_layer = SincConv_fast(70, 129)
        self.bn_sinc = nn.BatchNorm1d(70)
        self.layer1 = nn.Sequential(nn.Conv1d(70, 32, 3, padding=1), nn.BatchNorm1d(32), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.LeakyReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 2)

    def forward(self, x, h_r):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = F.max_pool1d(F.leaky_relu(self.bn_sinc(self.sinc_layer(x))), 3)
        x = F.max_pool1d(self.layer1(x), 3)
        x = F.max_pool1d(self.layer2(x), 3)
        x = F.max_pool1d(self.layer3(x), 3)
        z = self.pool(x).squeeze(-1)
        logits = self.fc(z)
        
        mu_ref, sigma = 0.65, 0.15
        h_r = h_r.to(logits.device)
        g_b = torch.exp(- (h_r - mu_ref)**2 / (2 * sigma**2))
        evidence = F.softplus(logits) * torch.stack([g_b, 1.0-g_b], dim=1)
        return evidence + 1

# --- 4. Run Experiment ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n--- 2. Starting Training on {DEVICE} ---")

dataset = ASVSpoof2019LADataset(flac_dir=found_flac, protocol_file=found_proto)
print(f"Dataset Loaded. Total samples: {len(dataset)}")

# Debug Subset (100 samples to be super fast)
indices = np.arange(100) 
train_loader = DataLoader(Subset(dataset, indices), batch_size=16, shuffle=True)

model = SI_ENN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("Training 1 Debug Epoch (100 samples)...")
model.train()
for i, (audio, labels, h_r) in enumerate(train_loader):
    audio, labels, h_r = audio.to(DEVICE), labels.to(DEVICE), h_r.to(DEVICE)
    optimizer.zero_grad()
    alpha = model(audio, h_r)
    
    # MSE Loss
    S = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / S
    y_hot = F.one_hot(labels, num_classes=2).float()
    loss = torch.mean(torch.sum((y_hot - prob)**2, dim=1))
    
    loss.backward()
    optimizer.step()
    
    if i % 2 == 0: # Print often for debug
        print(f"Step {i}, Loss: {loss.item():.4f}")

print("SUCCESS: Debug training run complete.")