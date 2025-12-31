# 17_tsne_visualization.py
# Version: 1.0
# Purpose: Generates t-SNE plots to visualize how SI-ENN separates Unseen Attacks vs Bonafide.

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import librosa
from scipy.stats import entropy
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE

# --- SETUP ---
print("--- Generating Final Visual: t-SNE Embeddings ---")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_root = '/content/drive/MyDrive/ASVspoof_Project'
ckpt_dir = os.path.join(project_root, 'checkpoints_curriculum')

# --- MODEL DEFINITIONS (Must match previous) ---
class SincConv_fast(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super(SincConv_fast, self).__init__()
        self.out_channels, self.sample_rate = out_channels, sample_rate
        self.kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        self.low_hz_ = nn.Parameter(torch.Tensor(out_channels).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(out_channels).view(-1, 1))
        low_hz = 30 + (np.linspace(0, sample_rate/2 - 50, out_channels))
        band_hz = np.linspace(50, sample_rate/2 - 50, out_channels)
        self.low_hz_.data = torch.tensor(low_hz).float().view(-1, 1) / sample_rate
        self.band_hz_.data = torch.tensor(band_hz).float().view(-1, 1) / sample_rate
        n_lin = torch.linspace(0, (self.kernel_size//2)-1, self.kernel_size//2)
        self.window_ = 0.54 - 0.46 * torch.cos(2 * 3.14159 * n_lin / self.kernel_size)
        self.register_buffer('window', self.window_.float())
    def forward(self, waveforms):
        low = 50 + torch.abs(self.low_hz_) * self.sample_rate
        high = torch.clamp(low + 50 + torch.abs(self.band_hz_) * self.sample_rate, 50, self.sample_rate/2)
        band = (high-low)[:,0]
        n = torch.arange(0, self.kernel_size//2).to(waveforms.device).float()
        f_times_t_low = torch.matmul(low, (2*3.14159 * n / self.sample_rate).view(1,-1))
        f_times_t_high = torch.matmul(high, (2*3.14159 * n / self.sample_rate).view(1,-1))
        denom = 2*3.14159 * n / self.sample_rate
        denom[0] = 1.0 
        band_pass_left = ((torch.sin(f_times_t_high)-torch.sin(f_times_t_low)) / denom.view(1,-1)) * self.window.view(1,-1)
        band_pass_left[:, 0] = 2.0 * band 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right = torch.flip(band_pass_left,dims=[1])
        filters = torch.cat([band_pass_left, band_pass_center, band_pass_right],dim=1).view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(waveforms, filters, stride=10, padding=self.kernel_size//2)

class SI_ENN_Adaptive(nn.Module):
    def __init__(self, mode='adaptive'):
        super(SI_ENN_Adaptive, self).__init__()
        self.mode = mode
        self.sinc_layer = SincConv_fast(70, 129)
        self.bn_sinc = nn.BatchNorm1d(70)
        self.layer1 = nn.Sequential(nn.Conv1d(70, 32, 3, padding=1), nn.BatchNorm1d(32), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.LeakyReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 2) 
        self.mu_ref = nn.Parameter(torch.tensor(0.77))
        self.sigma  = nn.Parameter(torch.tensor(0.08))

    def forward(self, x, h_r=None, return_feats=False):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = F.max_pool1d(F.leaky_relu(self.bn_sinc(self.sinc_layer(x))), 3)
        x = F.max_pool1d(self.layer1(x), 3)
        x = F.max_pool1d(self.layer2(x), 3)
        x = F.max_pool1d(self.layer3(x), 3)
        z = self.pool(x).squeeze(-1) # Features (Batch, 128)
        
        if return_feats: return z
        
        logits = self.fc(z)
        if self.mode == 'softmax': return logits 
        if h_r is None: return F.softplus(logits) + 1
        h_r = h_r.to(logits.device)
        sig_clamped = torch.clamp(torch.abs(self.sigma), min=0.01)
        g_b = torch.exp(- (h_r - self.mu_ref)**2 / (2 * sig_clamped**2))
        e_b = F.softplus(logits[:, 0]) * g_b
        e_s = F.softplus(logits[:, 1]) * (1.0 - g_b)
        return torch.stack([e_b, e_s], dim=1) + 1

# --- DATASET ---
class AttackSplitDataset(Dataset):
    def __init__(self, flac_dir, protocol_file, target_attacks=None, max_len=64000):
        self.flac_dir = flac_dir
        self.max_len = max_len
        full_df = pd.read_csv(protocol_file, sep=r'\s+', header=None, 
                             names=['Speaker','Filename','Environment','System','Label'],
                             dtype={'System': str, 'Label': str})
        full_df['System'] = full_df['System'].str.strip()
        full_df['Label'] = full_df['Label'].str.strip()
        if target_attacks:
            self.df = full_df[ (full_df['Label'] == 'bonafide') | (full_df['System'].isin(target_attacks)) ]
        else:
            self.df = full_df
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname, label = row['Filename'], row['Label']
        fpath = os.path.join(self.flac_dir, fname + '.flac')
        try: audio, _ = librosa.load(fpath, sr=16000)
        except: audio = np.zeros(self.max_len)
        if len(audio)<self.max_len: audio = np.pad(audio,(0,self.max_len-len(audio)))
        else: audio = audio[:self.max_len]
        seg = audio[:32000]
        if len(seg)>17:
            a = librosa.lpc(seg, order=16); res = np.convolve(seg,a,mode='same')
            spec = np.abs(np.fft.fft(res,n=512))**2; psd = spec[:256]/(np.sum(spec[:256])+1e-9)
            ent = entropy(psd); h_r = ent/np.log(len(psd))
        else: h_r=0.5
        return torch.tensor(audio).float(), torch.tensor(0 if label=='bonafide' else 1).long(), torch.tensor(h_r).float()

# --- LOAD DATA (Unseen Attacks) ---
proto_path = os.path.join(project_root, 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
flac_path = os.path.join(project_root, 'data/LA/ASVspoof2019_LA_train/flac')

print(">>> Preparing Data for t-SNE (A04-A06)...")
test_ds = AttackSplitDataset(flac_path, proto_path, target_attacks=['A04', 'A05', 'A06'])
bf_test = test_ds.df[test_ds.df['Label'] == 'bonafide'].index.tolist()
sp_test = test_ds.df[test_ds.df['Label'] == 'spoof'].index.tolist()
# Use 400 total points for clean t-SNE
idx = np.random.choice(bf_test, 200, replace=False).tolist() + np.random.choice(sp_test, 200, replace=False).tolist()
loader = DataLoader(torch.utils.data.Subset(test_ds, [test_ds.df.index.get_loc(i) for i in idx]), batch_size=32, shuffle=False)

# --- LOAD MODEL ---
print(">>> Loading SI-ENN Model...")
model = SI_ENN_Adaptive(mode='adaptive').to(DEVICE)
ckpt = torch.load(os.path.join(ckpt_dir, 'Curriculum_SI-ENN_ckpt.pth'))
model.load_state_dict(ckpt['model_state'])
model.eval()

# --- EXTRACT FEATURES & UNCERTAINTY ---
feats, uncerts, labels = [], [], []

with torch.no_grad():
    for aud, y, hr in loader:
        aud, y, hr = aud.to(DEVICE), y.to(DEVICE), hr.to(DEVICE)
        
        # 1. Get Features (z)
        z = model(aud, hr, return_feats=True)
        feats.extend(z.cpu().numpy())
        
        # 2. Get Uncertainty
        out = model(aud, hr)
        S = torch.sum(out, dim=1)
        u = 2 / S
        uncerts.extend(u.cpu().numpy())
        
        labels.extend(y.cpu().numpy())

feats = np.array(feats)
uncerts = np.array(uncerts)
labels = np.array(labels)

# --- RUN t-SNE ---
print(">>> Computing t-SNE (perplexity=30)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
z_embedded = tsne.fit_transform(feats)

# --- PLOT ---
print(">>> Plotting...")
plt.figure(figsize=(10, 8))

# Plot Bonafide (Blue circles)
bf_mask = (labels == 0)
plt.scatter(z_embedded[bf_mask, 0], z_embedded[bf_mask, 1], 
            c='blue', label='Bonafide', alpha=0.6, edgecolors='k', s=60)

# Plot Spoof (Red triangles), colored intensity by Uncertainty? 
# Actually, let's color ALL points by Uncertainty to see the "Bridge"
plt.figure(figsize=(12, 5))

# SUBPLOT 1: Ground Truth
plt.subplot(1, 2, 1)
plt.scatter(z_embedded[bf_mask, 0], z_embedded[bf_mask, 1], c='blue', label='Bonafide', alpha=0.5)
plt.scatter(z_embedded[~bf_mask, 0], z_embedded[~bf_mask, 1], c='red', label='Unseen Spoof (A04-A06)', alpha=0.5, marker='^')
plt.title("Ground Truth Classes")
plt.legend()
plt.xticks([]); plt.yticks([])

# SUBPLOT 2: Uncertainty Heatmap
plt.subplot(1, 2, 2)
sc = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=uncerts, cmap='viridis', alpha=0.8)
plt.colorbar(sc, label='Epistemic Uncertainty')
plt.title("SI-ENN Uncertainty Landscape")
plt.xticks([]); plt.yticks([])

save_path = os.path.join(project_root, 'Fig4_tSNE_Analysis.png')
plt.savefig(save_path, dpi=300)
print(f"\n>>> Figure saved to: {save_path}")
plt.show()