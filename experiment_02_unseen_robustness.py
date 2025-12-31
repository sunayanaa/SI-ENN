# 15_curriculum_unseen_test.py
# Version: 2.0 (Curriculum Learning Strategy)
# Purpose: Stage 1 (Softmax Features) -> Stage 2 (Freeze & Calibrate) to beat Baseline.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
import librosa
from scipy.stats import entropy
from google.colab import drive
import matplotlib.pyplot as plt

# --- 1. SETUP ---
print("--- Setup: Curriculum Learning for Unseen Attack Robustness ---")
drive.mount('/content/drive')
project_root = '/content/drive/MyDrive/ASVspoof_Project'
ckpt_dir = os.path.join(project_root, 'checkpoints_curriculum')
os.makedirs(ckpt_dir, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. MODEL DEFINITIONS (Adaptive) ---
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

    def forward(self, x, h_r=None):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = F.max_pool1d(F.leaky_relu(self.bn_sinc(self.sinc_layer(x))), 3)
        x = F.max_pool1d(self.layer1(x), 3)
        x = F.max_pool1d(self.layer2(x), 3)
        x = F.max_pool1d(self.layer3(x), 3)
        z = self.pool(x).squeeze(-1)
        logits = self.fc(z)
        if self.mode == 'softmax': return logits 
        if h_r is None: return F.softplus(logits) + 1
        h_r = h_r.to(logits.device)
        sig_clamped = torch.clamp(torch.abs(self.sigma), min=0.01)
        g_b = torch.exp(- (h_r - self.mu_ref)**2 / (2 * sig_clamped**2))
        e_b = F.softplus(logits[:, 0]) * g_b
        e_s = F.softplus(logits[:, 1]) * (1.0 - g_b)
        return torch.stack([e_b, e_s], dim=1) + 1

# --- 3. ROBUST DATASET ---
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

# Paths
proto_path = os.path.join(project_root, 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
flac_path = os.path.join(project_root, 'data/LA/ASVspoof2019_LA_train/flac')

# --- 4. PREPARE DATA ---
print("\n>>> Preparing Data (A01-A03 Known, A04-A06 Unseen)...")
train_ds = AttackSplitDataset(flac_path, proto_path, target_attacks=['A01', 'A02', 'A03'])
bf_idx = train_ds.df[train_ds.df['Label'] == 'bonafide'].index.tolist()
sp_idx = train_ds.df[train_ds.df['Label'] == 'spoof'].index.tolist()
n_bf = min(len(bf_idx), 2000) 
n_sp = min(len(sp_idx), 2000)
train_idx = np.random.choice(bf_idx, n_bf, replace=False).tolist() + np.random.choice(sp_idx, n_sp, replace=False).tolist()
train_loader = DataLoader(torch.utils.data.Subset(train_ds, [train_ds.df.index.get_loc(i) for i in train_idx]), batch_size=32, shuffle=True)

test_ds = AttackSplitDataset(flac_path, proto_path, target_attacks=['A04', 'A05', 'A06'])
bf_test = test_ds.df[test_ds.df['Label'] == 'bonafide'].index.tolist()
sp_test = test_ds.df[test_ds.df['Label'] == 'spoof'].index.tolist()
test_idx = np.random.choice(bf_test, 500, replace=False).tolist() + np.random.choice(sp_test, 500, replace=False).tolist()
test_loader = DataLoader(torch.utils.data.Subset(test_ds, [test_ds.df.index.get_loc(i) for i in test_idx]), batch_size=32, shuffle=False)

# --- 5. TWO-STAGE TRAINING ---

def edl_loss(alpha, y):
    S = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / S
    y_hot = torch.eye(2).to(DEVICE)[y]
    loss_mse = torch.sum((y_hot - p)**2, dim=1, keepdim=True) + torch.sum(alpha*(S-alpha)/((S*S)*(S+1)), dim=1, keepdim=True)
    alp_tilde = y_hot + (1-y_hot)*alpha
    kl = torch.lgamma(alp_tilde.sum(1)) - torch.lgamma(torch.tensor(2.0).to(DEVICE)) - torch.sum(torch.lgamma(alp_tilde),1)
    return torch.mean(loss_mse + 0.05 * kl)

def train_curriculum(name, epochs_stage1=8, epochs_stage2=8):
    print(f"\n>>> Training {name} (Curriculum Strategy)...")
    
    # Checkpoint Path
    ckpt_path = os.path.join(ckpt_dir, f"{name.replace(' ', '_')}_ckpt.pth")
    model = SI_ENN_Adaptive(mode='softmax').to(DEVICE) # Start in Softmax Mode
    
    # STAGE 1: FEATURE LEARNING (Softmax)
    print("   [Stage 1] Training Backbone with CrossEntropy (Softmax)...")
    opt = optim.Adam(model.parameters(), lr=1e-4)
    
    # Resume Logic (Simplified for Stage 1)
    start_epoch = 0
    if os.path.exists(ckpt_path):
        print(f"   [Resume] Found checkpoint, checking stage...")
        checkpoint = torch.load(ckpt_path)
        if checkpoint['stage'] == 1:
            model.load_state_dict(checkpoint['model_state'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"   [Resume] Resuming Stage 1 from Epoch {start_epoch}")
        elif checkpoint['stage'] == 2:
            print("   [Resume] Stage 1 complete. Jumping to Stage 2...")
            start_epoch = epochs_stage1 + 1 # Force skip
    
    if start_epoch < epochs_stage1:
        model.train()
        for i in range(start_epoch, epochs_stage1):
            el = 0
            for aud, y, hr in train_loader:
                aud, y, hr = aud.to(DEVICE), y.to(DEVICE), hr.to(DEVICE)
                opt.zero_grad()
                out = model(aud, hr)
                loss = nn.CrossEntropyLoss()(out, y) # Standard Loss
                loss.backward()
                opt.step()
                el += loss.item()
            print(f"   [Stage 1] Epoch {i+1}/{epochs_stage1}: Loss {el/len(train_loader):.4f}")
            torch.save({'stage': 1, 'epoch': i, 'model_state': model.state_dict()}, ckpt_path)

    # STAGE 2: UNCERTAINTY CALIBRATION (EDL)
    print("\n   [Stage 2] Switching to SI-ENN Mode (Freezing Backbone)...")
    model.mode = 'adaptive' # Switch to SI-ENN
    
    # Freeze Feature Extractor
    for name, param in model.named_parameters():
        if "fc" in name or "mu_ref" in name or "sigma" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Optimizer only for Head + Adaptive Params
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())
    opt = optim.Adam(params_to_update, lr=1e-4)
    
    # Resume Logic for Stage 2
    start_epoch = 0
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        if checkpoint['stage'] == 2:
            model.load_state_dict(checkpoint['model_state'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"   [Resume] Resuming Stage 2 from Epoch {start_epoch}")
        elif checkpoint['stage'] == 1:
            model.load_state_dict(checkpoint['model_state']) # Load Stage 1 weights
            
    model.train()
    for i in range(start_epoch, epochs_stage2):
        el = 0
        for aud, y, hr in train_loader:
            aud, y, hr = aud.to(DEVICE), y.to(DEVICE), hr.to(DEVICE)
            opt.zero_grad()
            out = model(aud, hr)
            loss = edl_loss(out, y) # EDL Loss
            loss.backward()
            opt.step()
            el += loss.item()
        print(f"   [Stage 2] Epoch {i+1}/{epochs_stage2}: Loss {el/len(train_loader):.4f}")
        torch.save({'stage': 2, 'epoch': i, 'model_state': model.state_dict()}, ckpt_path)
        
    return model

# --- 6. METRICS ---
def get_predictions_and_uncertainty(model, loader, mode):
    model.eval()
    preds, uncerts, gts = [], [], []
    with torch.no_grad():
        for aud, y, hr in loader:
            aud, y, hr = aud.to(DEVICE), y.to(DEVICE), hr.to(DEVICE)
            out = model(aud, hr)
            if mode == 'softmax':
                probs = torch.softmax(out, dim=1)
                u = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
                p_spoof = probs[:, 1]
            else:
                S = torch.sum(out, dim=1)
                u = 2 / S 
                p_spoof = out[:, 1] / S
            preds.extend(p_spoof.cpu().numpy())
            uncerts.extend(u.cpu().numpy())
            gts.extend(y.cpu().numpy())
    return np.array(preds), np.array(uncerts), np.array(gts)

def compute_risk_coverage(preds, uncerts, gts):
    sorted_indices = np.argsort(uncerts)[::-1]
    gts_sorted = gts[sorted_indices]
    preds_sorted = preds[sorted_indices]
    risks, coverages = [], []
    n = len(preds)
    for i in range(0, n, int(n/20)): 
        remaining_indices = sorted_indices[i:] 
        if len(remaining_indices) == 0: break
        sub_gts = gts[remaining_indices]
        sub_preds = preds[remaining_indices]
        hard_preds = (sub_preds > 0.5).astype(int)
        risk = np.mean(hard_preds != sub_gts)
        coverage = len(remaining_indices) / n
        risks.append(risk)
        coverages.append(coverage)
    return coverages, risks

# --- 7. EXECUTE ---
# Train Baseline (Pure Softmax)
print("\n>>> Training Baseline (Standard)...")
base_model = SI_ENN_Adaptive(mode='softmax').to(DEVICE)
base_opt = optim.Adam(base_model.parameters(), lr=1e-4)
for i in range(10): # 10 Epochs standard
    el=0
    for aud, y, hr in train_loader:
        aud, y, hr = aud.to(DEVICE), y.to(DEVICE), hr.to(DEVICE)
        base_opt.zero_grad()
        out = base_model(aud, hr)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        base_opt.step()
        el+=loss.item()
    print(f"   Baseline Epoch {i+1}: {el/len(train_loader):.4f}")

# Train SI-ENN (Curriculum)
sienn_model = train_curriculum("Curriculum SI-ENN", epochs_stage1=8, epochs_stage2=8)

print("\n>>> Computing Risk-Coverage on UNSEEN Attacks (A04-A06)...")
p_base, u_base, y_base = get_predictions_and_uncertainty(base_model, test_loader, 'softmax')
p_si, u_si, y_si = get_predictions_and_uncertainty(sienn_model, test_loader, 'adaptive')

cov_base, risk_base = compute_risk_coverage(p_base, u_base, y_base)
cov_si, risk_si = compute_risk_coverage(p_si, u_si, y_si)

def get_risk_at_50(cov, risk):
    idx = (np.abs(np.array(cov) - 0.5)).argmin()
    return risk[idx]

r50_base = get_risk_at_50(cov_base, risk_base)
r50_si = get_risk_at_50(cov_si, risk_si)

print("\n" + "="*50 + "\nUNSEEN ATTACK ROBUSTNESS RESULTS\n" + "="*50)
print(f"Risk @ 50% Coverage (Baseline) : {r50_base:.4f}")
print(f"Risk @ 50% Coverage (SI-ENN)   : {r50_si:.4f}")
print("-" * 50)
if r50_si < r50_base:
    print(f"SUCCESS: Rejection reduces error to {r50_si*100:.2f}%.")
else:
    print("INCONCLUSIVE.")
print("="*50)