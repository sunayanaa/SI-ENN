# 12_run_adaptive_ablation.py
# Version: 2.0 (Adaptive Parameter Strategy)
# Purpose: Introduces Learnable Mu/Sigma to solve the "Static Prior" failure.

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
import copy

# --- 1. SETUP ---
print("--- Setup: Adaptive Ablation Study ---")
drive.mount('/content/drive')
project_root = '/content/drive/MyDrive/ASVspoof_Project'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. ADAPTIVE MODEL DEFINITION ---
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
        
        # Backbone
        self.sinc_layer = SincConv_fast(70, 129)
        self.bn_sinc = nn.BatchNorm1d(70)
        self.layer1 = nn.Sequential(nn.Conv1d(70, 32, 3, padding=1), nn.BatchNorm1d(32), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.LeakyReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 2) 
        
        # --- KEY INNOVATION: LEARNABLE PARAMETERS ---
        # We initialize at 0.77 and 0.08, but allow gradients to move them!
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
        if self.mode == 'pure_edl': return F.softplus(logits) + 1
        
        # ADAPTIVE MODULATION
        if h_r is None: return F.softplus(logits) + 1
        
        h_r = h_r.to(logits.device)
        
        # Use the LEARNABLE mu_ref and sigma
        # Clamp sigma to avoid division by zero or negative spread
        sig_clamped = torch.clamp(torch.abs(self.sigma), min=0.01)
        g_b = torch.exp(- (h_r - self.mu_ref)**2 / (2 * sig_clamped**2))
        
        e_b = F.softplus(logits[:, 0]) * g_b
        e_s = F.softplus(logits[:, 1]) * (1.0 - g_b)
        return torch.stack([e_b, e_s], dim=1) + 1

# --- 3. DATA & SUBSET ---
class ASVSpoofDataset(Dataset):
    def __init__(self, flac_dir, protocol_file, max_len=64000):
        self.flac_dir = flac_dir
        self.max_len = max_len
        self.df = pd.read_csv(protocol_file, sep=' ', header=None, names=['Speaker','Filename','System','Null','Label'])
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

proto_path = os.path.join(project_root, 'data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
flac_path = os.path.join(project_root, 'data/LA/ASVspoof2019_LA_train/flac')

print("Creating Balanced Subset...")
full_ds = ASVSpoofDataset(flac_path, proto_path)
bonafide_indices = full_ds.df[full_ds.df['Label'] == 'bonafide'].index.tolist()
spoof_indices = full_ds.df[full_ds.df['Label'] == 'spoof'].index.tolist()
n_per_class = 250
balanced_idx = np.random.choice(bonafide_indices, n_per_class, replace=False).tolist() + \
               np.random.choice(spoof_indices, n_per_class, replace=False).tolist()
train_loader = DataLoader(torch.utils.data.Subset(full_ds, balanced_idx), batch_size=16, shuffle=True)
print(f"Subset created: {len(balanced_idx)} samples")

# --- 4. PRE-TRAINING ---
print("\n>>> Phase 1: Pre-training Backbone (Softmax) - 15 Epochs...")
base_model = SI_ENN_Adaptive(mode='softmax').to(DEVICE)
opt = optim.Adam(base_model.parameters(), lr=1e-4)
for epoch in range(15):
    epoch_loss = 0
    for aud, y, hr in train_loader:
        aud, y, hr = aud.to(DEVICE), y.to(DEVICE), hr.to(DEVICE)
        opt.zero_grad()
        out = base_model(aud, hr)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
warm_weights = copy.deepcopy(base_model.state_dict())
print(">>> Warm-up Complete.")

# --- 5. FINE-TUNING LOOP ---
def edl_loss(alpha, y):
    S = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / S
    y_hot = torch.eye(2).to(DEVICE)[y]
    loss_mse = torch.sum((y_hot - p)**2, dim=1, keepdim=True) + torch.sum(alpha*(S-alpha)/((S*S)*(S+1)), dim=1, keepdim=True)
    alp_tilde = y_hot + (1-y_hot)*alpha
    kl = torch.lgamma(alp_tilde.sum(1)) - torch.lgamma(torch.tensor(2.0).to(DEVICE)) - torch.sum(torch.lgamma(alp_tilde),1)
    return torch.mean(loss_mse + 0.05 * kl)

def run_adaptive_test(name, mode, epochs=20, freeze_mu=True):
    print(f"\n>>> Running: {name} (Freeze Mu={freeze_mu})...")
    model = SI_ENN_Adaptive(mode=mode).to(DEVICE)
    model.load_state_dict(warm_weights) # Load Warm Start
    
    # SETUP OPTIMIZER
    params = list(model.parameters())
    
    # If using Static (Old) method, we must freeze mu_ref so it doesn't learn
    if freeze_mu and mode == 'adaptive': 
        model.mu_ref.requires_grad = False
        model.sigma.requires_grad = False
    elif mode == 'adaptive':
        print(f"   [Adaptive Mode]: Mu/Sigma are LEARNABLE. Starting Mu={model.mu_ref.item():.3f}")
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, params), lr=1e-5)
    model.train()
    
    for epoch in range(epochs):
        for aud, y, hr in train_loader:
            aud, y, hr = aud.to(DEVICE), y.to(DEVICE), hr.to(DEVICE)
            optimizer.zero_grad()
            out = model(aud, hr)
            if mode == 'softmax': loss = nn.CrossEntropyLoss()(out, y)
            else: loss = edl_loss(out, y)
            loss.backward()
            optimizer.step()
            
    if mode == 'adaptive' and not freeze_mu:
        print(f"   [Result]: Mu shifted to {model.mu_ref.item():.4f}")
        
    return model

# --- 6. EXECUTE & EVAL ---
def evaluate_metric(model, loader, mode):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for aud, y, hr in loader:
            aud, y, hr = aud.to(DEVICE), y.to(DEVICE), hr.to(DEVICE)
            out = model(aud, hr)
            if mode == 'softmax': prob = torch.softmax(out, dim=1)[:, 1]
            else:
                S = torch.sum(out, dim=1)
                prob = (out[:, 1] / S)
            preds.extend(prob.cpu().numpy())
            gts.extend(y.cpu().numpy())
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(gts, preds, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[eer_idx]

results = {}

# 1. Baseline
m1 = run_adaptive_test("Baseline (Softmax)", "softmax")
results['Softmax'] = evaluate_metric(m1, train_loader, 'softmax')

# 2. Static SI-ENN (Old Proposed)
m2 = run_adaptive_test("Static SI-ENN (Fixed Mu=0.77)", "adaptive", freeze_mu=True)
results['Static_SI_ENN'] = evaluate_metric(m2, train_loader, 'adaptive')

# 3. Adaptive SI-ENN (New Proposed)
m3 = run_adaptive_test("Adaptive SI-ENN (Learnable Mu)", "adaptive", freeze_mu=False)
results['Adaptive_SI_ENN'] = evaluate_metric(m3, train_loader, 'adaptive')

print("\n" + "="*40 + "\nADAPTIVE ABLATION RESULTS\n" + "="*40)
print(f"1. Baseline (Softmax)   : {results['Softmax']*100:.2f}%")
print(f"2. Static SI-ENN        : {results['Static_SI_ENN']*100:.2f}%")
print(f"3. Adaptive SI-ENN      : {results['Adaptive_SI_ENN']*100:.2f}%")
print("="*40)
print("Note: If Adaptive < Static, it confirms learning helps alignment.")