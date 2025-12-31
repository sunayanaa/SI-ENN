# 18_plot_modulation_curve.py
# Purpose: Generates the 'Fig_Modulation.png' for Section III.
# Fix: Force-mounts drive and ensures directory exists.

import numpy as np
import matplotlib.pyplot as plt
import os
from google.colab import drive

# 1. MOUNT DRIVE (Force remount to fix connection issues)
print("Mounting Drive...")
drive.mount('/content/drive', force_remount=True)

# 2. DEFINE PATHS & CREATE FOLDER
project_root = '/content/drive/MyDrive/ASVspoof_Project'
if not os.path.exists(project_root):
    os.makedirs(project_root)
    print(f"Created directory: {project_root}")

save_path = os.path.join(project_root, 'Fig_Modulation.png')

# 3. GENERATE DATA (Parameters from Section IV-A of paper)
mu = 0.77
sigma = 0.08
h = np.linspace(0.0, 1.2, 500)
g_b = np.exp(-((h - mu)**2) / (2 * sigma**2))

# 4. PLOT
plt.figure(figsize=(8, 5))

# Plot Gaussian Curve
plt.plot(h, g_b, color='#2ca02c', linewidth=3, label=r'Bonafide Weight $g_b(H)$')
plt.axvline(mu, color='black', linestyle=':', alpha=0.5, label=r'Reference $\mu \approx 0.77$')

# Annotate "Danger Zones"
plt.text(0.40, 0.25, 'Deterministic\nSpoofs\n(Old TTS)', color='red', ha='center', fontsize=10, fontweight='bold')
plt.text(1.10, 0.25, 'Stochastic\nSpoofs\n(Neural)', color='red', ha='center', fontsize=10, fontweight='bold')

# Shade the suppression regions
plt.axvspan(0.2, 0.55, color='red', alpha=0.1) # Low entropy zone
plt.axvspan(0.95, 1.2, color='red', alpha=0.1) # High entropy zone

# Styling
plt.title(r'Signal-Informed Modulation Function $g_b(H)$', fontsize=14)
plt.xlabel('LP Residual Spectral Entropy ($H$)', fontsize=12)
plt.ylabel('Evidence Weight (Bonafide)', fontsize=12)
plt.legend(loc='upper right', fontsize=11, frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0.2, 1.2)
plt.ylim(0, 1.1)

# 5. SAVE
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nSUCCESS: Figure saved to: {save_path}")
plt.show()