# 00_setup.py
# Version: 1.0
# Installs necessary libraries and sets up the environment for the experiment.

!pip install -q librosa torch numpy matplotlib seaborn scikit-learn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy.stats import entropy

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Setup Complete. Libraries installed.")
