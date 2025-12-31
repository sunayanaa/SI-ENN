# 01_verify_protocols.py
# Version: 1.0
# Description: Verifies that the label files (protocols) are readable. Use this if programmatic download of LA.zip worked.

import pandas as pd
import os

base_path = '/content/drive/MyDrive/ASVspoof_Project/data/LA/ASVspoof2019_LA_cm_protocols'
train_proto_file = os.path.join(base_path, 'ASVspoof2019.LA.cm.train.trn.txt')

if os.path.exists(train_proto_file):
    print(f"Reading protocol: {train_proto_file}")
    
    # The protocol file has no header. Columns: SPEAKER_ID, AUDIO_FILE_NAME, SYSTEM_ID, KEY, KEY (Bonafide/Spoof)
    df_train = pd.read_csv(train_proto_file, sep=' ', header=None, names=['Speaker', 'Filename', 'System', 'Null', 'Label'])
    
    print("\nFirst 5 rows:")
    print(df_train.head())
    
    print(f"\nTotal Samples: {len(df_train)}")
    print(f"Bonafide Count: {len(df_train[df_train['Label'] == 'bonafide'])}")
    print(f"Spoof Count: {len(df_train[df_train['Label'] == 'spoof'])}")
else:
    print("Protocol file not found! Check extraction path.")