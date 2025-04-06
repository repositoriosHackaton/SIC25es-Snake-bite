import librosa.feature
import librosa.display
import pickle
import os

import pandas as pd
import numpy as np

data_path = 'filtered_voice'

df = pd.read_csv('filtered_voice_train.tsv', sep='\t')

sample_rate = 16000
n_mfcc = 13
mfcc_data = []

for index,row in df.iterrows():
    audio_path = os.path.join(data_path, row['path']+'.mp3')
    if os.path.exists(audio_path):
        y, sr = librosa.load(audio_path, sr=sample_rate)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_data.append({"mfcc": mfccs, "sentence": row['sentence']})

for entry in mfcc_data:
    mfcc = entry["mfcc"]
    if mfcc.shape[1] < 200:
        pad_width = 200 - mfcc.shape[1]
        mfcc = np.pad(mfcc,((0,0), (0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :200]
    entry["mfcc"] = mfcc

with open('mfcc.pkl', 'wb') as f:
    pickle.dump(mfcc_data, f)