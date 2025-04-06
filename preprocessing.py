import os
import shutil
import pandas as pd

dataset_path = 'data/voices/'
tsv_path = os.path.join(dataset_path, 'validated.tsv')
output_path = os.path.join(dataset_path, 'filtered')

os.makedirs(output_path, exist_ok=True)
df = pd.read_csv(tsv_path, sep='\t')

df_filtered = df[(df['sentence'].str.split().str.len() <= 5) & (df['down_votes'] == 0)]
df_filtered = df_filtered.sample(n=min(10000, len(df_filtered)), random_state=42)
df_filtered = df_filtered[['path','sentence']]

for filename in df_filtered['path']:
    path = filename + '.mp3'
    src = os.path.join(dataset_path, 'clips', path)
    dst = os.path.join(output_path, path)
    if os.path.exists(src):
        shutil.copy(src, dst)

path_filtered = os.path.join(output_path, 'filtered_train.tsv')
df_filtered.to_csv(path_filtered, sep='\t', index=False)