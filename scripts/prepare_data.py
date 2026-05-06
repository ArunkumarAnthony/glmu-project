import pandas as pd
import numpy as np
import os

# Ensure the output directory exists
os.makedirs('data/processed', exist_ok=True)

print("--- Processing Pre-Training Data (General Antimicrobial) ---")
try:
    # Read the actual Excel files for 3 & 4, and the CSV for 5
    df3 = pd.read_excel('data/raw/Dataset_3.xlsx')
    df4 = pd.read_excel('data/raw/Dataset_4.xlsx')
    df5 = pd.read_csv('data/raw/Dataset_5.csv')
except FileNotFoundError:
    print("Error: Could not find the raw files in data/raw/")
    exit()

# Combine and extract SMILES and Outcome
df_pre = pd.concat([df3, df4, df5], ignore_index=True)
df_pre = df_pre[['PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME']].dropna()
df_pre.rename(columns={'PUBCHEM_EXT_DATASOURCE_SMILES': 'smiles'}, inplace=True)

# Binarize: Active = 1, Inactive = 0
df_pre['activity'] = df_pre['PUBCHEM_ACTIVITY_OUTCOME'].apply(
    lambda x: 1 if str(x).strip().lower() == 'active' else 0
)
df_pre = df_pre[['smiles', 'activity']]

# Deduplicate
df_pre = df_pre.groupby('smiles', as_index=False).max()
df_pre.to_csv('data/processed/pre_train_ext.csv', index=False)
print(f"Saved {len(df_pre)} unique pre-training molecules to data/processed/pre_train_ext.csv")


print("\n--- Processing Fine-Tuning Data (H. pylori Specific) ---")
# Dataset 1: MIC values
df1 = pd.read_excel('data/raw/Dataset_1.xlsx')
df1 = df1[['SMILE', 'MIC']].dropna()
df1.rename(columns={'SMILE': 'smiles'}, inplace=True)

def parse_mic(val):
    val = str(val).replace('＞', '').replace('>', '').replace('<', '').replace('≤', '').strip()
    try:
        return float(val)
    except ValueError:
        return np.nan

df1['mic_value'] = df1['MIC'].apply(parse_mic)
df1['activity'] = df1['mic_value'].apply(lambda x: 1 if x <= 4.0 else 0)
df1 = df1[['smiles', 'activity']]

# Dataset 2: ChEMBL ug/mL values
df2 = pd.read_excel('data/raw/Dataset_2.xlsx', header=None, names=['ID', 'smiles', 'Relation', 'Value', 'Unit'])
df2['Value'] = pd.to_numeric(df2['Value'], errors='coerce')
df2 = df2.dropna(subset=['smiles', 'Value'])
df2['activity'] = df2['Value'].apply(lambda x: 1 if x <= 4.0 else 0)
df2 = df2[['smiles', 'activity']]

# Combine and Deduplicate
df_fine = pd.concat([df1, df2], ignore_index=True)
df_fine = df_fine.groupby('smiles', as_index=False).max() 
df_fine.to_csv('data/processed/fine_tune_ext.csv', index=False)
print(f"Saved {len(df_fine)} unique H. pylori fine-tuning molecules to data/processed/fine_tune_ext.csv")
print("\nData preparation complete! You are ready to train.")