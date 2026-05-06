import pandas as pd
import numpy as np

print("Processing raw GlmU ChEMBL data using the paper's binary methodology...")

# 1. Load the raw data you just dumped
df_raw = pd.read_csv('glmu_raw_chembl_dump.csv')

# 2. Isolate the essential columns
df = df_raw[['molecule_chembl_id', 'canonical_smiles', 'standard_type', 'standard_value', 'standard_units']].copy()

# 3. Clean missing data
df = df.dropna(subset=['canonical_smiles', 'standard_value'])
df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
df = df.dropna(subset=['standard_value'])

# 4. Standardize all units to micromolar (uM)
def convert_to_uM(row):
    if row['standard_units'] == 'nM':
        return row['standard_value'] / 1000
    elif row['standard_units'] == 'uM':
        return row['standard_value']
    else:
        return np.nan # Ignore strange units to maintain data purity

df['value_uM'] = df.apply(convert_to_uM, axis=1)
df = df.dropna(subset=['value_uM'])

# 5. Handle duplicates (if a drug was tested in 5 different papers, take the average IC50)
df_grouped = df.groupby('canonical_smiles').agg({'value_uM': 'mean'}).reset_index()

# 6. Apply the paper's Binary Classification Logic 
# Active (1) <= 10 uM, Inactive (0) > 10 uM
df_grouped['activity'] = df_grouped['value_uM'].apply(lambda x: 1 if x <= 10 else 0)

# 7. Format exactly for Chemprop (just smiles and activity columns)
df_final = df_grouped.rename(columns={'canonical_smiles': 'smiles'})[['smiles', 'activity']]

actives = len(df_final[df_final['activity'] == 1])
inactives = len(df_final[df_final['activity'] == 0])

print(f"\nProcessing Complete!")
print(f"Total unique compounds ready for training: {len(df_final)}")
print(f"-> Actives (Label 1): {actives}")
print(f"-> Inactives (Label 0): {inactives}")

# 8. Save the perfectly formatted training set
df_final.to_csv('glmu_chemprop_training_set.csv', index=False)
print("Saved to: glmu_chemprop_training_set.csv")