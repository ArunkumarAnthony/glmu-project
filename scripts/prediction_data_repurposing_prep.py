# [Cell] Clean SMILES and Map pert_iname to Repurposing Set
import pandas as pd
import os

# Define file paths
raw_screen_path = 'data/raw/prediction_set_repurposing.csv'
ref_lib_path = 'data/raw/screening_library.csv'
output_path = 'data/processed/prediction_set_repurposing_final.csv'

# Handle relative paths for Jupyter Notebook
if not os.path.exists(raw_screen_path):
    raw_screen_path = '../' + raw_screen_path
    ref_lib_path = '../' + ref_lib_path
    output_path = '../' + output_path

# 1. Load both datasets
df_target = pd.read_csv(raw_screen_path)
df_ref = pd.read_csv(ref_lib_path)

# ==========================================
# 2. THE CLEANING STEP (Remove CXSMILES tags)
# ==========================================
# This splits the string by whitespace and keeps only the first part (the pure SMILES)
df_target['SMILES'] = df_target['SMILES'].astype(str).str.split().str[0]

# 3. Extract Base Broad IDs (first 13 chars, e.g., BRD-K63001556)
df_target['base_id'] = df_target['Sample'].astype(str).str[:13]
df_ref['base_id'] = df_ref['broad_id'].astype(str).str[:13]

# 4. Clean the reference library to prevent duplicate merging
df_ref_unique = df_ref.drop_duplicates(subset=['base_id'])

# 5. Perform the Merge
df_merged = pd.merge(df_target, df_ref_unique[['base_id', 'pert_iname']], on='base_id', how='left')

# Drop the temporary base_id column
df_merged = df_merged.drop(columns=['base_id'])

# Rename 'SMILES' to 'smiles' just so it is completely standard for Chemprop
df_merged = df_merged.rename(columns={'SMILES': 'smiles'})

# 6. Save the output
df_merged.to_csv(output_path, index=False)

# 7. Print summary
total_mols = len(df_merged)
mapped_mols = df_merged['pert_iname'].notna().sum()

print(f"Library Cleaned and Mapped Successfully!")
print(f"Total molecules ready for screening: {total_mols:,}")
print(f"Successfully mapped a 'pert_iname': {mapped_mols:,}")
print(f"\nSaved new pristine file to: {output_path}")