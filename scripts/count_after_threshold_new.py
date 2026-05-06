import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

def compute_morgan_fingerprints(smiles_list, radius=2, num_bits=2048):
    """Generates binary Morgan fingerprints for a list of SMILES."""
    fps = []
    valid_indices = []
    
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(s))
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
            arr = np.zeros((1,), dtype=bool)
            from rdkit import DataStructs
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid_indices.append(i)
            
    return np.array(fps), valid_indices

print("============================================================")
print("       VIRTUAL SCREENING ATTRITION CASCADE CALCULATION      ")
print("============================================================\n")

# ---------------------------------------------------------
# STEP 0: Load Initial Data
# ---------------------------------------------------------
print("Loading reference H. pylori actives...")
df_train = pd.read_csv('data/processed/fine_tune_ext.csv')
known_actives_smiles = df_train[df_train['activity'] == 1]['smiles'].tolist()
active_fps, _ = compute_morgan_fingerprints(known_actives_smiles)

print("Loading raw AI screening predictions...")
df_preds = pd.read_csv('results/v6_repurposing_predictions.csv')
initial_count = len(df_preds)
print(f"-> STARTING LIBRARY SIZE: {initial_count} compounds\n")

# ---------------------------------------------------------
# STEP 1: Apply AI Confidence Threshold (> 0.80)
# ---------------------------------------------------------
print("Applying strict AI confidence threshold (> 0.80)...")
df_confident = df_preds[df_preds['activity'] > 0.80].reset_index(drop=True)
confident_count = len(df_confident)
dropped_by_ai = initial_count - confident_count

print(f"-> Dropped {dropped_by_ai} low-confidence predictions.")
print(f"-> REMAINING CONFIDENT HITS: {confident_count} compounds\n")

# ---------------------------------------------------------
# STEP 2: Apply Structural Novelty Threshold (< 0.30)
# ---------------------------------------------------------
print("Computing Tanimoto similarities against known actives for confident hits...")
confident_smiles = df_confident['smiles'].tolist()
confident_fps, valid_indices = compute_morgan_fingerprints(confident_smiles)

# Drop any smiles RDKit failed to parse
df_confident = df_confident.iloc[valid_indices].reset_index(drop=True)

# Calculate matrix distances
tanimoto_distances = pairwise_distances(confident_fps, active_fps, metric='jaccard', n_jobs=-1)
similarities = 1 - tanimoto_distances
df_confident['nearest_neighbor_similarity'] = np.max(similarities, axis=1)

print("Applying strict structural novelty threshold (Tc < 0.30)...")
df_final = df_confident[df_confident['nearest_neighbor_similarity'] < 0.30]
final_count = len(df_final)
dropped_by_tanimoto = len(df_confident) - final_count

print(f"-> Dropped {dropped_by_tanimoto} structurally redundant/known compounds.")
print(f"-> REMAINING NOVEL SCAFFOLDS: {final_count} compounds\n")

# ---------------------------------------------------------
# SUMMARY OUTPUT
# ---------------------------------------------------------
print("============================================================")
print("                     FINAL ATTRITION SUMMARY                ")
print("============================================================")
print(f"1. Initial Virtual Screening Library : {initial_count}")
print(f"2. Passed AI Confidence (> 0.80)     : {confident_count}")
print(f"3. Passed Structural Novelty (< 0.30): {final_count}")
print("============================================================")

# Save this perfect, pristine list
df_final.to_csv('results/v6_final_consensus_hits_clean.csv', index=False)
print("Saved final consensus list to: results/v6_final_consensus_hits_clean.csv")