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

print("--- Starting Novel Scaffold Discovery (Abaucin Nature Methodology) ---")

# 1. Load the known active molecules (Training Data)
print("Loading reference H. pylori actives...")
df_train = pd.read_csv('data/processed/fine_tune_ext.csv')
known_actives_smiles = df_train[df_train['activity'] == 1]['smiles'].tolist()
active_fps, _ = compute_morgan_fingerprints(known_actives_smiles)

print(f"Successfully generated fingerprints for {len(active_fps)} known active molecules.")

# 2. Load the predicted screening results 
# (Chemprop already kept pert_iname for us, so no need to merge!)
print("\nLoading screening predictions...")
df_preds = pd.read_csv('results/v6_repurposing_predictions.csv')

# Ensure we sort by the highest predicted probability
df_preds = df_preds.sort_values(by='activity', ascending=False).reset_index(drop=True)

# 3. Take the Top 500 highest confidence predictions to screen for novelty
top_n = 500
df_top = df_preds.head(top_n).copy()
print(f"Analyzing the Top {top_n} predicted hits for structural novelty...")

top_smiles = df_top['smiles'].tolist()
top_fps, valid_indices = compute_morgan_fingerprints(top_smiles)

# Drop any smiles that RDKit couldn't parse
df_top = df_top.iloc[valid_indices].reset_index(drop=True)

# 4. Calculate Pairwise Tanimoto Similarities
print("Computing pairwise Tanimoto similarities using matrix operations...")
tanimoto_distances = pairwise_distances(top_fps, active_fps, metric='jaccard', n_jobs=-1)
similarities = 1 - tanimoto_distances

# Find the maximum similarity to ANY known active for each predicted molecule
df_top['nearest_neighbor_similarity'] = np.max(similarities, axis=1)

# 5. Apply the Novelty Threshold (< 0.4 Similarity)
df_novel = df_top[df_top['nearest_neighbor_similarity'] < 0.4]

# Keep only the essential columns to make it clean
columns_to_keep = ['smiles', 'pert_iname', 'Sample', 'activity', 'nearest_neighbor_similarity']
# Check if pert_iname exists, just to be extremely safe
if 'pert_iname' in df_novel.columns:
    df_novel = df_novel[columns_to_keep]

# Save the final hits
output_file = 'results/v6_novel_scaffold_hits.csv'
df_novel.to_csv(output_file, index=False)

print("\n" + "="*60)
print(f"Initial Top Predictions Analyzed : {len(df_top)}")
print(f"Structurally Redundant Hits Dropped: {len(df_top) - len(df_novel)}")
print(f"True NOVEL Scaffolds Discovered    : {len(df_novel)}")
print("="*60)
print(f"Saved clean candidates to {output_file}\n")

print("Top 5 Novel Scaffold Candidates:")
print(df_novel.head().to_string(index=False))