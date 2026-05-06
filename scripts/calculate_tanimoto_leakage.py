import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import warnings

# Suppress RDKit warnings for cleaner output
warnings.filterwarnings("ignore")

def get_max_tanimoto(test_smiles_list, train_smiles_list):
    """Calculates the average maximum similarity of test molecules to the training set."""
    # Convert SMILES to 2D Morgan Fingerprints
    test_fps = []
    for s in test_smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            test_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            
    train_fps = []
    for s in train_smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            
    # For every test molecule, find its nearest neighbor in the training set
    max_sims = []
    for test_fp in test_fps:
        sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        max_sims.append(max(sims))
        
    return np.mean(max_sims), np.std(max_sims)

# 1. Load the full pool of molecules
df_all = pd.read_csv("data/processed/fine_tune_ext.csv")
all_smiles = set(df_all['smiles'].tolist())

# 2. Extract Random Split (Replicate 0) Test vs Train
print("Calculating Data Leakage for Model 3 (Random Split)...")
df_rand_test = pd.read_csv("models/finetune_v6_rdkit_random/replicate_0/model_0/test_predictions.csv")
rand_test_smiles = set(df_rand_test['smiles'].tolist())
rand_train_smiles = all_smiles - rand_test_smiles
rand_mean, rand_std = get_max_tanimoto(list(rand_test_smiles), list(rand_train_smiles))

# 3. Extract Scaffold Split (Replicate 0) Test vs Train
print("Calculating Data Leakage for Model 5 (Scaffold Split)...")
df_scaf_test = pd.read_csv("models/finetune_v6_rdkit_scaffold/replicate_0/model_0/test_predictions.csv")
scaf_test_smiles = set(df_scaf_test['smiles'].tolist())
scaf_train_smiles = all_smiles - scaf_test_smiles
scaf_mean, scaf_std = get_max_tanimoto(list(scaf_test_smiles), list(scaf_train_smiles))

# 4. Print the final proof for the manuscript
print("\n" + "="*50)
print("   Applicability Domain (Data Leakage) Analysis")
print("="*50)
print(f"Model 3 (Random Split) Test-to-Train Similarity   : {rand_mean:.4f} ± {rand_std:.4f}")
print(f"Model 5 (Scaffold Split) Test-to-Train Similarity : {scaf_mean:.4f} ± {scaf_std:.4f}")
print("="*50)
print("Conclusion: A lower similarity in the Scaffold split proves the model")
print("was tested on genuinely novel chemistry (Scaffold Hopping), making its")
print("validation metrics far more reliable for real-world drug discovery.")