# scripts/plot_top5_structures.py
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import os

# 1. Load the Final Merged Data
input_file = 'results/v6_thresholded_compounds_docking_merged.csv'
df = pd.read_csv(input_file)

# Sort by best docking energy
df = df.sort_values(by='Binding Energy (kcal/mol)', ascending=True).reset_index(drop=True)

# Get Top 5
top_5 = df.head(5)

mols = []
legends = []

# 2. Process the SMILES and create labels
print("--- Generating 2D Structures for Top 5 Hits ---")
for idx, row in top_5.iterrows():
    name = row['pert_iname']
    ai_score = row['activity']
    energy = row['Binding Energy (kcal/mol)']
    smiles = row['smiles']
    
    # Create RDKit Mol object
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mols.append(mol)
        # Create a detailed multi-line label for each molecule
        legend_text = f"{name}\nAI: {ai_score:.3f} | Dock: {energy:.2f} kcal/mol"
        legends.append(legend_text)

# 3. Draw the Grid
# MolsToGridImage creates a beautiful aligned grid automatically
img = Draw.MolsToGridImage(
    mols,
    molsPerRow=5, # Put all 5 in a single row for wide figures
    subImgSize=(400, 400), # High resolution for each molecule
    legends=legends,
    useSVG=False # Save as standard PNG
)

# 4. Save
os.makedirs('results', exist_ok=True)
output_file = 'results/Figure_6_Top5_Structures.png'

# Save the image object
img.save(output_file)
print(f"✅ Figure 6 saved to {output_file}")