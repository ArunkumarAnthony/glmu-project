
# scripts/smiles_to_pdb.py
import pandas as pd
import os
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit import Chem
from rdkit.Chem import AllChem

# 1. File Paths
input_csv = 'data/processed/prediction_set_repurposing_final.csv'
output_dir = 'docking/ligands_pdb'

# 2. Helper function to clean file names
def clean_filename(name, fallback_id):
    if pd.isna(name) or str(name).strip() == "" or str(name).lower() == 'nan':
        safe_name = str(fallback_id)
    else:
        safe_name = str(name)
    return re.sub(r'[\\/*?:"<>|]', "_", safe_name)

# 3. The Worker Function (Runs on separate CPU cores)
# We must pass strings, not RDKit objects, across processes to avoid memory crashes
def process_molecule(args):
    smiles, output_path = args
    
    # Skip if already done
    if os.path.exists(output_path):
        return True
        
    try:
        # Step A: Create 2D Mol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
            
        # Step B: Add explicit Hydrogens
        mol = Chem.AddHs(mol)
        
        # Step C: Generate 3D coordinates
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if res == -1:
            return False
            
        # Step D: Energy Minimization
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Step E: Save
        Chem.MolToPDBFile(mol, output_path)
        return True
        
    except Exception:
        return False

# 4. Main Execution Block (Required for multiprocessing in Python)
if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    total_mols = len(df)
    
    # Prepare the arguments for the workers
    print("Preparing tasks...")
    tasks = []
    for index, row in df.iterrows():
        smiles = row['smiles']
        file_name = clean_filename(row['pert_iname'], row['Sample'])
        output_path = os.path.join(output_dir, f"{file_name}.pdb")
        tasks.append((smiles, output_path))
        
    # Determine number of CPU cores to use (leave 1 core free so the PC doesn't freeze)
    max_cores = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"\n🚀 Launching Multi-CPU Processing on {max_cores} cores...")
    print(f"Total molecules to process: {total_mols}\n")
    
    success_count = 0
    fail_count = 0
    processed_count = 0

    # Execute in parallel
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        # Map tasks to the pool and wrap in as_completed to track progress
        futures = {executor.submit(process_molecule, task): task for task in tasks}
        
        for future in as_completed(futures):
            processed_count += 1
            result = future.result()
            
            if result:
                success_count += 1
            else:
                fail_count += 1
                
            # Print update every 500 molecules
            if processed_count % 500 == 0 or processed_count == total_mols:
                print(f"Progress: {processed_count}/{total_mols} | Success: {success_count} | Fails: {fail_count}")

    print("\n Multi-CPU 3D Conversion Complete!")
    print(f"Successfully generated: {success_count} PDB files")
    print(f"Failed to generate: {fail_count} (Due to invalid 3D geometry constraints)")