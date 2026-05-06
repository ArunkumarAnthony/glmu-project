import pandas as pd
import sys
from pathlib import Path

# Dynamically set paths based on the project structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "repo-sample-annotation-20240610.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "screening_library.csv"

def prepare_library(input_path, output_path):
    print(f"Loading the Drug Repurposing Hub dataset from {input_path}...")
    
    if not input_path.exists():
        print(f"Error: Could not find raw data at {input_path}")
        sys.exit(1)

    try:
        # Load the file, handle tab spacing, and skip metadata
        df = pd.read_csv(input_path, sep='\t', comment='!', low_memory=False)
        
        # Select required columns and drop missing SMILES
        df_clean = df[['smiles', 'pert_iname', 'broad_id']].copy()
        initial_len = len(df_clean)
        df_clean = df_clean.dropna(subset=['smiles'])
        
        # Save to processed data folder
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        
        print(f"Dropped {initial_len - len(df_clean)} rows missing SMILES data.")
        print(f"Successfully prepared {len(df_clean)} compounds!")
        print(f"File saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred while processing the data: {e}")

if __name__ == "__main__":
    prepare_library(INPUT_FILE, OUTPUT_FILE)