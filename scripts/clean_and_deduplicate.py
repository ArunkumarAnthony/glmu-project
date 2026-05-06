import pandas as pd
import sys
from pathlib import Path

# Dynamically set paths based on the project structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "screening_library.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "screening_library_final.csv"

def clean_and_deduplicate(input_path, output_path):
    print(f"Processing '{input_path.name}'...")
    
    if not input_path.exists():
        print(f"Error: Could not find input data at {input_path}")
        sys.exit(1)

    try:
        # Load the library
        df = pd.read_csv(input_path)
        initial_count = len(df)

        # 1. Strip CXSMILES tags (keeping only the standard SMILES part)
        df['smiles'] = df['smiles'].astype(str).str.split().str[0]

        # 2. Remove duplicates sequentially to ensure absolute uniqueness
        # First, drop any duplicate chemical structures
        df_unique = df.drop_duplicates(subset=['smiles'], keep='first')
        
        # Second, drop any duplicate compound names/IDs
        if 'pert_iname' in df_unique.columns:
            df_unique = df_unique.drop_duplicates(subset=['pert_iname'], keep='first')
        else:
            print("Warning: 'pert_iname' column not found. Deduplicated by SMILES only.")
            
        final_count = len(df_unique)

        # Save to the final screening file
        df_unique.to_csv(output_path, index=False)

        print(f"Initial compounds: {initial_count}")
        print(f"Duplicates/Salts removed: {initial_count - final_count}")
        print(f"Final unique compounds: {final_count}")
        print(f"Clean file saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    clean_and_deduplicate(INPUT_FILE, OUTPUT_FILE)