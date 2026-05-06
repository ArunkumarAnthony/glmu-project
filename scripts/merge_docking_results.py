import pandas as pd
import sys
from pathlib import Path

# Dynamically set paths to the 'results' folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# UPDATE 1: Point to the strictly filtered v6 AI candidates
DL_RESULTS_FILE = PROJECT_ROOT / "results" / "v6_final_consensus_hits_clean.csv"
DOCKING_RESULTS_FILE = PROJECT_ROOT / "data" / "processed" / "psef_docking_results.csv"
# UPDATE 2: Update the output file name
OUTPUT_FILE = PROJECT_ROOT / "results" / "v6_final_consensus_hits_docking_merged.csv"

def merge_consensus(dl_file, docking_file, output_path):
    print("--- Merging v6 Elite AI Hits with Docking Energies ---")

    if not dl_file.exists():
        print(f"Error: Missing AI file at {dl_file}")
        sys.exit(1)
    if not docking_file.exists():
        print(f"Error: Missing Docking file at {docking_file}")
        sys.exit(1)

    try:
        df_dl = pd.read_csv(dl_file)
        df_docking = pd.read_csv(docking_file)

        # Perform the merge
        df_consensus = pd.merge(
            df_dl,
            df_docking[['Ligand', 'Binding Energy (kcal/mol)']],
            left_on='pert_iname',
            right_on='Ligand',
            how='inner'
        )

        # Clean up redundant column
        if 'Ligand' in df_consensus.columns:
            df_consensus = df_consensus.drop(columns=['Ligand'])

        # Sort the results primarily by the strongest physical binding energy (lowest/most negative)
        if 'Binding Energy (kcal/mol)' in df_consensus.columns:
            df_consensus = df_consensus.sort_values(by='Binding Energy (kcal/mol)', ascending=True)

        df_consensus.to_csv(output_path, index=False)
        print("\n" + "="*65)
        print("             FINAL CONSENSUS CANDIDATES")
        print("="*65)
        print(f"Successfully matched {len(df_consensus)} elite AI compounds with Docking Data!")
        print("="*65)
        print(f"Consensus hit list saved to: {output_path}\n")

        print("Top 5 Ultimate Candidates:")
        print(df_consensus.head().to_string(index=False))

    except Exception as e:
        print(f"An error occurred during the merge: {e}")

if __name__ == "__main__":
    merge_consensus(DL_RESULTS_FILE, DOCKING_RESULTS_FILE, OUTPUT_FILE)