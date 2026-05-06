import pandas as pd

print("--- Applying Final Strict Thresholds ---")

# Load the previously generated novel scaffold hits
# (This file already has the Tanimoto similarities calculated)
input_file = 'results/v6_novel_scaffold_hits.csv'
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: Could not find {input_file}. Make sure you are in the correct directory.")
    exit()

initial_count = len(df)
print(f"Loaded {initial_count} molecules from {input_file}")

# Apply Threshold 1: Prediction Score > 0.2
df_filtered = df[df['activity'] > 0.2]
passed_activity = len(df_filtered)
print(f"Passed Prediction Score > 0.2 : {passed_activity}")

# Apply Threshold 2: Tanimoto Similarity < 0.3
df_final = df_filtered[df_filtered['nearest_neighbor_similarity'] < 0.3]
passed_tanimoto = len(df_final)

# Sort by prediction score (highest first)
df_final = df_final.sort_values(by='activity', ascending=False)

# Clean up column names just in case you still have the messy merge output (pert_iname_x)
if 'pert_iname_x' in df_final.columns:
    df_final = df_final.rename(columns={'pert_iname_x': 'pert_iname', 'broad_id_x': 'broad_id'})

# Keep only the essential columns for your final paper
columns_to_keep = ['smiles', 'pert_iname', 'broad_id', 'activity', 'nearest_neighbor_similarity']
existing_cols = [c for c in columns_to_keep if c in df_final.columns]
df_final = df_final[existing_cols]

# Save the final selected compounds
output_file = 'results/v6_thresholded_compounds.csv'
df_final.to_csv(output_file, index=False)

print("\n" + "="*55)
print("             FINAL ELITE CANDIDATES")
print("="*55)
print(f"Total novel scaffolds analyzed : {initial_count}")
print(f"Final compounds selected       : {passed_tanimoto}")
print("="*55)
print(f"Saved strictly filtered candidates to {output_file}\n")

print("Top 5 Elite Candidates:")
print(df_final.head().to_string(index=False))