# scripts/plot_consensus_scatter.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup Publication Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

# 2. Load the Final Merged Data
input_file = 'results/v6_thresholded_compounds_docking_merged.csv'
df = pd.read_csv(input_file)

# Sort by Docking Energy (More negative is better)
df = df.sort_values(by='Binding Energy (kcal/mol)', ascending=True).reset_index(drop=True)

# Separate the Top 5 from the Rest
top_5 = df.head(5)
the_rest = df.iloc[5:]

# 3. Create the Plot
plt.figure(figsize=(10, 8))

# Plot the bulk of the elite candidates
sns.scatterplot(
    x='activity', 
    y='Binding Energy (kcal/mol)', 
    data=the_rest, 
    color='#90A4AE', # Subtle Blue/Gray
    s=60, 
    alpha=0.6, 
    edgecolor='k',
    label='Elite Candidates'
)

# Highlight the Top 5 Candidates
sns.scatterplot(
    x='activity', 
    y='Binding Energy (kcal/mol)', 
    data=top_5, 
    color='#D32F2F', # Bold Red
    s=150, 
    marker='*', 
    edgecolor='k',
    label='Top 5 Ultimate Hits'
)

# Annotate the Top 5 with their names
for i, row in top_5.iterrows():
    plt.text(
        row['activity'] + 0.002, 
        row['Binding Energy (kcal/mol)'] - 0.1, 
        str(row['pert_iname']), 
        fontsize=10, 
        fontweight='bold',
        color='black'
    )

# Formatting
plt.title('Figure 5: Consensus Hit Distribution', fontweight='bold', fontsize=18, pad=15)
plt.xlabel('AI Predicted Probability (Higher is better)', fontweight='bold')
plt.ylabel('AutoDock Binding Energy (kcal/mol) (Lower is better)', fontweight='bold')
plt.legend(loc='upper right', frameon=True, shadow=True)

# Invert Y-axis so the BEST (most negative) scores are at the top!
plt.gca().invert_yaxis()

# Save
os.makedirs('results', exist_ok=True)
output_file = 'results/Figure_5_Consensus_Plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Figure 5 saved to {output_file}")