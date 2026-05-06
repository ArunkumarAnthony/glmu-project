# scripts/plot_validation_figures.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

# 1. Setup Publication Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)
colors = ['#B0BEC5', '#78909C', '#FFB74D', '#42A5F5', '#D32F2F'] # Final model is bold red
line_weights = [2, 2, 2, 2, 3.5] # Final model gets a thicker line

# 2. Load Ground Truth
df_truth = pd.read_csv("data/processed/fine_tune_ext.csv")
df_truth = df_truth[['smiles', 'activity']].rename(columns={'activity': 'true_activity'})

experiments = {
    "Train from Scratch": "models/finetune_v6_scratch_scaffold/replicate_*/model_0/test_predictions.csv",
    "Base MPNN (No RDKit)": "models/finetune_v6_base_scaffold/replicate_*/model_0/test_predictions.csv",
    "Random Split": "models/finetune_v6_rdkit_random/replicate_*/model_0/test_predictions.csv",
    "Single Model": "models/finetune_v6_rdkit_single/model_0/test_predictions.csv",
    "Final model": "models/finetune_v6_rdkit_scaffold/replicate_*/model_0/test_predictions.csv"
}

# 3. Create Separate Figures
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
fig_prc, ax_prc = plt.subplots(figsize=(8, 6))
fig_cal, ax_cal = plt.subplots(figsize=(8, 6))
fig_bar, ax_bar = plt.subplots(figsize=(10, 6))

# Variables to store data for the Bar Chart and Calibration
bar_labels = []
bar_roc = []
bar_prc = []
final_y_true = []
final_y_prob = []

print("--- Generating Publication Validation Figures ---")

# 4. Loop through experiments to draw curves
for idx, (name, path_pattern) in enumerate(experiments.items()):
    files = glob.glob(path_pattern)
    if not files:
        print(f" Missing files for {name}")
        continue
        
    y_true_all = []
    y_prob_all = []
    
    # Concatenate predictions from all folds to get smooth, comprehensive curves
    for f in files:
        df_pred = pd.read_csv(f)[['smiles', 'activity']].rename(columns={'activity': 'pred_prob'})
        df_merged = pd.merge(df_pred, df_truth, on='smiles', how='inner')
        y_true_all.extend(df_merged['true_activity'].values)
        y_prob_all.extend(df_merged['pred_prob'].values)
        
    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)
    
    # Save Final Model data for Calibration Curve
    if name == "Final model":
        final_y_true = y_true_all
        final_y_prob = y_prob_all

    # --- Plot A: ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=colors[idx], lw=line_weights[idx], label=f'{name} (AUC = {roc_auc:.3f})')
    
    # --- Plot B: PRC Curve ---
    precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all)
    prc_auc = average_precision_score(y_true_all, y_prob_all)
    ax_prc.plot(recall, precision, color=colors[idx], lw=line_weights[idx], label=f'{name} (AUC = {prc_auc:.3f})')
    
    # Save data for Bar Chart
    bar_labels.append(name)
    bar_roc.append(roc_auc)
    bar_prc.append(prc_auc)
    print(f" Processed curves for {name}")

# Formatting Plot A (ROC)
ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate', fontweight='bold')
ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
ax_roc.set_title('A. Receiver Operating Characteristic (ROC)', fontweight='bold', fontsize=16)
ax_roc.legend(loc="lower right")

# Formatting Plot B (PRC)
baseline = sum(final_y_true) / len(final_y_true) if len(final_y_true) > 0 else 0.5
ax_prc.axhline(baseline, color='k', linestyle='--', lw=2, alpha=0.5, label=f'Baseline ({baseline:.2f})')
ax_prc.set_xlim([0.0, 1.0])
ax_prc.set_ylim([0.0, 1.05])
ax_prc.set_xlabel('Recall (Sensitivity)', fontweight='bold')
ax_prc.set_ylabel('Precision (PPV)', fontweight='bold')
ax_prc.set_title('B. Precision-Recall Curve (PRC)', fontweight='bold', fontsize=16)
ax_prc.legend(loc="lower left")

# --- Plot C: Calibration Curve ---
if len(final_y_true) > 0:
    prob_true, prob_pred = calibration_curve(final_y_true, final_y_prob, n_bins=10)
    ax_cal.plot(prob_pred, prob_true, 's-', color='#D32F2F', lw=3, label='Final model')
    ax_cal.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly Calibrated')
    
    ax_cal.set_xlim([0.0, 1.0])
    ax_cal.set_ylim([0.0, 1.0])
    ax_cal.set_xlabel('Mean Predicted Probability', fontweight='bold')
    ax_cal.set_ylabel('Fraction of True Positives', fontweight='bold')
    ax_cal.set_title('C. Calibration Plot (Reliability Diagram)', fontweight='bold', fontsize=16)
    ax_cal.legend(loc="lower right")

# --- Plot D: Ablation Bar Chart ---
x = np.arange(len(bar_labels))
width = 0.35

rects1 = ax_bar.bar(x - width/2, bar_roc, width, label='ROC-AUC', color='#5C6BC0')
rects2 = ax_bar.bar(x + width/2, bar_prc, width, label='PRC-AUC', color='#26A69A')

ax_bar.set_ylabel('Area Under Curve (AUC)', fontweight='bold')
ax_bar.set_title('D. Ablation Performance Summary', fontweight='bold', fontsize=16)
ax_bar.set_xticks(x)
# Wrap long labels
clean_labels = [l.replace(" ", "\n") if len(l) > 15 else l for l in bar_labels]
ax_bar.set_xticklabels(clean_labels, rotation=0)
ax_bar.legend(loc='lower right')
ax_bar.set_ylim([0.0, 1.05])

# Add text labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax_bar.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
autolabel(rects1)
autolabel(rects2)

# 5. Finalize and Save Separate Files
os.makedirs('results', exist_ok=True)

# Save Plot A
fig_roc.tight_layout()
fig_roc.savefig('results/Figure_4A_ROC.png', dpi=300, bbox_inches='tight')

# Save Plot B
fig_prc.tight_layout()
fig_prc.savefig('results/Figure_4B_PRC.png', dpi=300, bbox_inches='tight')

# Save Plot C
fig_cal.tight_layout()
fig_cal.savefig('results/Figure_4C_Calibration.png', dpi=300, bbox_inches='tight')

# Save Plot D
fig_bar.tight_layout()
fig_bar.savefig('results/Figure_4D_Ablation_Bar.png', dpi=300, bbox_inches='tight')

print("\n Figure generation complete! Saved 4 separate stunning 300 DPI images to the 'results' directory.")