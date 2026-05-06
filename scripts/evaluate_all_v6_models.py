import pandas as pd
import numpy as np
import glob
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, confusion_matrix

# 1. Load the exact answers and rename the column so it doesn't clash
df_truth = pd.read_csv("data/processed/fine_tune_ext.csv")
df_truth = df_truth[['smiles', 'activity']].rename(columns={'activity': 'true_activity'})

experiments = {
    "1. Train from Scratch (No Pretrain)": "models/finetune_v6_scratch_scaffold/replicate_*/model_0/test_predictions.csv",
    "2. Base MPNN (No RDKit)": "models/finetune_v6_base_scaffold/replicate_*/model_0/test_predictions.csv",
    "3. Random Split": "models/finetune_v6_rdkit_random/replicate_*/model_0/test_predictions.csv",
    "4. Single Model (No Ensemble)": "models/finetune_v6_rdkit_single/model_0/test_predictions.csv",
    "5. Final Model (Pretrain + RDKit + Ensemble + scaffold split)": "models/finetune_v6_rdkit_scaffold/replicate_*/model_0/test_predictions.csv"
}

results = []
THRESHOLD = 0.5 # Standard threshold for binary metrics

for name, path_pattern in experiments.items():
    files = glob.glob(path_pattern)
    if not files:
        print(f"Could not find files for {name}")
        continue
        
    roc_aucs, prc_aucs = [], []
    f1_scores, accuracies = [], []
    sensitivities, specificities = [], []
    
    for f in files:
        # 2. Load predictions and rename the column
        df_pred = pd.read_csv(f)
        df_pred = df_pred[['smiles', 'activity']].rename(columns={'activity': 'pred_prob'})
        
        # 3. Safely merge based on the exact SMILES strings
        df_merged = pd.merge(df_pred, df_truth, on='smiles', how='inner')
        
        if df_merged.empty:
            continue
            
        y_true = df_merged['true_activity'].values
        y_prob = df_merged['pred_prob'].values
        
        # Convert probabilities to 0 or 1 based on threshold
        y_pred_bin = (y_prob >= THRESHOLD).astype(int)
        
        # Continuous Metrics (No threshold needed)
        roc_aucs.append(roc_auc_score(y_true, y_prob))
        prc_aucs.append(average_precision_score(y_true, y_prob))
        
        # Binary Classification Metrics
        f1_scores.append(f1_score(y_true, y_pred_bin))
        accuracies.append(accuracy_score(y_true, y_pred_bin))
        
        # Confusion Matrix for Sensitivity and Specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
        sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        
    if not roc_aucs:
        continue
        
    # Helper function to compute Mean ± Std Dev
    def format_metric(metric_list):
        mean_val = np.mean(metric_list)
        std_val = np.std(metric_list)
        return f"{mean_val:.4f} ± {std_val:.4f}" if len(metric_list) > 1 else f"{mean_val:.4f}"
    
    results.append({
        "Model Architecture": name,
        "Folds": len(roc_aucs),
        "ROC-AUC": format_metric(roc_aucs),
        "PRC-AUC": format_metric(prc_aucs),
        "F1-Score": format_metric(f1_scores),
        "Accuracy": format_metric(accuracies),
        "Sensitivity": format_metric(sensitivities),
        "Specificity": format_metric(specificities)
    })

# Create and print the Markdown Table
df_results = pd.DataFrame(results)
print("\n### Table 1: Model Validation and Ablation Study")
print(df_results.to_markdown(index=False))

# Save to CSV for your records
df_results.to_csv("results/v6_ablation_metrics.csv", index=False)
print("\n Saved to results/v6_ablation_metrics.csv")