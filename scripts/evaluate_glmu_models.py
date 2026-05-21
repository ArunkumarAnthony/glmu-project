import pandas as pd
import numpy as np
import glob
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)
from scipy.stats import mannwhitneyu

# ============================================================
# Load Ground Truth
# ============================================================

df_truth = pd.read_csv(
    "data/processed/glmu_chemprop_training_set.csv"
)

df_truth = df_truth[["smiles", "activity"]].rename(
    columns={"activity": "true_activity"}
)

# ============================================================
# Experiment Definitions
# ============================================================

experiments = {
    "Antibiotics → GLmU": (
        "models/glmu_from_basev6_nohpylori/"
        "replicate_*/model_0/test_predictions.csv"
    ),
    "Antibiotics → H.pylori → GLmU (Freeze)": (
        "models/glmu_transfer_ensemble_freeze/"
        "replicate_*/model_0/test_predictions.csv"
    ),
    "Antibiotics → H.pylori → GLmU (Unfreeze)": (
        "models/glmu_transfer_ensemble_no_freeze/"
        "replicate_*/model_0/test_predictions.csv"
    ),
}

# ============================================================
# Config
# ============================================================

THRESHOLD = 0.5

results = []
raw_roc = {}

# ============================================================
# Evaluation Loop
# ============================================================

for experiment_name, path_pattern in experiments.items():

    files = sorted(glob.glob(path_pattern))

    if not files:
        print(f"\n  No files found for: {experiment_name}")
        continue

    roc_scores    = []
    prc_scores    = []
    f1_scores     = []
    mcc_scores    = []
    sensitivity_scores = []
    specificity_scores = []

    print(f"\n{'='*60}")
    print(f"  {experiment_name}")
    print(f"  Found {len(files)} replicate(s)")
    print(f"{'='*60}")

    for file in files:

        df_pred = pd.read_csv(file)
        df_pred = df_pred[["smiles", "activity"]].rename(
            columns={"activity": "pred_prob"}
        )

        df_merged = pd.merge(
            df_truth,
            df_pred,
            on="smiles",
            how="inner"
        )

        if df_merged.empty:
            print(f"  Empty merge for {file}")
            continue

        y_true = df_merged["true_activity"].values
        y_prob = df_merged["pred_prob"].values
        y_pred = (y_prob >= THRESHOLD).astype(int)

        roc = roc_auc_score(y_true, y_prob)
        prc = average_precision_score(y_true, y_prob)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        roc_scores.append(roc)
        prc_scores.append(prc)
        f1_scores.append(f1)
        mcc_scores.append(mcc)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)

        replicate_label = file.split("/")[-3]
        print(
            f"  {replicate_label:<20}  "
            f"ROC={roc:.4f}  PRC={prc:.4f}  "
            f"MCC={mcc:.4f}  F1={f1:.4f}  "
            f"Sens={sensitivity:.4f}  Spec={specificity:.4f}"
        )

    print(f"\n  ROC per replicate : {[round(x, 4) for x in roc_scores]}")
    print(f"  PRC per replicate : {[round(x, 4) for x in prc_scores]}")
    print(f"  MCC per replicate : {[round(x, 4) for x in mcc_scores]}")

    raw_roc[experiment_name] = roc_scores

    def mean_std(values):
        return f"{np.mean(values):.4f} ± {np.std(values):.4f}"

    results.append({
        "Experiment":  experiment_name,
        "Replicates":  len(roc_scores),
        "ROC-AUC":     mean_std(roc_scores),
        "PRC-AUC":     mean_std(prc_scores),
        "F1-Score":    mean_std(f1_scores),
        "MCC":         mean_std(mcc_scores),
        "Sensitivity": mean_std(sensitivity_scores),
        "Specificity": mean_std(specificity_scores),
    })

# ============================================================
# Summary Table
# ============================================================

df_results = pd.DataFrame(results)

print("\n")
print("=" * 80)
print("GLmU Transfer Learning Evaluation")
print("=" * 80)
print(df_results.to_markdown(index=False))

# ============================================================
# Statistical Tests (Mann-Whitney U)
# ============================================================

print("\n")
print("=" * 80)
print("Statistical Comparisons (Mann-Whitney U, two-sided)")
print("=" * 80)

experiment_names = list(raw_roc.keys())

comparisons = [
    (experiment_names[0], experiment_names[1]),
    (experiment_names[0], experiment_names[2]),
    (experiment_names[1], experiment_names[2]),
]

for name_a, name_b in comparisons:
    if name_a not in raw_roc or name_b not in raw_roc:
        continue

    a = raw_roc[name_a]
    b = raw_roc[name_b]

    stat, p = mannwhitneyu(a, b, alternative="two-sided")

    mean_a = np.mean(a)
    mean_b = np.mean(b)
    direction = ">" if mean_a > mean_b else "<"

    significance = (
        "***" if p < 0.001 else
        "**"  if p < 0.01  else
        "*"   if p < 0.05  else
        "ns"
    )

    short_a = name_a.split("→")[-1].strip()
    short_b = name_b.split("→")[-1].strip()

    print(
        f"  {short_a:<30} {direction}  {short_b:<30} "
        f"p={p:.4f}  U={stat:.1f}  {significance}"
    )

print("\n  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")

# ============================================================
# Variance Summary
# ============================================================

print("\n")
print("=" * 80)
print("Variance Summary (ROC-AUC std across replicates)")
print("=" * 80)

for name, scores in raw_roc.items():
    short = name.split("→")[-1].strip()
    std   = np.std(scores)
    label = (
        "low variance  ✓" if std < 0.05 else
        "moderate"        if std < 0.10 else
        "high variance !"
    )
    print(f"  {short:<40}  std={std:.4f}  [{label}]")

# ============================================================
# Save Results
# ============================================================

output_path = "results/glmu_transfer_learning_results.csv"
df_results.to_csv(output_path, index=False)
print(f"\n  Results saved to: {output_path}\n")