import pandas as pd

r0 = pd.read_csv("models/glmu_from_basev6_nohpylori/replicate_0/model_0/test_predictions.csv")
r4 = pd.read_csv("models/glmu_from_basev6_nohpylori/replicate_4/model_0/test_predictions.csv")

overlap = set(r0["smiles"]) & set(r4["smiles"])
print(f"Replicate 0 test size : {len(r0)}")
print(f"Replicate 4 test size : {len(r4)}")
print(f"Overlapping molecules : {len(overlap)}")

# also check freeze replicate for comparison
f0 = pd.read_csv("models/glmu_transfer_ensemble_freeze/replicate_0/model_0/test_predictions.csv")
f1 = pd.read_csv("models/glmu_transfer_ensemble_freeze/replicate_1/model_0/test_predictions.csv")
freeze_overlap = set(f0["smiles"]) & set(f1["smiles"])
print(f"\nFreeze replicate 0 test size : {len(f0)}")
print(f"Freeze replicate 1 test size : {len(f1)}")
print(f"Overlapping molecules        : {len(freeze_overlap)}")