#!/bin/bash

for i in {0..9}
do
  chemprop train \
    --data-path data/processed/glmu_chemprop_training_set.csv \
    --task-type classification \
    --smiles-columns smiles \
    --target-columns activity \
    --molecule-featurizers v1_rdkit_2d_normalized \
    --checkpoint models/finetune_v6_rdkit_scaffold/replicate_${i}/model_0/best.pt \
    --save-dir models/glmu_transfer_ensemble_freeze/replicate_${i} \
    --epochs 10 \
    --init-lr 1e-5 \
    --metrics roc prc f1 binary-mcc \
    --class-balance \
    --split scaffold_balanced \
    --freeze-encoder \
    --num-replicates 1
done