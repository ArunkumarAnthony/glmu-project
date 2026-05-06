#!/bin/bash
chemprop train \
  --data-path data/processed/fine_tune_ext.csv \
  --task-type classification \
  --smiles-columns smiles \
  --target-columns activity \
  --molecule-featurizers v1_rdkit_2d_normalized \
  --save-dir models/finetune_v6_scratch_scaffold \
  --epochs 50 \
  --class-balance \
  --split scaffold_balanced \
  --num-replicates 10