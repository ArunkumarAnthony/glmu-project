chemprop predict \
    --test-path data/processed/prediction_set_repurposing_final.csv \
    --model-path models/finetune_v6_rdkit_scaffold/replicate_*/model_0/best.pt \
    --preds-path results/v6_repurposing_predictions.csv \
    --molecule-featurizers v1_rdkit_2d_normalized