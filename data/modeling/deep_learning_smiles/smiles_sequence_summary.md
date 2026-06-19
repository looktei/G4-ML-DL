# Phase 3A-2 SMILES Sequence Model

Generated: 2026-06-12T01:22:34

## Scope

- Model: PyTorch SMILES token embedding + CNN + bidirectional GRU classifier.
- Dataset: `/home/looktei/Desktop/test1/script/data/modeling/g4ldb_chembl_binary_dataset.csv`
- Rows: 9,344; positives: 4,672; negatives: 4,672.
- Input: canonical SMILES token sequence, max length 256 tokens.
- Architecture: embedding 128, CNN channels 96, BiGRU hidden 128, dense hidden 256, dropout 0.25.
- Device: `cuda`.
- Runtime: 22.8 minutes.
- Token length median/IQR/99th percentile: 61 / 47-79 / 196.
- Negative-label caveat remains: ChEMBL examples are assumed-negative decoys.

## Primary Results

| split    |   roc_auc |   pr_auc |   mcc |   brier_score |   balanced_accuracy |   sensitivity_recall |   specificity |   best_epoch |   vocab_size |   truncated_n |
|:---------|----------:|---------:|------:|--------------:|--------------------:|---------------------:|--------------:|-------------:|-------------:|--------------:|
| random   |     0.987 |    0.989 | 0.909 |         0.037 |               0.955 |                0.952 |         0.957 |           12 |          156 |            29 |
| scaffold |     0.974 |    0.975 | 0.839 |         0.068 |               0.919 |                0.899 |         0.939 |           27 |          167 |            29 |

## Repeated Split Robustness

| split    |   roc_auc_mean |   roc_auc_std |   pr_auc_mean |   pr_auc_std |   balanced_accuracy_mean |   balanced_accuracy_std |   mcc_mean |   mcc_std |   brier_score_mean |   brier_score_std |
|:---------|---------------:|--------------:|--------------:|-------------:|-------------------------:|------------------------:|-----------:|----------:|-------------------:|------------------:|
| random   |          0.986 |         0.002 |         0.987 |        0.002 |                    0.944 |                   0.007 |      0.889 |     0.014 |              0.046 |             0.006 |
| scaffold |          0.974 |         0.004 |         0.975 |        0.005 |                    0.901 |                   0.021 |      0.809 |     0.037 |              0.080 |             0.015 |

## Comparison With Earlier Baselines

| split    |   roc_auc_smiles |   pr_auc_smiles |   mcc_smiles |   brier_score_smiles |   balanced_accuracy_smiles |   roc_auc_rf |   pr_auc_rf |   mcc_rf |   brier_score_rf |   balanced_accuracy_rf |   roc_auc_mlp |   pr_auc_mlp |   mcc_mlp |   brier_score_mlp |   balanced_accuracy_mlp |   delta_roc_auc_smiles_minus_rf |   delta_roc_auc_smiles_minus_mlp |   delta_pr_auc_smiles_minus_rf |   delta_pr_auc_smiles_minus_mlp |   delta_mcc_smiles_minus_rf |   delta_mcc_smiles_minus_mlp |   delta_balanced_accuracy_smiles_minus_rf |   delta_balanced_accuracy_smiles_minus_mlp |   delta_brier_score_smiles_minus_rf |   delta_brier_score_smiles_minus_mlp |
|:---------|-----------------:|----------------:|-------------:|---------------------:|---------------------------:|-------------:|------------:|---------:|-----------------:|-----------------------:|--------------:|-------------:|----------:|------------------:|------------------------:|--------------------------------:|---------------------------------:|-------------------------------:|--------------------------------:|----------------------------:|-----------------------------:|------------------------------------------:|-------------------------------------------:|------------------------------------:|-------------------------------------:|
| random   |           0.9872 |          0.9885 |       0.9091 |               0.0375 |                     0.9545 |       0.9901 |      0.9912 |   0.9080 |           0.0561 |                 0.9540 |        0.9875 |       0.9882 |    0.9026 |            0.0387 |                  0.9513 |                         -0.0029 |                          -0.0002 |                        -0.0027 |                          0.0003 |                      0.0010 |                       0.0064 |                                    0.0005 |                                     0.0032 |                             -0.0186 |                              -0.0013 |
| random   |           0.9872 |          0.9885 |       0.9091 |               0.0375 |                     0.9545 |       0.9885 |      0.9882 |   0.9112 |           0.0575 |                 0.9556 |        0.9875 |       0.9882 |    0.9026 |            0.0387 |                  0.9513 |                         -0.0013 |                          -0.0002 |                         0.0003 |                          0.0003 |                     -0.0022 |                       0.0064 |                                   -0.0011 |                                     0.0032 |                             -0.0201 |                              -0.0013 |
| scaffold |           0.9739 |          0.9754 |       0.8391 |               0.0680 |                     0.9192 |       0.9728 |      0.9753 |   0.7925 |           0.0860 |                 0.8925 |        0.9690 |       0.9729 |    0.8135 |            0.0795 |                  0.9058 |                          0.0011 |                           0.0048 |                         0.0001 |                          0.0026 |                      0.0466 |                       0.0256 |                                    0.0267 |                                     0.0134 |                             -0.0180 |                              -0.0114 |
| scaffold |           0.9739 |          0.9754 |       0.8391 |               0.0680 |                     0.9192 |       0.9734 |      0.9755 |   0.8001 |           0.0829 |                 0.8973 |        0.9690 |       0.9729 |    0.8135 |            0.0795 |                  0.9058 |                          0.0004 |                           0.0048 |                        -0.0001 |                          0.0026 |                      0.0390 |                       0.0256 |                                    0.0219 |                                     0.0134 |                             -0.0149 |                              -0.0114 |

Interpretation: positive deltas favor the SMILES model except for Brier score, where lower is better.

## Generated Files

- `smiles_sequence_metrics.csv`
- `smiles_sequence_repeated_split_summary.csv`
- `smiles_sequence_training_history.csv`
- `smiles_sequence_calibration_bins.csv`
- `smiles_sequence_learning_curves.png`
- `smiles_sequence_primary_roc_pr_curves.png`
- `smiles_sequence_primary_confusion_matrices.png`
- `smiles_sequence_calibration_curves.png`
- `smiles_sequence_repeated_split_robustness.png`
- `smiles_sequence_random_seed42.pt`
- `smiles_sequence_scaffold_seed42.pt`
- `smiles_sequence_metadata.json`

## Caution

This model learns directly from SMILES token order, so it tests representation learning beyond fixed Morgan fingerprints. However, SMILES strings are only one linearization of a molecule and can be sensitive to syntax; scaffold-aware performance remains the main generalization signal.
