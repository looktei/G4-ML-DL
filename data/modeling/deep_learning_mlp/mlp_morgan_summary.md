# Phase 3A-1 MLP Morgan Fingerprint Baseline

Generated: 2026-06-12T00:56:25

## Scope

- Model: PyTorch MLP on Morgan fingerprints.
- Dataset: `/home/looktei/Desktop/test1/script/data/modeling/g4ldb_chembl_binary_dataset.csv`
- Rows: 9,344; positives: 4,672; negatives: 4,672.
- Morgan fingerprint: radius 2, 2048 bits.
- Architecture: hidden dims [512, 256], dropout 0.25.
- Device: `cuda`.
- Runtime: 1.1 minutes.
- Negative-label caveat remains: ChEMBL examples are assumed-negative decoys.

## Primary Results

| split    |   roc_auc |   pr_auc |   mcc |   brier_score |   balanced_accuracy |   sensitivity_recall |   specificity |   best_epoch |
|:---------|----------:|---------:|------:|--------------:|--------------------:|---------------------:|--------------:|-------------:|
| random   |     0.987 |    0.988 | 0.903 |         0.039 |               0.951 |                0.950 |         0.953 |            9 |
| scaffold |     0.969 |    0.973 | 0.813 |         0.079 |               0.906 |                0.873 |         0.939 |           14 |

## Repeated Split Robustness

| split    |   roc_auc_mean |   roc_auc_std |   pr_auc_mean |   pr_auc_std |   balanced_accuracy_mean |   balanced_accuracy_std |   mcc_mean |   mcc_std |   brier_score_mean |   brier_score_std |
|:---------|---------------:|--------------:|--------------:|-------------:|-------------------------:|------------------------:|-----------:|----------:|-------------------:|------------------:|
| random   |          0.988 |         0.002 |         0.989 |        0.001 |                    0.954 |                   0.005 |      0.908 |     0.010 |              0.038 |             0.004 |
| scaffold |          0.969 |         0.004 |         0.971 |        0.004 |                    0.903 |                   0.006 |      0.809 |     0.011 |              0.080 |             0.008 |

## Comparison With Phase 1B Random Forest

| split    |   roc_auc_mlp |   pr_auc_mlp |   mcc_mlp |   brier_score_mlp |   balanced_accuracy_mlp |   roc_auc_rf |   pr_auc_rf |   mcc_rf |   brier_score_rf |   balanced_accuracy_rf |   delta_roc_auc_mlp_minus_rf |   delta_pr_auc_mlp_minus_rf |   delta_mcc_mlp_minus_rf |   delta_brier_score_mlp_minus_rf |   delta_balanced_accuracy_mlp_minus_rf |
|:---------|--------------:|-------------:|----------:|------------------:|------------------------:|-------------:|------------:|---------:|-----------------:|-----------------------:|-----------------------------:|----------------------------:|-------------------------:|---------------------------------:|---------------------------------------:|
| random   |        0.9875 |       0.9882 |    0.9026 |            0.0387 |                  0.9513 |       0.9901 |      0.9912 |   0.9080 |           0.0561 |                 0.9540 |                      -0.0027 |                     -0.0030 |                  -0.0054 |                          -0.0174 |                                -0.0027 |
| random   |        0.9875 |       0.9882 |    0.9026 |            0.0387 |                  0.9513 |       0.9885 |      0.9882 |   0.9112 |           0.0575 |                 0.9556 |                      -0.0011 |                      0.0001 |                  -0.0086 |                          -0.0188 |                                -0.0043 |
| scaffold |        0.9690 |       0.9729 |    0.8135 |            0.0795 |                  0.9058 |       0.9728 |      0.9753 |   0.7925 |           0.0860 |                 0.8925 |                      -0.0037 |                     -0.0025 |                   0.0209 |                          -0.0065 |                                 0.0134 |
| scaffold |        0.9690 |       0.9729 |    0.8135 |            0.0795 |                  0.9058 |       0.9734 |      0.9755 |   0.8001 |           0.0829 |                 0.8973 |                      -0.0044 |                     -0.0026 |                   0.0133 |                          -0.0034 |                                 0.0085 |

Interpretation: positive deltas favor the MLP except for Brier score, where lower is better.

## Generated Files

- `mlp_morgan_metrics.csv`
- `mlp_morgan_repeated_split_summary.csv`
- `mlp_morgan_training_history.csv`
- `mlp_morgan_calibration_bins.csv`
- `mlp_morgan_learning_curves.png`
- `mlp_morgan_primary_roc_pr_curves.png`
- `mlp_morgan_primary_confusion_matrices.png`
- `mlp_morgan_calibration_curves.png`
- `mlp_morgan_repeated_split_robustness.png`
- `mlp_morgan_random_seed42.pt`
- `mlp_morgan_scaffold_seed42.pt`
- `mlp_morgan_metadata.json`

## Caution

This MLP is a controlled deep learning baseline using the same Morgan representation as the classical Random Forest. If it does not clearly outperform the RF baseline, that is still useful: it suggests that gains may require representation learning from SMILES or molecular graphs rather than only swapping the classifier.
