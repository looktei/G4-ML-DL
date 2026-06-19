# Phase 3A-4 Molecular Graph GNN

Generated: 2026-06-12T15:31:31

## Scope

- Model: PyTorch Geometric molecular graph classifier with GINEConv layers.
- Dataset: `/home/looktei/Desktop/test1/script/data/modeling/g4ldb_chembl_binary_dataset.csv`
- Rows: 9,344; positives: 4,672; negatives: 4,672.
- Graph features: atom categorical descriptors plus directed bond categorical descriptors from RDKit.
- Architecture: hidden_dim=128, layers=4, dense_hidden=256, dropout=0.2.
- Device: `cuda`.
- Runtime: 31.2 minutes.
- Negative-label caveat remains: ChEMBL examples are assumed-negative decoys.

## Primary Results

| split    |   roc_auc |   pr_auc |   mcc |   brier_score |   balanced_accuracy |   sensitivity_recall |   specificity |   best_epoch |
|:---------|----------:|---------:|------:|--------------:|--------------------:|---------------------:|--------------:|-------------:|
| random   |     0.992 |    0.993 | 0.918 |         0.031 |               0.959 |                0.959 |         0.958 |           49 |
| scaffold |     0.980 |    0.983 | 0.883 |         0.050 |               0.941 |                0.907 |         0.974 |           56 |

## Repeated Split Robustness

| split    |   roc_auc_mean |   roc_auc_std |   pr_auc_mean |   pr_auc_std |   balanced_accuracy_mean |   balanced_accuracy_std |   mcc_mean |   mcc_std |   brier_score_mean |   brier_score_std |
|:---------|---------------:|--------------:|--------------:|-------------:|-------------------------:|------------------------:|-----------:|----------:|-------------------:|------------------:|
| random   |          0.988 |         0.003 |         0.989 |        0.002 |                    0.949 |                   0.009 |      0.899 |     0.018 |              0.041 |             0.006 |
| scaffold |          0.976 |         0.004 |         0.979 |        0.005 |                    0.921 |                   0.022 |      0.847 |     0.037 |              0.063 |             0.016 |

## Comparison With Previous Models

| split    |   roc_auc_gnn |   pr_auc_gnn |   mcc_gnn |   brier_score_gnn |   balanced_accuracy_gnn |   roc_auc |   pr_auc |    mcc |   brier_score |   balanced_accuracy |   roc_auc_mlp |   pr_auc_mlp |   mcc_mlp |   brier_score_mlp |   balanced_accuracy_mlp |   delta_roc_auc_gnn_minus_mlp |   delta_pr_auc_gnn_minus_mlp |   delta_mcc_gnn_minus_mlp |   delta_balanced_accuracy_gnn_minus_mlp |   delta_brier_score_gnn_minus_mlp |   roc_auc_smiles |   pr_auc_smiles |   mcc_smiles |   brier_score_smiles |   balanced_accuracy_smiles |   delta_roc_auc_gnn_minus_smiles |   delta_pr_auc_gnn_minus_smiles |   delta_mcc_gnn_minus_smiles |   delta_balanced_accuracy_gnn_minus_smiles |   delta_brier_score_gnn_minus_smiles |   roc_auc_chemberta |   pr_auc_chemberta |   mcc_chemberta |   brier_score_chemberta |   balanced_accuracy_chemberta |   delta_roc_auc_gnn_minus_chemberta |   delta_pr_auc_gnn_minus_chemberta |   delta_mcc_gnn_minus_chemberta |   delta_balanced_accuracy_gnn_minus_chemberta |   delta_brier_score_gnn_minus_chemberta |
|:---------|--------------:|-------------:|----------:|------------------:|------------------------:|----------:|---------:|-------:|--------------:|--------------------:|--------------:|-------------:|----------:|------------------:|------------------------:|------------------------------:|-----------------------------:|--------------------------:|----------------------------------------:|----------------------------------:|-----------------:|----------------:|-------------:|---------------------:|---------------------------:|---------------------------------:|--------------------------------:|-----------------------------:|-------------------------------------------:|-------------------------------------:|--------------------:|-------------------:|----------------:|------------------------:|------------------------------:|------------------------------------:|-----------------------------------:|--------------------------------:|----------------------------------------------:|----------------------------------------:|
| random   |        0.9924 |       0.9933 |    0.9176 |            0.0314 |                  0.9588 |    0.9901 |   0.9912 | 0.9080 |        0.0561 |              0.9540 |        0.9875 |       0.9882 |    0.9026 |            0.0387 |                  0.9513 |                        0.0049 |                       0.0050 |                    0.0150 |                                  0.0075 |                           -0.0073 |           0.9872 |          0.9885 |       0.9091 |               0.0375 |                     0.9545 |                           0.0052 |                          0.0048 |                       0.0085 |                                     0.0043 |                              -0.0060 |              0.9686 |             0.9706 |          0.8254 |                  0.0719 |                        0.9106 |                              0.0238 |                             0.0227 |                          0.0922 |                                        0.0482 |                                 -0.0405 |
| random   |        0.9924 |       0.9933 |    0.9176 |            0.0314 |                  0.9588 |    0.9885 |   0.9882 | 0.9112 |        0.0575 |              0.9556 |        0.9875 |       0.9882 |    0.9026 |            0.0387 |                  0.9513 |                        0.0049 |                       0.0050 |                    0.0150 |                                  0.0075 |                           -0.0073 |           0.9872 |          0.9885 |       0.9091 |               0.0375 |                     0.9545 |                           0.0052 |                          0.0048 |                       0.0085 |                                     0.0043 |                              -0.0060 |              0.9686 |             0.9706 |          0.8254 |                  0.0719 |                        0.9106 |                              0.0238 |                             0.0227 |                          0.0922 |                                        0.0482 |                                 -0.0405 |
| scaffold |        0.9801 |       0.9835 |    0.8832 |            0.0503 |                  0.9406 |    0.9728 |   0.9753 | 0.7925 |        0.0860 |              0.8925 |        0.9690 |       0.9729 |    0.8135 |            0.0795 |                  0.9058 |                        0.0110 |                       0.0106 |                    0.0698 |                                  0.0348 |                           -0.0292 |           0.9739 |          0.9754 |       0.8391 |               0.0680 |                     0.9192 |                           0.0062 |                          0.0080 |                       0.0442 |                                     0.0214 |                              -0.0177 |              0.9518 |             0.9521 |          0.7142 |                  0.1131 |                        0.8513 |                              0.0283 |                             0.0313 |                          0.1691 |                                        0.0893 |                                 -0.0628 |
| scaffold |        0.9801 |       0.9835 |    0.8832 |            0.0503 |                  0.9406 |    0.9734 |   0.9755 | 0.8001 |        0.0829 |              0.8973 |        0.9690 |       0.9729 |    0.8135 |            0.0795 |                  0.9058 |                        0.0110 |                       0.0106 |                    0.0698 |                                  0.0348 |                           -0.0292 |           0.9739 |          0.9754 |       0.8391 |               0.0680 |                     0.9192 |                           0.0062 |                          0.0080 |                       0.0442 |                                     0.0214 |                              -0.0177 |              0.9518 |             0.9521 |          0.7142 |                  0.1131 |                        0.8513 |                              0.0283 |                             0.0313 |                          0.1691 |                                        0.0893 |                                 -0.0628 |

Interpretation: positive deltas favor the GNN except for Brier score, where lower is better.

## Generated Files

- `gnn_molecular_graph_metrics.csv`
- `gnn_molecular_graph_repeated_split_summary.csv`
- `gnn_molecular_graph_training_history.csv`
- `gnn_molecular_graph_calibration_bins.csv`
- `gnn_molecular_graph_learning_curves.png`
- `gnn_molecular_graph_primary_roc_pr_curves.png`
- `gnn_molecular_graph_primary_confusion_matrices.png`
- `gnn_molecular_graph_calibration_curves.png`
- `gnn_molecular_graph_repeated_split_robustness.png`
- `gnn_molecular_graph_random_seed42.pt`
- `gnn_molecular_graph_scaffold_seed42.pt`
- `gnn_molecular_graph_metadata.json`

## Caution

This model learns directly from molecular graph topology and atom/bond descriptors. Strong performance would support graph representation learning, but it should still be interpreted under the assumed-negative decoy caveat and checked against scaffold split behavior before using it for compound prioritization.
