# Phase 3A-3 ChemBERTa Fine-Tuning

Generated: 2026-06-12T13:12:41

## Scope

- Model: `DeepChem/ChemBERTa-10M-MLM` with a sequence-classification head.
- Dataset: `/home/looktei/Desktop/test1/script/data/modeling/g4ldb_chembl_binary_dataset.csv`
- Rows: 9,344; positives: 4,672; negatives: 4,672.
- Input: canonical SMILES tokenized by the Hugging Face model tokenizer, max length 256.
- Training: epochs 6, patience 2, batch size 32, learning rate 2e-05.
- Splits run: random, scaffold; primary seed 42; repeated seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18].
- Device: `cuda`.
- Runtime: 55.1 minutes.
- Token length median/IQR/99th percentile: 60 / 47-78 / 168; truncated rows: 15.
- Negative-label caveat remains: ChEMBL examples are assumed-negative decoys.

## Primary Results

| split    |   roc_auc |   pr_auc |   mcc |   brier_score |   balanced_accuracy |   sensitivity_recall |   specificity |   best_epoch |
|:---------|----------:|---------:|------:|--------------:|--------------------:|---------------------:|--------------:|-------------:|
| random   |     0.969 |    0.971 | 0.825 |         0.072 |               0.911 |                0.861 |         0.960 |            6 |
| scaffold |     0.952 |    0.952 | 0.714 |         0.113 |               0.851 |                0.761 |         0.941 |            6 |

## Repeated Split Robustness

| split    |   roc_auc_mean |   roc_auc_std |   pr_auc_mean |   pr_auc_std |   balanced_accuracy_mean |   balanced_accuracy_std |   mcc_mean |   mcc_std |   brier_score_mean |   brier_score_std |
|:---------|---------------:|--------------:|--------------:|-------------:|-------------------------:|------------------------:|-----------:|----------:|-------------------:|------------------:|
| random   |          0.965 |         0.002 |         0.965 |        0.003 |                    0.903 |                   0.007 |      0.808 |     0.014 |              0.075 |             0.004 |
| scaffold |          0.948 |         0.005 |         0.949 |        0.005 |                    0.849 |                   0.008 |      0.711 |     0.013 |              0.115 |             0.006 |

## Comparison With Earlier Models

| split    |   roc_auc_chemberta |   pr_auc_chemberta |   mcc_chemberta |   brier_score_chemberta |   balanced_accuracy_chemberta |   roc_auc_rf |   pr_auc_rf |   mcc_rf |   brier_score_rf |   balanced_accuracy_rf |   roc_auc_mlp |   pr_auc_mlp |   mcc_mlp |   brier_score_mlp |   balanced_accuracy_mlp |   roc_auc_smiles |   pr_auc_smiles |   mcc_smiles |   brier_score_smiles |   balanced_accuracy_smiles |   delta_roc_auc_chemberta_minus_rf |   delta_roc_auc_chemberta_minus_mlp |   delta_roc_auc_chemberta_minus_smiles |   delta_pr_auc_chemberta_minus_rf |   delta_pr_auc_chemberta_minus_mlp |   delta_pr_auc_chemberta_minus_smiles |   delta_mcc_chemberta_minus_rf |   delta_mcc_chemberta_minus_mlp |   delta_mcc_chemberta_minus_smiles |   delta_balanced_accuracy_chemberta_minus_rf |   delta_balanced_accuracy_chemberta_minus_mlp |   delta_balanced_accuracy_chemberta_minus_smiles |   delta_brier_score_chemberta_minus_rf |   delta_brier_score_chemberta_minus_mlp |   delta_brier_score_chemberta_minus_smiles |
|:---------|--------------------:|-------------------:|----------------:|------------------------:|------------------------------:|-------------:|------------:|---------:|-----------------:|-----------------------:|--------------:|-------------:|----------:|------------------:|------------------------:|-----------------:|----------------:|-------------:|---------------------:|---------------------------:|-----------------------------------:|------------------------------------:|---------------------------------------:|----------------------------------:|-----------------------------------:|--------------------------------------:|-------------------------------:|--------------------------------:|-----------------------------------:|---------------------------------------------:|----------------------------------------------:|-------------------------------------------------:|---------------------------------------:|----------------------------------------:|-------------------------------------------:|
| random   |              0.9686 |             0.9706 |          0.8254 |                  0.0719 |                        0.9106 |       0.9901 |      0.9912 |   0.9080 |           0.0561 |                 0.9540 |        0.9875 |       0.9882 |    0.9026 |            0.0387 |                  0.9513 |           0.9872 |          0.9885 |       0.9091 |               0.0375 |                     0.9545 |                            -0.0215 |                             -0.0189 |                                -0.0186 |                           -0.0206 |                            -0.0177 |                               -0.0179 |                        -0.0826 |                         -0.0772 |                            -0.0837 |                                      -0.0434 |                                       -0.0407 |                                          -0.0439 |                                 0.0158 |                                  0.0332 |                                     0.0345 |
| random   |              0.9686 |             0.9706 |          0.8254 |                  0.0719 |                        0.9106 |       0.9885 |      0.9882 |   0.9112 |           0.0575 |                 0.9556 |        0.9875 |       0.9882 |    0.9026 |            0.0387 |                  0.9513 |           0.9872 |          0.9885 |       0.9091 |               0.0375 |                     0.9545 |                            -0.0199 |                             -0.0189 |                                -0.0186 |                           -0.0176 |                            -0.0177 |                               -0.0179 |                        -0.0859 |                         -0.0772 |                            -0.0837 |                                      -0.0450 |                                       -0.0407 |                                          -0.0439 |                                 0.0144 |                                  0.0332 |                                     0.0345 |
| scaffold |              0.9518 |             0.9521 |          0.7142 |                  0.1131 |                        0.8513 |       0.9728 |      0.9753 |   0.7925 |           0.0860 |                 0.8925 |        0.9690 |       0.9729 |    0.8135 |            0.0795 |                  0.9058 |           0.9739 |          0.9754 |       0.8391 |               0.0680 |                     0.9192 |                            -0.0210 |                             -0.0172 |                                -0.0221 |                           -0.0232 |                            -0.0208 |                               -0.0233 |                        -0.0783 |                         -0.0993 |                            -0.1249 |                                      -0.0412 |                                       -0.0545 |                                          -0.0679 |                                 0.0271 |                                  0.0336 |                                     0.0451 |
| scaffold |              0.9518 |             0.9521 |          0.7142 |                  0.1131 |                        0.8513 |       0.9734 |      0.9755 |   0.8001 |           0.0829 |                 0.8973 |        0.9690 |       0.9729 |    0.8135 |            0.0795 |                  0.9058 |           0.9739 |          0.9754 |       0.8391 |               0.0680 |                     0.9192 |                            -0.0217 |                             -0.0172 |                                -0.0221 |                           -0.0234 |                            -0.0208 |                               -0.0233 |                        -0.0859 |                         -0.0993 |                            -0.1249 |                                      -0.0460 |                                       -0.0545 |                                          -0.0679 |                                 0.0302 |                                  0.0336 |                                     0.0451 |

Interpretation: positive deltas favor ChemBERTa except for Brier score, where lower is better.

## Generated Files

- `chemberta_metrics.csv`
- `chemberta_repeated_split_summary.csv`
- `chemberta_training_history.csv`
- `chemberta_calibration_bins.csv`
- `chemberta_learning_curves.png`
- `chemberta_primary_roc_pr_curves.png`
- `chemberta_primary_confusion_matrices.png`
- `chemberta_calibration_curves.png`
- `chemberta_repeated_split_robustness.png`
- `chemberta_random_seed42/` and `chemberta_scaffold_seed42/` when primary models are saved
- `chemberta_metadata.json`
- `chemberta_preprocessing.joblib`

## Caution

This run fine-tunes a small pretrained masked-language model directly from Hugging Face. Scaffold-aware performance is the main generalization signal, and the classification head starts from random weights even though the encoder is pretrained.
