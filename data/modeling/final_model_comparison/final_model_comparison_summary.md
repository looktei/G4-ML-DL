# Final Model Comparison: G4 Ligand Binary Classifier

This report aggregates the final available RF, MLP, SMILES, ChemBERTa, and GNN results across random and scaffold splits.

## Primary seed 42 comparison

### Random split
| Model | ROC-AUC | PR-AUC | MCC | Balanced Acc. | Brier |
| --- | --- | --- | --- | --- | --- |
| ChemBERTa-10M | 0.9686 | 0.9706 | 0.8254 | 0.9106 | 0.0719 |
| Molecular Graph GNN | 0.9924 | 0.9933 | 0.9176 | 0.9588 | 0.0314 |
| Morgan MLP | 0.9875 | 0.9882 | 0.9026 | 0.9513 | 0.0387 |
| Morgan RF | 0.9901 | 0.9912 | 0.9080 | 0.9540 | 0.0561 |
| Morgan RF | 0.9885 | 0.9882 | 0.9112 | 0.9556 | 0.0575 |
| SMILES CNN-BiGRU | 0.9872 | 0.9885 | 0.9091 | 0.9545 | 0.0375 |

### Scaffold split
| Model | ROC-AUC | PR-AUC | MCC | Balanced Acc. | Brier |
| --- | --- | --- | --- | --- | --- |
| ChemBERTa-10M | 0.9518 | 0.9521 | 0.7142 | 0.8513 | 0.1131 |
| Molecular Graph GNN | 0.9801 | 0.9835 | 0.8832 | 0.9406 | 0.0503 |
| Morgan MLP | 0.9690 | 0.9729 | 0.8135 | 0.9058 | 0.0795 |
| Morgan RF | 0.9728 | 0.9753 | 0.7925 | 0.8925 | 0.0860 |
| Morgan RF | 0.9734 | 0.9755 | 0.8001 | 0.8973 | 0.0829 |
| SMILES CNN-BiGRU | 0.9739 | 0.9754 | 0.8391 | 0.9192 | 0.0680 |

## Repeated-seed robustness

### Random split
| Model | N | ROC-AUC | PR-AUC | MCC | Balanced Acc. | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| ChemBERTa-10M | 19 | 0.9654 +/- 0.0022 | 0.9649 +/- 0.0028 | 0.8082 +/- 0.0135 | 0.9028 +/- 0.0070 | 0.0747 +/- 0.0041 |
| Molecular Graph GNN | 19 | 0.9881 +/- 0.0026 | 0.9892 +/- 0.0023 | 0.8994 +/- 0.0178 | 0.9493 +/- 0.0092 | 0.0406 +/- 0.0063 |
| Morgan MLP | 19 | 0.9879 +/- 0.0018 | 0.9895 +/- 0.0015 | 0.9083 +/- 0.0098 | 0.9541 +/- 0.0050 | 0.0380 +/- 0.0038 |
| Morgan RF | 20 | 0.9879 +/- 0.0019 | 0.9888 +/- 0.0017 | 0.8913 +/- 0.0109 | 0.9456 +/- 0.0055 | 0.0583 +/- 0.0020 |
| SMILES CNN-BiGRU | 19 | 0.9863 +/- 0.0022 | 0.9873 +/- 0.0023 | 0.8892 +/- 0.0141 | 0.9440 +/- 0.0074 | 0.0458 +/- 0.0061 |

### Scaffold split
| Model | N | ROC-AUC | PR-AUC | MCC | Balanced Acc. | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| ChemBERTa-10M | 19 | 0.9478 +/- 0.0046 | 0.9488 +/- 0.0050 | 0.7109 +/- 0.0135 | 0.8495 +/- 0.0076 | 0.1147 +/- 0.0057 |
| Molecular Graph GNN | 19 | 0.9760 +/- 0.0042 | 0.9789 +/- 0.0047 | 0.8468 +/- 0.0373 | 0.9208 +/- 0.0216 | 0.0634 +/- 0.0161 |
| Morgan MLP | 19 | 0.9690 +/- 0.0044 | 0.9710 +/- 0.0040 | 0.8089 +/- 0.0114 | 0.9027 +/- 0.0061 | 0.0801 +/- 0.0082 |
| Morgan RF | 20 | 0.9724 +/- 0.0022 | 0.9748 +/- 0.0022 | 0.7938 +/- 0.0079 | 0.8933 +/- 0.0041 | 0.0860 +/- 0.0023 |
| SMILES CNN-BiGRU | 19 | 0.9743 +/- 0.0042 | 0.9754 +/- 0.0047 | 0.8093 +/- 0.0365 | 0.9010 +/- 0.0214 | 0.0798 +/- 0.0154 |

## Metric winners from primary seed 42

### Random split
- roc_auc: Molecular Graph GNN (highest, 0.9924)
- pr_auc: Molecular Graph GNN (highest, 0.9933)
- mcc: Molecular Graph GNN (highest, 0.9176)
- balanced_accuracy: Molecular Graph GNN (highest, 0.9588)
- brier_score: Molecular Graph GNN (lowest, 0.0314)

### Scaffold split
- roc_auc: Molecular Graph GNN (highest, 0.9801)
- pr_auc: Molecular Graph GNN (highest, 0.9835)
- mcc: Molecular Graph GNN (highest, 0.8832)
- balanced_accuracy: Molecular Graph GNN (highest, 0.9406)
- brier_score: Molecular Graph GNN (lowest, 0.0503)

## Interpretation

- Best average primary-metric rank on scaffold split: Molecular Graph GNN (mean rank 1.00).
- Best average primary-metric rank on random split: Molecular Graph GNN (mean rank 1.00).
- Scaffold split should be prioritized for drug-discovery interpretation because it tests generalization to chemically distinct scaffolds.
- Repeated-seed robustness now uses the same seed set for all comparator families.
- Use ROC-AUC/PR-AUC for ranking quality, MCC/balanced accuracy for thresholded classification quality, and Brier score for calibration quality.

## Output files

- `final_model_primary_seed42_comparison.csv`
- `final_model_repeated_seed_summary.csv`
- `final_model_primary_rankings.csv`
- `final_model_metric_winners.csv`
- `final_model_primary_metric_heatmap.png`
