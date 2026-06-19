# Phase 1 Classical Baseline Training Summary

Generated: 2026-06-09T18:06:52

## Dataset

- Dataset: `/home/looktei/Desktop/test1/script/data/modeling/g4ldb_chembl_binary_dataset.csv`
- Rows: 9,344
- Positive: 4,672
- Negative: 4,672
- Framing: G4LDB-positive vs ChEMBL assumed-negative classifier.
- Caveat: ChEMBL negatives are assumed-negative, not experimentally verified non-G4 binders.

## Training Setup

- Replicate mode: n_replicates=20 seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
- Primary seed: 42
- Random seed (single-seed backward compat): 42
- Test size: 0.2
- Morgan fingerprint: radius 2, 2048 bits
- Feature sets: descriptor, Morgan fingerprint, combined.
- Models: logistic regression, random forest, LightGBM if available, XGBoost if available.

## Split Summary

| split    |   train_n |   test_n |   train_positive |   train_negative |   test_positive |   test_negative |   test_positive_rate |
|:---------|----------:|---------:|-----------------:|-----------------:|----------------:|----------------:|---------------------:|
| random   |      7475 |     1869 |             3737 |             3738 |             935 |             934 |             0.500268 |
| scaffold |      7475 |     1869 |             3738 |             3737 |             934 |             935 |             0.499732 |

## Top Models By ROC-AUC

| split   | feature_set   | model               |   roc_auc |   roc_auc_std |   pr_auc |    mcc |   mcc_std |
|:--------|:--------------|:--------------------|----------:|--------------:|---------:|-------:|----------:|
| random  | combined      | random_forest       |    0.9886 |        0.0017 |   0.9896 | 0.8946 |    0.0122 |
| random  | combined      | lightgbm            |    0.9882 |        0.0016 |   0.9894 | 0.8978 |    0.0106 |
| random  | morgan        | random_forest       |    0.9873 |        0.0018 |   0.9879 | 0.8880 |    0.0085 |
| random  | morgan        | lightgbm            |    0.9866 |        0.0016 |   0.9879 | 0.8915 |    0.0092 |
| random  | combined      | xgboost             |    0.9776 |        0.0025 |   0.9790 | 0.8544 |    0.0114 |
| random  | morgan        | xgboost             |    0.9755 |        0.0026 |   0.9769 | 0.8431 |    0.0110 |
| random  | combined      | logistic_regression |    0.9742 |        0.0026 |   0.9768 | 0.8596 |    0.0115 |
| random  | morgan        | logistic_regression |    0.9731 |        0.0026 |   0.9754 | 0.8555 |    0.0129 |

## Key Results

- Best random split model: `combined / random_forest` with ROC-AUC 0.989, PR-AUC 0.990, MCC 0.895.
- Best scaffold split model: `morgan / random_forest` with ROC-AUC 0.973, PR-AUC 0.975, MCC 0.797.
- Best descriptor-only baseline: `random_forest` on `random` split with ROC-AUC 0.858.

## Initial Interpretation

- Descriptor-only performance should be treated as a bias indicator. If it is high, simple physicochemical properties are carrying classification signal.
- Morgan and combined models should be judged more strongly on scaffold split than random split, because scaffold split better approximates generalization to new chemotypes.
- This Phase 1 run is intended for model triage and bias checking before downstream Nipah G4 compound prioritization.

## Top Feature Importance Rows

| split   | feature_set   | model               | feature_label    |   importance |
|:--------|:--------------|:--------------------|:-----------------|-------------:|
| random  | descriptor    | logistic_regression | TPSA             |      2.02966 |
| random  | descriptor    | logistic_regression | Molecular weight |      1.30946 |
| random  | descriptor    | logistic_regression | AlogP            |      0.65104 |
| random  | descriptor    | logistic_regression | HBD              |      0.63070 |
| random  | descriptor    | logistic_regression | HBA              |      0.53827 |
| random  | descriptor    | random_forest       | TPSA             |      0.30608 |
| random  | descriptor    | random_forest       | Molecular weight |      0.26980 |
| random  | descriptor    | random_forest       | AlogP            |      0.22911 |
| random  | descriptor    | random_forest       | HBA              |      0.11697 |
| random  | descriptor    | random_forest       | HBD              |      0.07804 |
| random  | descriptor    | lightgbm            | TPSA             |   3009.00000 |
| random  | descriptor    | lightgbm            | Molecular weight |   2145.00000 |
| random  | descriptor    | lightgbm            | AlogP            |   1810.00000 |
| random  | descriptor    | lightgbm            | HBA              |   1245.00000 |
| random  | descriptor    | lightgbm            | HBD              |    791.00000 |
| random  | descriptor    | xgboost             | Molecular weight |      0.23166 |
| random  | descriptor    | xgboost             | HBA              |      0.22522 |
| random  | descriptor    | xgboost             | TPSA             |      0.21718 |
| random  | descriptor    | xgboost             | HBD              |      0.18739 |
| random  | descriptor    | xgboost             | AlogP            |      0.13855 |

## Output Files

- `baseline_model_performance.csv`
- `baseline_split_summary.csv`
- `baseline_feature_importance_top.csv`
- `baseline_training_metadata.json`
- `baseline_*_comparison.png`
- `best_baseline_roc_pr_curves.png`
- `best_baseline_confusion_matrices.png`
- `best_scaffold_model.joblib`
