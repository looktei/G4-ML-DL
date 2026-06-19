# Phase 1B Baseline Validation Summary

Generated: 2026-06-10T00:04:07

## Scope

- Candidates validated: `morgan/random_forest + combined/random_forest`.
- Dataset: `/home/looktei/Desktop/test1/script/data/modeling/g4ldb_chembl_binary_dataset.csv`
- Rows: 9,344; positives: 4,672; negatives: 4,672.
- Morgan fingerprint: radius 2, 2048 bits.
- Random forest trees: 300; min_samples_leaf=2; class_weight=`balanced_subsample`; max_features=`sqrt`.
- Negative-label caveat remains: ChEMBL examples are assumed-negative decoys.

## Primary Validation

| split    |   roc_auc |   pr_auc |   mcc |   brier_score |   balanced_accuracy |   sensitivity_recall |   specificity |
|:---------|----------:|---------:|------:|--------------:|--------------------:|---------------------:|--------------:|
| random   |     0.990 |    0.991 | 0.908 |         0.056 |               0.954 |                0.949 |         0.959 |
| scaffold |     0.973 |    0.975 | 0.793 |         0.086 |               0.892 |                0.824 |         0.961 |
| random   |     0.989 |    0.988 | 0.911 |         0.058 |               0.956 |                0.950 |         0.961 |
| scaffold |     0.973 |    0.976 | 0.800 |         0.083 |               0.897 |                0.839 |         0.956 |

## Repeated Split Robustness

| split    |   roc_auc_mean |   roc_auc_std |   roc_auc_min |   roc_auc_max |   pr_auc_mean |   pr_auc_std |   pr_auc_min |   pr_auc_max |   balanced_accuracy_mean |   balanced_accuracy_std |   balanced_accuracy_min |   balanced_accuracy_max |   mcc_mean |   mcc_std |   mcc_min |   mcc_max |   brier_score_mean |   brier_score_std |   brier_score_min |   brier_score_max |
|:---------|---------------:|--------------:|--------------:|--------------:|--------------:|-------------:|-------------:|-------------:|-------------------------:|------------------------:|------------------------:|------------------------:|-----------:|----------:|----------:|----------:|-------------------:|------------------:|------------------:|------------------:|
| random   |          0.988 |         0.002 |         0.984 |         0.993 |         0.989 |        0.002 |        0.985 |        0.993 |                    0.946 |                   0.005 |                   0.934 |                   0.961 |      0.891 |     0.011 |     0.868 |     0.922 |              0.058 |             0.002 |             0.053 |             0.062 |
| scaffold |          0.972 |         0.002 |         0.968 |         0.977 |         0.975 |        0.002 |        0.968 |        0.979 |                    0.893 |                   0.004 |                   0.885 |                   0.903 |      0.794 |     0.008 |     0.777 |     0.810 |              0.086 |             0.002 |             0.081 |             0.090 |

Detailed repeated split rows:

| split    |   seed |   roc_auc |   pr_auc |   mcc |   brier_score |   balanced_accuracy |
|:---------|-------:|----------:|---------:|------:|--------------:|--------------------:|
| random   |      0 |     0.985 |    0.986 | 0.869 |         0.062 |               0.934 |
| random   |      1 |     0.988 |    0.989 | 0.885 |         0.058 |               0.942 |
| random   |      2 |     0.988 |    0.989 | 0.897 |         0.058 |               0.949 |
| random   |      3 |     0.989 |    0.990 | 0.901 |         0.057 |               0.950 |
| random   |      4 |     0.987 |    0.988 | 0.891 |         0.058 |               0.945 |
| random   |      5 |     0.989 |    0.990 | 0.905 |         0.057 |               0.952 |
| random   |      6 |     0.990 |    0.991 | 0.902 |         0.056 |               0.951 |
| random   |      7 |     0.989 |    0.990 | 0.884 |         0.058 |               0.942 |
| random   |      8 |     0.990 |    0.991 | 0.900 |         0.057 |               0.950 |
| random   |      9 |     0.988 |    0.989 | 0.887 |         0.059 |               0.943 |
| random   |     10 |     0.988 |    0.989 | 0.893 |         0.060 |               0.946 |
| random   |     11 |     0.990 |    0.990 | 0.887 |         0.058 |               0.943 |
| random   |     12 |     0.993 |    0.993 | 0.922 |         0.053 |               0.961 |
| random   |     13 |     0.990 |    0.991 | 0.901 |         0.055 |               0.950 |
| random   |     14 |     0.988 |    0.989 | 0.898 |         0.060 |               0.949 |
| random   |     15 |     0.986 |    0.987 | 0.884 |         0.061 |               0.942 |
| random   |     16 |     0.988 |    0.990 | 0.896 |         0.057 |               0.948 |
| random   |     17 |     0.987 |    0.989 | 0.878 |         0.058 |               0.939 |
| random   |     18 |     0.990 |    0.990 | 0.915 |         0.055 |               0.957 |
| random   |     19 |     0.990 |    0.990 | 0.898 |         0.057 |               0.949 |
| scaffold |      0 |     0.974 |    0.977 | 0.800 |         0.086 |               0.896 |
| scaffold |      1 |     0.972 |    0.974 | 0.791 |         0.088 |               0.893 |
| scaffold |      2 |     0.974 |    0.977 | 0.789 |         0.087 |               0.889 |
| scaffold |      3 |     0.972 |    0.975 | 0.800 |         0.087 |               0.896 |
| scaffold |      4 |     0.968 |    0.968 | 0.778 |         0.089 |               0.886 |
| scaffold |      5 |     0.972 |    0.975 | 0.792 |         0.087 |               0.893 |
| scaffold |      6 |     0.971 |    0.974 | 0.800 |         0.087 |               0.896 |
| scaffold |      7 |     0.970 |    0.973 | 0.782 |         0.089 |               0.887 |
| scaffold |      8 |     0.973 |    0.976 | 0.793 |         0.088 |               0.892 |
| scaffold |      9 |     0.976 |    0.978 | 0.799 |         0.083 |               0.897 |
| scaffold |     10 |     0.971 |    0.974 | 0.791 |         0.088 |               0.891 |
| scaffold |     11 |     0.968 |    0.971 | 0.780 |         0.090 |               0.886 |
| scaffold |     12 |     0.973 |    0.975 | 0.789 |         0.089 |               0.890 |
| scaffold |     13 |     0.973 |    0.976 | 0.801 |         0.086 |               0.897 |
| scaffold |     14 |     0.969 |    0.972 | 0.777 |         0.090 |               0.885 |
| scaffold |     15 |     0.971 |    0.974 | 0.786 |         0.088 |               0.889 |
| scaffold |     16 |     0.974 |    0.977 | 0.800 |         0.085 |               0.896 |
| scaffold |     17 |     0.973 |    0.976 | 0.789 |         0.086 |               0.891 |
| scaffold |     18 |     0.971 |    0.974 | 0.787 |         0.088 |               0.890 |
| scaffold |     19 |     0.973 |    0.976 | 0.792 |         0.085 |               0.893 |
| random   |      0 |     0.984 |    0.986 | 0.868 |         0.062 |               0.934 |
| random   |      1 |     0.986 |    0.988 | 0.880 |         0.059 |               0.940 |
| random   |      2 |     0.985 |    0.987 | 0.886 |         0.060 |               0.943 |
| random   |      3 |     0.987 |    0.987 | 0.891 |         0.059 |               0.945 |
| random   |      4 |     0.986 |    0.987 | 0.888 |         0.059 |               0.944 |
| random   |      5 |     0.988 |    0.987 | 0.897 |         0.059 |               0.949 |
| random   |      6 |     0.988 |    0.988 | 0.895 |         0.057 |               0.948 |
| random   |      7 |     0.987 |    0.988 | 0.884 |         0.060 |               0.942 |
| random   |      8 |     0.990 |    0.990 | 0.900 |         0.058 |               0.950 |
| random   |      9 |     0.986 |    0.988 | 0.886 |         0.060 |               0.943 |
| random   |     10 |     0.988 |    0.988 | 0.882 |         0.060 |               0.941 |
| random   |     11 |     0.989 |    0.989 | 0.887 |         0.060 |               0.943 |
| random   |     12 |     0.991 |    0.990 | 0.902 |         0.055 |               0.951 |
| random   |     13 |     0.989 |    0.990 | 0.889 |         0.057 |               0.944 |
| random   |     14 |     0.987 |    0.986 | 0.883 |         0.062 |               0.942 |
| random   |     15 |     0.984 |    0.985 | 0.881 |         0.062 |               0.941 |
| random   |     16 |     0.986 |    0.989 | 0.891 |         0.059 |               0.945 |
| random   |     17 |     0.986 |    0.988 | 0.878 |         0.059 |               0.939 |
| random   |     18 |     0.988 |    0.990 | 0.902 |         0.056 |               0.951 |
| random   |     19 |     0.989 |    0.988 | 0.892 |         0.057 |               0.946 |
| scaffold |      0 |     0.976 |    0.978 | 0.806 |         0.083 |               0.900 |
| scaffold |      1 |     0.973 |    0.975 | 0.791 |         0.084 |               0.893 |
| scaffold |      2 |     0.977 |    0.979 | 0.799 |         0.082 |               0.896 |
| scaffold |      3 |     0.973 |    0.975 | 0.803 |         0.085 |               0.898 |
| scaffold |      4 |     0.971 |    0.972 | 0.791 |         0.085 |               0.893 |
| scaffold |      5 |     0.974 |    0.976 | 0.796 |         0.084 |               0.895 |
| scaffold |      6 |     0.973 |    0.976 | 0.793 |         0.084 |               0.893 |
| scaffold |      7 |     0.971 |    0.974 | 0.796 |         0.086 |               0.895 |
| scaffold |      8 |     0.975 |    0.978 | 0.800 |         0.084 |               0.896 |
| scaffold |      9 |     0.975 |    0.976 | 0.810 |         0.082 |               0.903 |
| scaffold |     10 |     0.972 |    0.975 | 0.794 |         0.086 |               0.893 |
| scaffold |     11 |     0.968 |    0.970 | 0.788 |         0.089 |               0.890 |
| scaffold |     12 |     0.972 |    0.975 | 0.800 |         0.088 |               0.896 |
| scaffold |     13 |     0.974 |    0.977 | 0.801 |         0.083 |               0.897 |
| scaffold |     14 |     0.971 |    0.973 | 0.789 |         0.086 |               0.891 |
| scaffold |     15 |     0.971 |    0.973 | 0.799 |         0.086 |               0.897 |
| scaffold |     16 |     0.976 |    0.979 | 0.809 |         0.081 |               0.902 |
| scaffold |     17 |     0.974 |    0.976 | 0.796 |         0.084 |               0.895 |
| scaffold |     18 |     0.969 |    0.972 | 0.785 |         0.086 |               0.890 |
| scaffold |     19 |     0.972 |    0.974 | 0.789 |         0.085 |               0.892 |

## Calibration

- Brier score is included for both the random and scaffold primary evaluations; lower values indicate better probability calibration.
- Use `baseline_calibration_curves.png` to inspect whether predicted probabilities are over- or under-confident.

## Applicability Domain

- In-domain: max Morgan Tanimoto similarity to the training set >= 0.50.
- Borderline: >= 0.35 and < 0.50.
- Out-of-domain: < 0.35.

| domain        |    n |   positive_rate |   mean_max_tanimoto_train |   accuracy |   roc_auc |   pr_auc |   brier_score |   mcc |
|:--------------|-----:|----------------:|--------------------------:|-----------:|----------:|---------:|--------------:|------:|
| in-domain     | 1286 |           0.516 |                     0.691 |      0.973 |     0.994 |    0.995 |         0.049 | 0.946 |
| borderline    |  362 |           0.547 |                     0.423 |      0.702 |     0.893 |    0.898 |         0.160 | 0.455 |
| out-of-domain |  221 |           0.335 |                     0.293 |      0.778 |     0.831 |    0.768 |         0.155 | 0.476 |

## Top Morgan Bits

|   rank |   morgan_bit |   importance |   bit_radius | representative_smarts                              |
|-------:|-------------:|-------------:|-------------:|:---------------------------------------------------|
|      1 |          463 |      0.01848 |            0 | [n+]                                               |
|      2 |         1357 |      0.01779 |            1 | [#6]:[#6](:[#6]):[#6]                              |
|      3 |         1019 |      0.01666 |            0 | C                                                  |
|      4 |          650 |      0.01518 |            0 | O                                                  |
|      5 |            1 |      0.01452 |            2 | [#6]-[#6](=[#8])-[#6](:[#6]:[#6]):[#6](:[#6])-[#6] |
|      6 |         1917 |      0.01435 |            1 | [#6]=[#8]                                          |
|      7 |         1754 |      0.01374 |            1 | [#6]-[#6](:[#6]):[#6]                              |
|      8 |         1325 |      0.01291 |            1 | [#6]-[#6]-[#6]                                     |
|      9 |         1855 |      0.01143 |            1 | [#6]:[#7]:[#6]                                     |
|     10 |          117 |      0.01057 |            1 | [#6]-[#7]-[#6]                                     |
|     11 |          807 |      0.01017 |            0 | C                                                  |
|     12 |          695 |      0.01016 |            0 | O                                                  |

## Generated Files

- `baseline_validation_metrics.csv`
- `baseline_repeated_split_metrics.csv`
- `baseline_repeated_split_summary.csv`
- `baseline_calibration_bins.csv`
- `baseline_calibration_curves.png`
- `baseline_applicability_domain_primary_scaffold.csv`
- `baseline_applicability_domain_summary.csv`
- `baseline_applicability_domain.png`
- `baseline_rf_morgan_bit_importance.csv`
- `baseline_rf_top_morgan_bits.png`
- `primary_random_morgan_rf_model.joblib`
- `primary_scaffold_morgan_rf_model.joblib`
