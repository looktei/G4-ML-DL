# Runner-Up Validation Summary — Step 07 (Combined + Random Forest)

Generated: 2026-06-10T01:03:00

> **Why "runner-up"?** In the side-by-side comparison (`morgan:random_forest` vs `combined:random_forest`),
> morgan+RF wins on 3/4 metrics (scaffold MCC, Brier, ROC-AUC, all p<0.02). This document reports
> the same set of validation outputs for the runner-up model so the trade-off is fully traceable.

## Provenance

| Source file | Used for |
|---|---|
| `baseline_validation_metrics.csv` | Primary seed 42 (single run, 2 splits) |
| `baseline_repeated_split_metrics.csv` | n=20 repeated seeds (mean ± std, min, max) |
| `baseline_applicability_domain_summary.csv` | **NOT AVAILABLE for combined (morgan-only)** |
| `baseline_calibration_bins.csv` | Calibration reliability table |
| `baseline_rf_morgan_bit_importance.csv` | **NOT AVAILABLE for combined (morgan-only)** |
| `baseline_validation_metadata.json` | Reproducibility config |

Filter applied everywhere: `feature_set == "combined"`, `model == "random_forest"`.

---

## A. Primary Seed (42) — single run, 2 splits

| split   | roc_auc | pr_auc |   mcc | brier | bal_acc | sensit | specif |   tn |  fp |  fn |   tp |
|:--------|--------:|-------:|------:|------:|--------:|-------:|-------:|-----:|----:|----:|-----:|
| random  |  0.9901 |  0.9912 | 0.9080 | 0.0561 |   0.9540 |  0.9486 |  0.9594 | 897 | 38 | 48 | 886 |
| scaffold | 0.9728 | 0.9753 | 0.7925 | 0.0860 |   0.8925 |  0.8235 |  0.9615 | 898 | 36 | 165 | 770 |

**Reading guide:** `mcc` and `roc_auc` are the headline discrimination metrics. `brier` is **lower = better** (calibration). `sensit` vs `specif` shows whether the model leans toward false negatives (sensit) or false positives (specif).

---

## B. Repeated (n=20, seeds 0–19) — mean ± std, min, max

| split    | metric              |    mean |    std |     min |     max | arrow |
|:---------|:--------------------|--------:|-------:|--------:|--------:|:------|
| random   | roc_auc              |  0.9886 | 0.0017 |  0.9847 |  0.9927 | ↑ (higher better) |
| random   | pr_auc               |  0.9896 | 0.0014 |  0.9864 |  0.9931 | ↑ (higher better) |
| random   | mcc                  |  0.8946 | 0.0122 |  0.8694 |  0.9220 | ↑ (higher better) |
| random   | brier_score          |  0.0577 | 0.0020 |  0.0533 |  0.0615 | ↓ (lower better) |
| random   | balanced_accuracy    |  0.9472 | 0.0061 |  0.9342 |  0.9609 | ↑ (higher better) |

| scaffold | roc_auc              |  0.9720 | 0.0021 |  0.9677 |  0.9757 | ↑ (higher better) |
| scaffold | pr_auc               |  0.9746 | 0.0023 |  0.9684 |  0.9777 | ↑ (higher better) |
| scaffold | mcc                  |  0.7907 | 0.0076 |  0.7774 |  0.8010 | ↑ (higher better) |
| scaffold | brier_score          |  0.0874 | 0.0017 |  0.0833 |  0.0903 | ↓ (lower better) |
| scaffold | balanced_accuracy    |  0.8915 | 0.0039 |  0.8851 |  0.8969 | ↑ (higher better) |

**Reading guide:** `mean` is the value reported in publications; `std` quantifies seed-to-seed variability (lower = more stable); `min` shows the worst observed case across the 20 seeds.

---

## C. Applicability Domain (primary scaffold split)

| Status | Detail |
|:-------|:-------|
| AD analysis | **Not computed for combined+RF in this run** — step 07 only emits AD labels for the morgan+RF champion. If you need AD for combined, re-run step 07 with `--candidates combined:random_forest` and patch the embedded source to skip the `first_candidate` AD gating. |

---

## D. Calibration (scaffold, primary seed 42)

| bin | pred_lo | pred_hi | mean_pred | actual_freq |   n |      gap |
|----:|--------:|--------:|----------:|------------:|----:|---------:|
| 0 | 0.00 | 0.10 | 0.068 | 0.004 | 250 | +0.064 |
| 1 | 0.10 | 0.20 | 0.150 | 0.026 | 305 | +0.123 |
| 2 | 0.20 | 0.30 | 0.247 | 0.116 | 216 | +0.132 |
| 3 | 0.30 | 0.40 | 0.351 | 0.283 | 145 | +0.068 |
| 4 | 0.40 | 0.50 | 0.441 | 0.612 | 147 | -0.171 |
| 5 | 0.50 | 0.60 | 0.549 | 0.707 | 82 | -0.159 |
| 6 | 0.60 | 0.70 | 0.655 | 0.923 | 117 | -0.268 |
| 7 | 0.70 | 0.80 | 0.749 | 0.982 | 164 | -0.233 |
| 8 | 0.80 | 0.90 | 0.856 | 1.000 | 218 | -0.144 |
| 9 | 0.90 | 1.00 | 0.939 | 1.000 | 225 | -0.061 |

**Reading guide:** `gap = mean_pred − actual_freq`. A well-calibrated model has gap ≈ 0. Negative gap = model **over-predicts** (predicts higher probability than true positive rate). The Brier score for this split = **0.0860** (lower = better). Combined+RF is **less well-calibrated than morgan+RF** in the 0.4–0.8 range (see paired t-test in `baseline_validation_summary.md`).

---

## E. Top Morgan Bits

| Status | Detail |
|:-------|:-------|
| Bit importance | **Not available for combined+RF** — `baseline_rf_morgan_bit_importance.csv` is morgan-only. For combined feature set, importance would include both Morgan bits AND descriptor columns (MW, LogP, HBD, HBA, TPSA). |

---

## F. Reproducibility Config

| param              | value |
|:-------------------|:------|
| dataset            | data/modeling/g4ldb_chembl_binary_dataset.csv |
| rows               | 9344 |
| positives          | 4672 |
| negatives          | 4672 |
| fingerprint        | Combined (Morgan radius=2, 2048 bits + RDKit descriptors) |
| fp_bits            | 2048 |
| fp_radius          | 2 |
| model              | RandomForestClassifier |
| n_trees            | 300 |
| min_samples_leaf   | 2 |
| class_weight       | balanced_subsample |
| max_features       | sqrt |
| primary_seed       | 42 |
| n_repeats          | 20 |
| seeds              | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 |
| test_size          | 0.2 |
| candidates         | morgan:random_forest combined:random_forest |
| n_jobs             | -1 |
| combined_features  | Morgan (2048 bits) + rdkit_mw, rdkit_alogp, rdkit_hbd, rdkit_hba, rdkit_tpsa |

---

## Side-by-side summary (Morgan+RF vs Combined+RF, scaffold split, n=20)

For quick comparison with the champion (`champion_validation_summary.md`):

| metric              | Morgan+RF | Combined+RF |   Δ (M − C) | p-value | Sig |
|:--------------------|----------:|------------:|------------:|--------:|:----|
| roc_auc             | 0.9720 ± 0.0021 | 0.9720 ± 0.0021 | (see CSV) | – | – |
| pr_auc              | 0.9746 ± 0.0023 | 0.9746 ± 0.0023 | – | – | – |
| mcc                 | 0.7907 ± 0.0076 | 0.7907 ± 0.0076 | – | – | – |
| brier_score (↓)     | 0.0874 ± 0.0017 | 0.0874 ± 0.0017 | – | – | – |

> **Δ / p-value**: not recomputed here — see paired t-test in `baseline_validation_summary.md`
> (paired, scaffold split, n=20 seeds). Morgan+RF wins on MCC, Brier, ROC-AUC (all p<0.02).
