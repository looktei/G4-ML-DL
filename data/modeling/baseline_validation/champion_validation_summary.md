# Champion Validation Summary — Step 07 (Morgan + Random Forest)

Generated: 2026-06-10T00:58:08

## Provenance

| Source file | Used for |
|---|---|
| `baseline_validation_metrics.csv` | Primary seed 42 (single run, 2 splits) |
| `baseline_repeated_split_metrics.csv` | n=20 repeated seeds (mean ± std, min, max) |
| `baseline_applicability_domain_summary.csv` | AD bucketing (in-domain / borderline / OOD) |
| `baseline_calibration_bins.csv` | Calibration reliability table |
| `baseline_rf_morgan_bit_importance.csv` | Top 10 Morgan bits + SMARTS |
| `baseline_validation_metadata.json` | Reproducibility config |

Filter applied everywhere: `feature_set == "morgan"` (champion).

---

## A. Primary Seed (42) — single run, 2 splits

| split   | roc_auc | pr_auc |   mcc | brier | bal_acc | sensit | specif |   tn |  fp |  fn |   tp |
|:--------|--------:|-------:|------:|------:|--------:|-------:|-------:|-----:|----:|----:|-----:|
| random  |  0.9885 |  0.9882 | 0.9112 | 0.0575 |   0.9556 |  0.9497 |  0.9615 | 899 | 36 | 47 | 887 |
| scaffold | 0.9734 | 0.9755 | 0.8001 | 0.0829 |   0.8973 |  0.8385 |  0.9561 | 893 | 41 | 151 | 784 |

**Reading guide:** `mcc` and `roc_auc` are the headline discrimination metrics. `brier` is **lower = better** (calibration). `sensit` vs `specif` shows whether the model leans toward false negatives (sensit) or false positives (specif).

---

## B. Repeated (n=20, seeds 0–19) — mean ± std, min, max

| split    | metric              |    mean |    std |     min |     max | arrow |
|:---------|:--------------------|--------:|-------:|--------:|--------:|:------|
| random   | roc_auc              |  0.9873 | 0.0018 |  0.9839 |  0.9909 | ↑ (higher better) |
| random   | pr_auc               |  0.9879 | 0.0015 |  0.9847 |  0.9903 | ↑ (higher better) |
| random   | mcc                  |  0.8880 | 0.0085 |  0.8680 |  0.9017 | ↑ (higher better) |
| random   | brier_score          |  0.0589 | 0.0019 |  0.0550 |  0.0624 | ↓ (lower better) |
| random   | balanced_accuracy    |  0.9439 | 0.0043 |  0.9336 |  0.9508 | ↑ (higher better) |

| scaffold | roc_auc              |  0.9728 | 0.0023 |  0.9675 |  0.9766 | ↑ (higher better) |
| scaffold | pr_auc               |  0.9751 | 0.0023 |  0.9704 |  0.9786 | ↑ (higher better) |
| scaffold | mcc                  |  0.7968 | 0.0071 |  0.7851 |  0.8099 | ↑ (higher better) |
| scaffold | brier_score          |  0.0846 | 0.0019 |  0.0813 |  0.0888 | ↓ (lower better) |
| scaffold | balanced_accuracy    |  0.8952 | 0.0035 |  0.8897 |  0.9027 | ↑ (higher better) |

**Reading guide:** `mean` is the value reported in publications; `std` quantifies seed-to-seed variability (lower = more stable); `min` shows the worst observed case across the 20 seeds.

---

## C. Applicability Domain (primary scaffold split)

| domain        |    n | fraction | mean_tani | accuracy | roc_auc |    mcc |
|:--------------|-----:|---------:|----------:|---------:|--------:|-------:|
| in-domain     | 1286 | 0.688 |     0.691 |    0.973 | 0.9942 | 0.9458 |
| borderline    | 362 | 0.194 |     0.423 |    0.702 | 0.8927 | 0.4553 |
| out-of-domain | 221 | 0.118 |     0.293 |    0.778 | 0.8307 | 0.4761 |

Total test: 1869 compounds.

**Reading guide:** `mean_tani` is the mean maximum Tanimoto similarity of each test compound to the training set. In-domain (≥0.5) compounds are predicted reliably; borderline (0.35–0.5) predictions need caution; OOD (<0.35) predictions should be flagged as low confidence.

---

## D. Calibration (scaffold, primary seed 42)

| bin | pred_lo | pred_hi | mean_pred | actual_freq |   n |      gap |
|----:|--------:|--------:|----------:|------------:|----:|---------:|
| 0 | 0.00 | 0.10 | 0.066 | 0.004 | 256 | +0.062 |
| 1 | 0.10 | 0.20 | 0.147 | 0.016 | 308 | +0.130 |
| 2 | 0.20 | 0.30 | 0.245 | 0.130 | 208 | +0.115 |
| 3 | 0.30 | 0.40 | 0.351 | 0.273 | 154 | +0.078 |
| 4 | 0.40 | 0.50 | 0.445 | 0.644 | 118 | -0.199 |
| 5 | 0.50 | 0.60 | 0.553 | 0.735 | 83 | -0.182 |
| 6 | 0.60 | 0.70 | 0.652 | 0.858 | 106 | -0.206 |
| 7 | 0.70 | 0.80 | 0.748 | 0.982 | 168 | -0.234 |
| 8 | 0.80 | 0.90 | 0.857 | 0.995 | 196 | -0.138 |
| 9 | 0.90 | 1.00 | 0.945 | 1.000 | 272 | -0.055 |

**Reading guide:** `gap = mean_pred − actual_freq`. A well-calibrated model has gap ≈ 0. Negative gap = model **over-predicts** (predicts higher probability than true positive rate). The Brier score for this split = **0.0829** (lower = better). Probability outputs are usable for ranking; absolute thresholding needs Platt or isotonic calibration.

---

## E. Top 10 Morgan Bits (RF feature importance, scaffold split)

| rank | bit_id | importance | substructure (SMARTS) |
|-----:|-------:|-----------:|:----------------------|
| 1 | 463 | 0.0185 | `[n+]` |
| 2 | 1357 | 0.0178 | `[#6]:[#6](:[#6]):[#6]` |
| 3 | 1019 | 0.0167 | `C` |
| 4 | 650 | 0.0152 | `O` |
| 5 | 1 | 0.0145 | `[#6]-[#6](=[#8])-[#6](:[#6]:[#6]):[#6](:[#6])-[#6]` |
| 6 | 1917 | 0.0143 | `[#6]=[#8]` |
| 7 | 1754 | 0.0137 | `[#6]-[#6](:[#6]):[#6]` |
| 8 | 1325 | 0.0129 | `[#6]-[#6]-[#6]` |
| 9 | 1855 | 0.0114 | `[#6]:[#7]:[#6]` |
| 10 | 117 | 0.0106 | `[#6]-[#7]-[#6]` |

**Reading guide:** Importance values are Gini impurity decreases from the random forest (scaffold split). Bit #463 `[n+]` (positively charged aromatic nitrogen) and bit #1357 (aromatic C) are the strongest G4 pharmacophore features — consistent with the literature for cationic G4 ligands.

---

## F. Reproducibility Config

| param              | value |
|:-------------------|:------|
| dataset            | {args_v.get('dataset', '?')} |
| rows               | {ctx.get('rows', '?')} |
| positives          | {ctx.get('positive', '?')} |
| negatives          | {ctx.get('negative', '?')} |
| fingerprint        | Morgan (radius=2, 2048 bits) |
| fp_bits            | {args_v.get('fingerprint_bits', '?')} |
| fp_radius          | {args_v.get('fingerprint_radius', '?')} |
| model              | RandomForestClassifier |
| n_trees            | {args_v.get('rf_trees', '?')} |
| min_samples_leaf   | 2 |
| class_weight       | balanced_subsample |
| max_features       | sqrt |
| primary_seed       | {args_v.get('primary_seed', '?')} |
| n_repeats          | {len(args_v.get('seeds', []))} |
| seeds              | {' '.join(map(str, args_v.get('seeds', [])))} |
| test_size          | {args_v.get('test_size', '?')} |
| candidates         | {' '.join(args_v.get('candidates', []))} |
| in_domain_thresh   | {args_v.get('in_domain_threshold', '?')} |
| borderline_thresh  | {args_v.get('borderline_threshold', '?')} |
| n_jobs             | {args_v.get('n_jobs', '?')} |
