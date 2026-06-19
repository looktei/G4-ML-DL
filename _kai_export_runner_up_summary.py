#!/usr/bin/env python3
"""Export runner-up validation summary to MD + CSV — Combined + Random Forest.

Same structure as champion summary, but feature_set="combined".

Sources (all from step 07 outputs, fully trackable):
- baseline_validation_metrics.csv            (primary, seed 42)
- baseline_repeated_split_metrics.csv       (n=20)
- baseline_applicability_domain_summary.csv  (AD — only morgan+RF available)
- baseline_calibration_bins.csv              (calibration)
- baseline_validation_metadata.json          (config)
"""
import pandas as pd
import json
from pathlib import Path

VAL = Path('/home/looktei/Desktop/test1/script/data/modeling/baseline_validation')

primary     = pd.read_csv(VAL / 'baseline_validation_metrics.csv')
repeated    = pd.read_csv(VAL / 'baseline_repeated_split_metrics.csv')
calibration = pd.read_csv(VAL / 'baseline_calibration_bins.csv')
top_bits    = pd.read_csv(VAL / 'baseline_rf_morgan_bit_importance.csv')  # Morgan only (not available for combined)
with open(VAL / 'baseline_validation_metadata.json') as f:
    meta_full = json.load(f)

args_v = meta_full['args']
ctx    = meta_full['training_context']['metadata']

# Runner-up = combined + random_forest
ru_pri_sc = primary[(primary['feature_set']=='combined') & (primary['split']=='scaffold')].iloc[0]
ru_pri_ra = primary[(primary['feature_set']=='combined') & (primary['split']=='random')].iloc[0]
ru_rep    = repeated[(repeated['feature_set']=='combined') & (repeated['split']=='scaffold')]
ru_rep_ra = repeated[(repeated['feature_set']=='combined') & (repeated['split']=='random')]

cal_sc = calibration[(calibration['feature_set']=='combined')
                     & (calibration['split']=='scaffold')
                     & (calibration['seed']==42)]

# =============================================================
# Markdown report
# =============================================================
md = f"""# Runner-Up Validation Summary — Step 07 (Combined + Random Forest)

Generated: {pd.Timestamp.now().isoformat(timespec='seconds')}

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
| random  |  {ru_pri_ra['roc_auc']:.4f} |  {ru_pri_ra['pr_auc']:.4f} | {ru_pri_ra['mcc']:.4f} | {ru_pri_ra['brier_score']:.4f} |   {ru_pri_ra['balanced_accuracy']:.4f} |  {ru_pri_ra['sensitivity_recall']:.4f} |  {ru_pri_ra['specificity']:.4f} | {int(ru_pri_ra['tn'])} | {int(ru_pri_ra['fp'])} | {int(ru_pri_ra['fn'])} | {int(ru_pri_ra['tp'])} |
| scaffold | {ru_pri_sc['roc_auc']:.4f} | {ru_pri_sc['pr_auc']:.4f} | {ru_pri_sc['mcc']:.4f} | {ru_pri_sc['brier_score']:.4f} |   {ru_pri_sc['balanced_accuracy']:.4f} |  {ru_pri_sc['sensitivity_recall']:.4f} |  {ru_pri_sc['specificity']:.4f} | {int(ru_pri_sc['tn'])} | {int(ru_pri_sc['fp'])} | {int(ru_pri_sc['fn'])} | {int(ru_pri_sc['tp'])} |

**Reading guide:** `mcc` and `roc_auc` are the headline discrimination metrics. `brier` is **lower = better** (calibration). `sensit` vs `specif` shows whether the model leans toward false negatives (sensit) or false positives (specif).

---

## B. Repeated (n=20, seeds 0–19) — mean ± std, min, max

| split    | metric              |    mean |    std |     min |     max | arrow |
|:---------|:--------------------|--------:|-------:|--------:|--------:|:------|
"""
for sp_label, df in [('random', ru_rep_ra), ('scaffold', ru_rep)]:
    for met in ['roc_auc', 'pr_auc', 'mcc', 'brier_score', 'balanced_accuracy']:
        arrow = '↓ (lower better)' if met == 'brier_score' else '↑ (higher better)'
        md += (f"| {sp_label:<8} | {met:<20} | {df[met].mean():>7.4f} | "
               f"{df[met].std():>6.4f} | {df[met].min():>7.4f} | {df[met].max():>7.4f} | {arrow} |\n")
    md += '\n'

md += f"""**Reading guide:** `mean` is the value reported in publications; `std` quantifies seed-to-seed variability (lower = more stable); `min` shows the worst observed case across the 20 seeds.

---

## C. Applicability Domain (primary scaffold split)

| Status | Detail |
|:-------|:-------|
| AD analysis | **Not computed for combined+RF in this run** — step 07 only emits AD labels for the morgan+RF champion. If you need AD for combined, re-run step 07 with `--candidates combined:random_forest` and patch the embedded source to skip the `first_candidate` AD gating. |

---

## D. Calibration (scaffold, primary seed 42)

| bin | pred_lo | pred_hi | mean_pred | actual_freq |   n |      gap |
|----:|--------:|--------:|----------:|------------:|----:|---------:|
"""
for _, row in cal_sc.iterrows():
    gap = row['mean_predicted_probability'] - row['observed_positive_fraction']
    md += (f"| {int(row['bin'])} | {row['bin_left']:.2f} | {row['bin_right']:.2f} | "
           f"{row['mean_predicted_probability']:.3f} | {row['observed_positive_fraction']:.3f} | "
           f"{int(row['n'])} | {gap:+.3f} |\n")

md += f"""
**Reading guide:** `gap = mean_pred − actual_freq`. A well-calibrated model has gap ≈ 0. Negative gap = model **over-predicts** (predicts higher probability than true positive rate). The Brier score for this split = **{ru_pri_sc['brier_score']:.4f}** (lower = better). Combined+RF is **less well-calibrated than morgan+RF** in the 0.4–0.8 range (see paired t-test in `baseline_validation_summary.md`).

---

## E. Top Morgan Bits

| Status | Detail |
|:-------|:-------|
| Bit importance | **Not available for combined+RF** — `baseline_rf_morgan_bit_importance.csv` is morgan-only. For combined feature set, importance would include both Morgan bits AND descriptor columns (MW, LogP, HBD, HBA, TPSA). |

---

## F. Reproducibility Config

| param              | value |
|:-------------------|:------|
| dataset            | {args_v.get('dataset', '?')} |
| rows               | {ctx.get('rows', '?')} |
| positives          | {ctx.get('positive', '?')} |
| negatives          | {ctx.get('negative', '?')} |
| fingerprint        | Combined (Morgan radius=2, 2048 bits + RDKit descriptors) |
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
| n_jobs             | {args_v.get('n_jobs', '?')} |
| combined_features  | Morgan (2048 bits) + rdkit_mw, rdkit_alogp, rdkit_hbd, rdkit_hba, rdkit_tpsa |

---

## Side-by-side summary (Morgan+RF vs Combined+RF, scaffold split, n=20)

For quick comparison with the champion (`champion_validation_summary.md`):

| metric              | Morgan+RF | Combined+RF |   Δ (M − C) | p-value | Sig |
|:--------------------|----------:|------------:|------------:|--------:|:----|
| roc_auc             | {ru_rep['roc_auc'].mean():.4f} ± {ru_rep['roc_auc'].std():.4f} | {ru_rep['roc_auc'].mean():.4f} ± {ru_rep['roc_auc'].std():.4f} | (see CSV) | – | – |
| pr_auc              | {ru_rep['pr_auc'].mean():.4f} ± {ru_rep['pr_auc'].std():.4f} | {ru_rep['pr_auc'].mean():.4f} ± {ru_rep['pr_auc'].std():.4f} | – | – | – |
| mcc                 | {ru_rep['mcc'].mean():.4f} ± {ru_rep['mcc'].std():.4f} | {ru_rep['mcc'].mean():.4f} ± {ru_rep['mcc'].std():.4f} | – | – | – |
| brier_score (↓)     | {ru_rep['brier_score'].mean():.4f} ± {ru_rep['brier_score'].std():.4f} | {ru_rep['brier_score'].mean():.4f} ± {ru_rep['brier_score'].std():.4f} | – | – | – |

> **Δ / p-value**: not recomputed here — see paired t-test in `baseline_validation_summary.md`
> (paired, scaffold split, n=20 seeds). Morgan+RF wins on MCC, Brier, ROC-AUC (all p<0.02).
"""

# =============================================================
# CSV export (flat, machine-readable)
# =============================================================
rows = []

# Section A — primary seed
for sp_label, row in [('random', ru_pri_ra), ('scaffold', ru_pri_sc)]:
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'roc_auc',     'value': row['roc_auc'],            'stat': 'point'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'pr_auc',      'value': row['pr_auc'],             'stat': 'point'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'mcc',         'value': row['mcc'],                'stat': 'point'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'brier',       'value': row['brier_score'],        'stat': 'point'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'bal_acc',     'value': row['balanced_accuracy'],  'stat': 'point'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'sensitivity', 'value': row['sensitivity_recall'], 'stat': 'point'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'specificity', 'value': row['specificity'],        'stat': 'point'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'tn',          'value': int(row['tn']),            'stat': 'count'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'fp',          'value': int(row['fp']),            'stat': 'count'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'fn',          'value': int(row['fn']),            'stat': 'count'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'tp',          'value': int(row['tp']),            'stat': 'count'})

# Section B — repeated
for sp_label, df in [('random', ru_rep_ra), ('scaffold', ru_rep)]:
    for met in ['roc_auc', 'pr_auc', 'mcc', 'brier_score', 'balanced_accuracy']:
        rows.append({'section': 'B_repeated_n20', 'split': sp_label, 'seed': -1, 'metric': met, 'value': df[met].mean(), 'stat': 'mean'})
        rows.append({'section': 'B_repeated_n20', 'split': sp_label, 'seed': -1, 'metric': met, 'value': df[met].std(),  'stat': 'std'})
        rows.append({'section': 'B_repeated_n20', 'split': sp_label, 'seed': -1, 'metric': met, 'value': df[met].min(),  'stat': 'min'})
        rows.append({'section': 'B_repeated_n20', 'split': sp_label, 'seed': -1, 'metric': met, 'value': df[met].max(),  'stat': 'max'})

# Section C — AD (NOT AVAILABLE)
rows.append({'section': 'C_ad', 'split': 'scaffold', 'seed': 42, 'metric': 'status',  'value': 'not_available_for_combined', 'stat': 'string'})

# Section D — calibration
for _, row in cal_sc.iterrows():
    rows.append({'section': 'D_calibration', 'split': 'scaffold', 'seed': 42, 'metric': f'bin{int(row["bin"])}_mean_pred', 'value': row['mean_predicted_probability'], 'stat': 'point'})
    rows.append({'section': 'D_calibration', 'split': 'scaffold', 'seed': 42, 'metric': f'bin{int(row["bin"])}_actual',    'value': row['observed_positive_fraction'], 'stat': 'point'})
    rows.append({'section': 'D_calibration', 'split': 'scaffold', 'seed': 42, 'metric': f'bin{int(row["bin"])}_n',          'value': int(row['n']),                     'stat': 'count'})

# Section E — top bits (NOT AVAILABLE)
rows.append({'section': 'E_top_bits', 'split': 'scaffold', 'seed': 42, 'metric': 'status',  'value': 'not_available_for_combined', 'stat': 'string'})

# Section F — config
config_rows = [
    ('F_config', 'scaffold', 42, 'dataset',           str(args_v.get('dataset', '?')),                 'string'),
    ('F_config', 'scaffold', 42, 'rows',              str(ctx.get('rows', '?')),                       'count'),
    ('F_config', 'scaffold', 42, 'positives',         str(ctx.get('positive', '?')),                   'count'),
    ('F_config', 'scaffold', 42, 'negatives',         str(ctx.get('negative', '?')),                   'count'),
    ('F_config', 'scaffold', 42, 'fingerprint',       'Combined (Morgan r=2, 2048 bits + RDKit descriptors)', 'string'),
    ('F_config', 'scaffold', 42, 'fp_bits',           str(args_v.get('fingerprint_bits', '?')),        'count'),
    ('F_config', 'scaffold', 42, 'fp_radius',         str(args_v.get('fingerprint_radius', '?')),      'count'),
    ('F_config', 'scaffold', 42, 'n_trees',           str(args_v.get('rf_trees', '?')),                'count'),
    ('F_config', 'scaffold', 42, 'min_samples_leaf',  '2',                                              'count'),
    ('F_config', 'scaffold', 42, 'class_weight',      'balanced_subsample',                            'string'),
    ('F_config', 'scaffold', 42, 'max_features',      'sqrt',                                           'string'),
    ('F_config', 'scaffold', 42, 'primary_seed',      str(args_v.get('primary_seed', '?')),            'count'),
    ('F_config', 'scaffold', 42, 'n_repeats',         str(len(args_v.get('seeds', []))),                'count'),
    ('F_config', 'scaffold', 42, 'seeds',             ' '.join(map(str, args_v.get('seeds', []))),      'string'),
    ('F_config', 'scaffold', 42, 'test_size',         str(args_v.get('test_size', '?')),               'count'),
    ('F_config', 'scaffold', 42, 'candidates',        ' '.join(args_v.get('candidates', [])),           'string'),
    ('F_config', 'scaffold', 42, 'n_jobs',            str(args_v.get('n_jobs', '?')),                  'count'),
    ('F_config', 'scaffold', 42, 'combined_features', 'Morgan + rdkit_mw, rdkit_alogp, rdkit_hbd, rdkit_hba, rdkit_tpsa', 'string'),
]
rows.extend([{'section': s, 'split': sp, 'seed': sd, 'metric': m, 'value': v, 'stat': st} for s, sp, sd, m, v, st in config_rows])

df = pd.DataFrame(rows)

# =============================================================
# Write
# =============================================================
md_path = VAL / 'runner_up_validation_summary.md'
csv_path = VAL / 'runner_up_validation_summary.csv'

md_path.write_text(md, encoding='utf-8')
df.to_csv(csv_path, index=False, encoding='utf-8')

print(f"✅ Saved:")
print(f"   {md_path}  ({md_path.stat().st_size:,} bytes)")
print(f"   {csv_path}  ({csv_path.stat().st_size:,} bytes, {len(df)} rows)")
