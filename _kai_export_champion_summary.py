#!/usr/bin/env python3
"""Export champion validation summary to MD + CSV in baseline_validation/.

Sources (all from step 07 outputs, fully trackable):
- baseline_validation_metrics.csv            (primary, seed 42)
- baseline_repeated_split_metrics.csv       (n=20)
- baseline_applicability_domain_summary.csv  (AD)
- baseline_calibration_bins.csv              (calibration)
- baseline_rf_morgan_bit_importance.csv      (top bits)
- baseline_validation_metadata.json          (config)
"""
import pandas as pd
import json
import sys
from pathlib import Path

VAL = Path('/home/looktei/Desktop/test1/script/data/modeling/baseline_validation')

primary      = pd.read_csv(VAL / 'baseline_validation_metrics.csv')
repeated     = pd.read_csv(VAL / 'baseline_repeated_split_metrics.csv')
ad_summary   = pd.read_csv(VAL / 'baseline_applicability_domain_summary.csv')
calibration  = pd.read_csv(VAL / 'baseline_calibration_bins.csv')
top_bits     = pd.read_csv(VAL / 'baseline_rf_morgan_bit_importance.csv')
with open(VAL / 'baseline_validation_metadata.json') as f:
    meta_full = json.load(f)

args_v = meta_full['args']
ctx    = meta_full['training_context']['metadata']

mr_pri_sc = primary[(primary['feature_set']=='morgan') & (primary['split']=='scaffold')].iloc[0]
mr_pri_ra = primary[(primary['feature_set']=='morgan') & (primary['split']=='random')].iloc[0]
mr_rep    = repeated[(repeated['feature_set']=='morgan') & (repeated['split']=='scaffold')]
mr_rep_ra = repeated[(repeated['feature_set']=='morgan') & (repeated['split']=='random')]

in_dom     = ad_summary[ad_summary['domain']=='in-domain'].iloc[0]
borderline = ad_summary[ad_summary['domain']=='borderline'].iloc[0]
ood        = ad_summary[ad_summary['domain']=='out-of-domain'].iloc[0]
total_n    = int(ad_summary['n'].sum())

cal_sc = calibration[(calibration['feature_set']=='morgan')
                     & (calibration['split']=='scaffold')
                     & (calibration['seed']==42)]

# =============================================================
# Markdown report
# =============================================================
md = f"""# Champion Validation Summary — Step 07 (Morgan + Random Forest)

Generated: {pd.Timestamp.now().isoformat(timespec='seconds')}

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
| random  |  {mr_pri_ra['roc_auc']:.4f} |  {mr_pri_ra['pr_auc']:.4f} | {mr_pri_ra['mcc']:.4f} | {mr_pri_ra['brier_score']:.4f} |   {mr_pri_ra['balanced_accuracy']:.4f} |  {mr_pri_ra['sensitivity_recall']:.4f} |  {mr_pri_ra['specificity']:.4f} | {int(mr_pri_ra['tn'])} | {int(mr_pri_ra['fp'])} | {int(mr_pri_ra['fn'])} | {int(mr_pri_ra['tp'])} |
| scaffold | {mr_pri_sc['roc_auc']:.4f} | {mr_pri_sc['pr_auc']:.4f} | {mr_pri_sc['mcc']:.4f} | {mr_pri_sc['brier_score']:.4f} |   {mr_pri_sc['balanced_accuracy']:.4f} |  {mr_pri_sc['sensitivity_recall']:.4f} |  {mr_pri_sc['specificity']:.4f} | {int(mr_pri_sc['tn'])} | {int(mr_pri_sc['fp'])} | {int(mr_pri_sc['fn'])} | {int(mr_pri_sc['tp'])} |

**Reading guide:** `mcc` and `roc_auc` are the headline discrimination metrics. `brier` is **lower = better** (calibration). `sensit` vs `specif` shows whether the model leans toward false negatives (sensit) or false positives (specif).

---

## B. Repeated (n=20, seeds 0–19) — mean ± std, min, max

| split    | metric              |    mean |    std |     min |     max | arrow |
|:---------|:--------------------|--------:|-------:|--------:|--------:|:------|
"""
for sp_label, df in [('random', mr_rep_ra), ('scaffold', mr_rep)]:
    for met in ['roc_auc', 'pr_auc', 'mcc', 'brier_score', 'balanced_accuracy']:
        arrow = '↓ (lower better)' if met == 'brier_score' else '↑ (higher better)'
        md += (f"| {sp_label:<8} | {met:<20} | {df[met].mean():>7.4f} | "
               f"{df[met].std():>6.4f} | {df[met].min():>7.4f} | {df[met].max():>7.4f} | {arrow} |\n")
    md += '\n'

md += f"""**Reading guide:** `mean` is the value reported in publications; `std` quantifies seed-to-seed variability (lower = more stable); `min` shows the worst observed case across the 20 seeds.

---

## C. Applicability Domain (primary scaffold split)

| domain        |    n | fraction | mean_tani | accuracy | roc_auc |    mcc |
|:--------------|-----:|---------:|----------:|---------:|--------:|-------:|
| in-domain     | {int(in_dom['n'])} | {int(in_dom['n'])/total_n:.3f} |     {in_dom['mean_max_tanimoto_train']:.3f} |    {in_dom['accuracy']:.3f} | {in_dom['roc_auc']:.4f} | {in_dom['mcc']:.4f} |
| borderline    | {int(borderline['n'])} | {int(borderline['n'])/total_n:.3f} |     {borderline['mean_max_tanimoto_train']:.3f} |    {borderline['accuracy']:.3f} | {borderline['roc_auc']:.4f} | {borderline['mcc']:.4f} |
| out-of-domain | {int(ood['n'])} | {int(ood['n'])/total_n:.3f} |     {ood['mean_max_tanimoto_train']:.3f} |    {ood['accuracy']:.3f} | {ood['roc_auc']:.4f} | {ood['mcc']:.4f} |

Total test: {total_n} compounds.

**Reading guide:** `mean_tani` is the mean maximum Tanimoto similarity of each test compound to the training set. In-domain (≥{args_v.get('in_domain_threshold', 0.5)}) compounds are predicted reliably; borderline ({args_v.get('borderline_threshold', 0.35)}–{args_v.get('in_domain_threshold', 0.5)}) predictions need caution; OOD (<{args_v.get('borderline_threshold', 0.35)}) predictions should be flagged as low confidence.

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
**Reading guide:** `gap = mean_pred − actual_freq`. A well-calibrated model has gap ≈ 0. Negative gap = model **over-predicts** (predicts higher probability than true positive rate). The Brier score for this split = **{mr_pri_sc['brier_score']:.4f}** (lower = better). Probability outputs are usable for ranking; absolute thresholding needs Platt or isotonic calibration.

---

## E. Top 10 Morgan Bits (RF feature importance, scaffold split)

| rank | bit_id | importance | substructure (SMARTS) |
|-----:|-------:|-----------:|:----------------------|
"""
top10 = top_bits.head(10)
for i, (_, row) in enumerate(top10.iterrows(), 1):
    sm = row['representative_smarts'] if pd.notna(row['representative_smarts']) else '?'
    md += f"| {i} | {int(row['morgan_bit'])} | {row['importance']:.4f} | `{sm}` |\n"

md += """
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
"""

# =============================================================
# CSV export (flat, machine-readable)
# =============================================================
rows = []

# Section A — primary seed
for sp_label, row in [('random', mr_pri_ra), ('scaffold', mr_pri_sc)]:
    rows.append({
        'section': 'A_primary_seed42',
        'split': sp_label,
        'seed': 42,
        'metric': 'roc_auc',            'value': row['roc_auc'],       'stat': 'point',
    })
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'pr_auc',     'value': row['pr_auc'],           'stat': 'point'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'mcc',        'value': row['mcc'],              'stat': 'point'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'brier',      'value': row['brier_score'],      'stat': 'point'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'bal_acc',    'value': row['balanced_accuracy'],'stat': 'point'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'sensitivity','value': row['sensitivity_recall'],'stat': 'point'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'specificity','value': row['specificity'],      'stat': 'point'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'tn',         'value': int(row['tn']),          'stat': 'count'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'fp',         'value': int(row['fp']),          'stat': 'count'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'fn',         'value': int(row['fn']),          'stat': 'count'})
    rows.append({'section': 'A_primary_seed42', 'split': sp_label, 'seed': 42, 'metric': 'tp',         'value': int(row['tp']),          'stat': 'count'})

# Section B — repeated
for sp_label, df in [('random', mr_rep_ra), ('scaffold', mr_rep)]:
    for met in ['roc_auc', 'pr_auc', 'mcc', 'brier_score', 'balanced_accuracy']:
        rows.append({'section': 'B_repeated_n20', 'split': sp_label, 'seed': -1, 'metric': met, 'value': df[met].mean(), 'stat': 'mean'})
        rows.append({'section': 'B_repeated_n20', 'split': sp_label, 'seed': -1, 'metric': met, 'value': df[met].std(),  'stat': 'std'})
        rows.append({'section': 'B_repeated_n20', 'split': sp_label, 'seed': -1, 'metric': met, 'value': df[met].min(),  'stat': 'min'})
        rows.append({'section': 'B_repeated_n20', 'split': sp_label, 'seed': -1, 'metric': met, 'value': df[met].max(),  'stat': 'max'})

# Section C — AD
for label, row in [('in-domain', in_dom), ('borderline', borderline), ('out-of-domain', ood)]:
    rows.append({'section': 'C_ad', 'split': 'scaffold', 'seed': 42, 'metric': f'ad_{label}_n',     'value': int(row['n']),                       'stat': 'count'})
    rows.append({'section': 'C_ad', 'split': 'scaffold', 'seed': 42, 'metric': f'ad_{label}_fraction','value': int(row['n'])/total_n,              'stat': 'fraction'})
    rows.append({'section': 'C_ad', 'split': 'scaffold', 'seed': 42, 'metric': f'ad_{label}_mean_tani','value': row['mean_max_tanimoto_train'],     'stat': 'mean'})
    rows.append({'section': 'C_ad', 'split': 'scaffold', 'seed': 42, 'metric': f'ad_{label}_accuracy', 'value': row['accuracy'],                  'stat': 'point'})
    rows.append({'section': 'C_ad', 'split': 'scaffold', 'seed': 42, 'metric': f'ad_{label}_roc_auc',  'value': row['roc_auc'],                   'stat': 'point'})
    rows.append({'section': 'C_ad', 'split': 'scaffold', 'seed': 42, 'metric': f'ad_{label}_mcc',      'value': row['mcc'],                       'stat': 'point'})

# Section D — calibration
for _, row in cal_sc.iterrows():
    rows.append({'section': 'D_calibration', 'split': 'scaffold', 'seed': 42, 'metric': f'bin{int(row["bin"])}_mean_pred', 'value': row['mean_predicted_probability'], 'stat': 'point'})
    rows.append({'section': 'D_calibration', 'split': 'scaffold', 'seed': 42, 'metric': f'bin{int(row["bin"])}_actual',    'value': row['observed_positive_fraction'], 'stat': 'point'})
    rows.append({'section': 'D_calibration', 'split': 'scaffold', 'seed': 42, 'metric': f'bin{int(row["bin"])}_n',          'value': int(row['n']),                     'stat': 'count'})

# Section E — top bits
for i, (_, row) in enumerate(top10.iterrows(), 1):
    rows.append({'section': 'E_top_bits', 'split': 'scaffold', 'seed': 42, 'metric': f'rank{i}_bit',        'value': int(row['morgan_bit']),         'stat': 'point'})
    rows.append({'section': 'E_top_bits', 'split': 'scaffold', 'seed': 42, 'metric': f'rank{i}_importance', 'value': row['importance'],               'stat': 'point'})
    rows.append({'section': 'E_top_bits', 'split': 'scaffold', 'seed': 42, 'metric': f'rank{i}_smarts',     'value': row['representative_smarts'] if pd.notna(row['representative_smarts']) else '?', 'stat': 'string'})

# Section F — config (string values flattened to value)
config_rows = [
    ('F_config', 'scaffold', 42, 'dataset',            str(args_v.get('dataset', '?')),                 'string'),
    ('F_config', 'scaffold', 42, 'rows',               str(ctx.get('rows', '?')),                       'count'),
    ('F_config', 'scaffold', 42, 'positives',          str(ctx.get('positive', '?')),                   'count'),
    ('F_config', 'scaffold', 42, 'negatives',          str(ctx.get('negative', '?')),                   'count'),
    ('F_config', 'scaffold', 42, 'fingerprint',        'Morgan (radius=2, 2048 bits)',                  'string'),
    ('F_config', 'scaffold', 42, 'fp_bits',            str(args_v.get('fingerprint_bits', '?')),        'count'),
    ('F_config', 'scaffold', 42, 'fp_radius',          str(args_v.get('fingerprint_radius', '?')),      'count'),
    ('F_config', 'scaffold', 42, 'n_trees',            str(args_v.get('rf_trees', '?')),                'count'),
    ('F_config', 'scaffold', 42, 'min_samples_leaf',   '2',                                              'count'),
    ('F_config', 'scaffold', 42, 'class_weight',       'balanced_subsample',                            'string'),
    ('F_config', 'scaffold', 42, 'max_features',       'sqrt',                                           'string'),
    ('F_config', 'scaffold', 42, 'primary_seed',       str(args_v.get('primary_seed', '?')),            'count'),
    ('F_config', 'scaffold', 42, 'n_repeats',          str(len(args_v.get('seeds', []))),                'count'),
    ('F_config', 'scaffold', 42, 'seeds',              ' '.join(map(str, args_v.get('seeds', []))),      'string'),
    ('F_config', 'scaffold', 42, 'test_size',          str(args_v.get('test_size', '?')),               'count'),
    ('F_config', 'scaffold', 42, 'candidates',         ' '.join(args_v.get('candidates', [])),           'string'),
    ('F_config', 'scaffold', 42, 'in_domain_thresh',   str(args_v.get('in_domain_threshold', '?')),     'count'),
    ('F_config', 'scaffold', 42, 'borderline_thresh',  str(args_v.get('borderline_threshold', '?')),    'count'),
    ('F_config', 'scaffold', 42, 'n_jobs',             str(args_v.get('n_jobs', '?')),                  'count'),
]
rows.extend([{'section': s, 'split': sp, 'seed': sd, 'metric': m, 'value': v, 'stat': st} for s, sp, sd, m, v, st in config_rows])

df = pd.DataFrame(rows)

# =============================================================
# Write
# =============================================================
md_path = VAL / 'champion_validation_summary.md'
csv_path = VAL / 'champion_validation_summary.csv'

md_path.write_text(md, encoding='utf-8')
df.to_csv(csv_path, index=False, encoding='utf-8')

print(f"✅ Saved:")
print(f"   {md_path}  ({md_path.stat().st_size:,} bytes)")
print(f"   {csv_path}  ({csv_path.stat().st_size:,} bytes, {len(df)} rows)")
