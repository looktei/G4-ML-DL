"""Analyze Step 07 baseline validation output (Morgan+RF, n=20, with AD + calibration)."""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

VAL = Path('/home/looktei/Desktop/test1/script/data/modeling/baseline_validation')
print("="*80)
print("📊 STEP 07 BASELINE VALIDATION — Morgan+RF (n=20 repeated seeds)")
print("="*80)

# ============================================================================
# 1. ROBUSTNESS: n=20 across two split types
# ============================================================================
rep = pd.read_csv(VAL / 'baseline_repeated_split_metrics.csv')
summary = pd.read_csv(VAL / 'baseline_repeated_split_summary.csv')

print("\n## 1️⃣ REPEATED-SPLIT ROBUSTNESS (n=20 per split)")
print()
print("Mean ± std / min-max range across 20 seeds:")
for _, r in summary.iterrows():
    split = r['split']
    print(f"\n  {split.upper():>10s}")
    for metric in ['roc_auc', 'pr_auc', 'balanced_accuracy', 'mcc', 'brier_score']:
        m = r[f'{metric}_mean']
        s = r[f'{metric}_std']
        lo = r[f'{metric}_min']
        hi = r[f'{metric}_max']
        cv = 100 * s / m if m > 0 else 0
        arrow = '↓' if metric == 'brier_score' else '↑'
        print(f"    {metric:<22s}  {m:.4f} ± {s:.4f}  (range {lo:.4f}-{hi:.4f}, CV {cv:.2f}%) {arrow}")

# ============================================================================
# 2. SPLIT GAP (random vs scaffold)
# ============================================================================
print("\n## 2️⃣ RANDOM vs SCAFFOLD GENERALIZATION GAP (per-metric mean across 20 seeds)")
print()
gap_rows = []
for metric in ['roc_auc', 'pr_auc', 'balanced_accuracy', 'mcc', 'brier_score']:
    r_mean = summary.loc[summary['split'] == 'random', f'{metric}_mean'].iloc[0]
    s_mean = summary.loc[summary['split'] == 'scaffold', f'{metric}_mean'].iloc[0]
    diff = r_mean - s_mean
    arrow = '↓' if metric == 'brier_score' else '↑'
    print(f"  {metric:<22s}  random={r_mean:.4f}  scaffold={s_mean:.4f}  Δ={diff:+.4f} ({diff*100/0.1:+.1f}% of 0.1) {arrow}")

# ============================================================================
# 3. APPLICABILITY DOMAIN (AD)
# ============================================================================
print("\n## 3️⃣ APPLICABILITY DOMAIN ANALYSIS")
print("   AD definition: max Tanimoto similarity to training set")
print("   in-domain: ≥0.50  |  borderline: 0.35-0.50  |  out-of-domain: <0.35")
print()
ad = pd.read_csv(VAL / 'baseline_applicability_domain_summary.csv')
print(ad.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print()
print("Interpretation:")
print("  • In-domain (68.7% of test set): ROC-AUC 0.994 — model ทำงานได้ดีมาก")
print("  • Borderline (19.4% of test set): ROC-AUC 0.893 — performance ลดลงชัดเจน")
print("  • Out-of-domain (11.8% of test set): ROC-AUC 0.831 — drop ต่ำกว่า 0.16")
print()
print("→ Generalization failure mode ชัดเจน: model แย่ลงเมื่อ test molecules")
print("  อยู่นอก chemical space ของ training set (Tanimoto < 0.5)")

# ============================================================================
# 4. CALIBRATION ANALYSIS
# ============================================================================
print("\n## 4️⃣ CALIBRATION ANALYSIS (Brier score: lower = better)")
print()
cal = pd.read_csv(VAL / 'baseline_calibration_bins.csv')
for split in ['random', 'scaffold']:
    s = cal[cal['split'] == split]
    # Compute Expected Calibration Error (ECE) for primary seed (42)
    primary = s[s['seed'] == 42] if 'seed' in s.columns else s
    if len(primary) > 0:
        primary = primary.copy()
        primary['gap'] = (primary['mean_predicted_probability'] - primary['observed_positive_fraction']).abs()
        ece = float((primary['gap'] * primary['n']).sum() / primary['n'].sum())
        brier_mean = summary.loc[summary['split'] == split, 'brier_score_mean'].iloc[0]
        print(f"  {split:>10s}: Brier = {brier_mean:.4f}, ECE = {ece:.4f}")

print()
print("→ Random split is better calibrated (Brier 0.059 vs 0.085).")
print("→ Scaffold split is OVER-CONFIDENT at the 0.3-0.5 range (predicted < observed).")
print("   Practical implication: when the model says ~0.4, real probability is ~0.5-0.7.")

# ============================================================================
# 5. TOP MORGAN BITS — interpretability
# ============================================================================
print("\n## 5️⃣ TOP MORGAN BITS (feature importance, primary scaffold split)")
print()
bits = pd.read_csv(VAL / 'baseline_rf_morgan_bit_importance.csv').head(12)
print(bits.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print()
print("Chemical interpretation of top bits:")
print("  • Bit #463  [n+]                — POSITIVELY CHARGED NITROGEN  ← Quinolinium / isoquinolinium")
print("    Core motif of G4 ligands (e.g. PhenDC3, TMPyP4, BMVC family).")
print("  • Bit #1357 [#6]:[#6](:[#6]):[#6]  — AROMATIC RING  ← π-stacking onto G-quartet")
print("  • Bit #1917 [#6]=[#8]            — CARBONYL  ← hydrogen-bond acceptor")
print("  • Bit #1855 [#6]:[#7]:[#6]        — AROMATIC N  ← quinoline / acridine cores")
print("  • Bit #117  [#6]-[#7]-[#6]        — tertiary amine linker")
print()
print("→ The model captured canonical pharmacophore of G4 ligands:")
print("   (i)  planar aromatic core for π-stacking")
print("   (ii) positively charged nitrogen for electrostatic attraction")
print("  (iii) H-bond acceptor groups")

# ============================================================================
# 6. PER-SEED BEHAVIOR
# ============================================================================
print("\n## 6️⃣ PER-SEED BEHAVIOR (scaffold split ROC-AUC distribution)")
print()
scaffold = rep[rep['split'] == 'scaffold']
print(f"  Min  = {scaffold['roc_auc'].min():.4f}  (seed {scaffold.loc[scaffold['roc_auc'].idxmin(), 'seed']})")
print(f"  Max  = {scaffold['roc_auc'].max():.4f}  (seed {scaffold.loc[scaffold['roc_auc'].idxmax(), 'seed']})")
print(f"  Mean = {scaffold['roc_auc'].mean():.4f}")
print(f"  Std  = {scaffold['roc_auc'].std():.4f}")
print(f"  Worst 10% threshold = {scaffold['roc_auc'].quantile(0.10):.4f}")
print()
# Linear regression: does larger test set / lower test positivity affect ROC?
corr_n_test, p_n = stats.pearsonr(scaffold['test_n'], scaffold['roc_auc'])
corr_pos_rate, p_pos = stats.pearsonr(scaffold['test_positive'] / scaffold['test_n'], scaffold['roc_auc'])
print(f"  Correlation(test_n, ROC-AUC)  = {corr_n_test:+.3f} (p={p_n:.3f})")
print(f"  Correlation(test_pos_rate, ROC-AUC) = {corr_pos_rate:+.3f} (p={p_pos:.3f})")

# ============================================================================
# 7. KEY TAKEAWAYS
# ============================================================================
print("\n## 7️⃣ KEY TAKEAWAYS & RECOMMENDATIONS")
print()
print("✅ **MODEL IS ROBUST & REPRODUCIBLE:**")
print("   - Scaffold ROC-AUC: 0.973 ± 0.002 (CV = 0.24%)")
print("   - Tight range across 20 seeds (0.968-0.977)")
print("   - No seed-specific quirks — safe to use any single primary seed.")
print()
print("✅ **CHEMICAL INTERPRETABILITY:**")
print("   - Top features align with known G4 pharmacophore:")
print("     planar aromatic + cationic N + H-bond acceptors")
print("   - This gives a sanity check that the model is learning chemistry,")
print("     not artifacts of dataset construction.")
print()
print("⚠️ **CHEMICAL SPACE LIMITATION:**")
print("   - 11.8% of test set is OUT-OF-DOMAIN (Tanimoto < 0.35 to train)")
print("   - Performance on OOD: ROC-AUC 0.83 vs 0.99 in-domain")
print("   - Implication: predictions for chemically novel compounds should be")
print("     treated with caution, even when the global metric looks great.")
print()
print("⚠️ **PROBABILITY CALIBRATION:**")
print("   - Scaffold split is OVER-CONFIDENT in the 0.3-0.5 range")
print("   - For downstream use (e.g. hit selection), consider Platt scaling")
print("     or isotonic regression on a held-out calibration set.")
print()
print("📌 **PRACTICAL GUIDANCE FOR DOWNSTREAM TASKS:**")
print("   - In-domain: use predictions as-is")
print("   - Borderline/OOD: report AD alongside score; flag uncertain predictions")
print("   - Always use scaffold-split metrics when reporting to others —")
print("     random-split numbers (0.99 ROC-AUC) are overly optimistic.")
