# Phase 1B Model Selection Report

Generated: 2026-06-10T00:04:07

## Decision

Retain `Morgan fingerprint + Random Forest` as the controlled Phase 1 classical baseline.

## Evidence From Phase 1 Training

| split    |   roc_auc |   pr_auc |   mcc |   balanced_accuracy |
|:---------|----------:|---------:|------:|--------------------:|
| random   |     0.987 |    0.988 | 0.888 |               0.944 |
| scaffold |     0.973 |    0.975 | 0.797 |               0.895 |

## Phase 1B Validation Evidence

- Primary random split: ROC-AUC 0.990, PR-AUC 0.991, MCC 0.908, Brier 0.056.
- Primary scaffold split: ROC-AUC 0.973, PR-AUC 0.975, MCC 0.793, Brier 0.086.
- 20-seed random split ROC-AUC mean 0.988 +/- 0.002.
- 20-seed scaffold split ROC-AUC mean 0.972 +/- 0.002.

## Interpretation

- Scaffold performance remains high, so the candidate is not only memorizing close random-split analogues.
- Repeated splits show whether the result is sensitive to seed choice; inspect the CSV for any weak seeds before publication.
- Probability outputs are usable for ranking only with the reported calibration caveat. If absolute probability thresholds become important, calibrate the model with a held-out calibration set.
- Applicability-domain labels should accompany any future Nipah G4 screening output, especially for compounds below the borderline threshold.
- Top Morgan bit mappings are representative hashed environments, not unique medicinal-chemistry explanations; collisions are possible with hashed fingerprints.

## Recommended Next Use

Use this model as the Phase 1 screening baseline and report scores with scaffold-split metrics, Brier score, and applicability-domain tags. Do not describe predictions as confirmed Nipah G4 binding activity without experimental validation.
