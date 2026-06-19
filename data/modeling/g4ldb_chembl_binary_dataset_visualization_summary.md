# G4LDB/ChEMBL Binary Dataset Visualization Summary

Generated: 2026-06-09T09:57:46+00:00

Source dataset: `data/modeling/g4ldb_chembl_binary_dataset.csv`

Dashboard: `data/modeling/g4ldb_chembl_binary_dataset_overview.png`

## Validation

- Total rows: 9,344
- Label counts: 1=4,672, 0=4,672
- Unique canonical SMILES: 9,344
- Duplicate canonical SMILES in combined dataset: 0
- Duplicate canonical SMILES among negatives: 0
- Positive-negative canonical SMILES overlap: 0
- Missing core field values: 0
- Max selected negative similarity to any G4LDB positive: 0.699
- Similarity filter kept/removed candidates: 37,752/224
- Max removed candidate similarity: 1

## Descriptor Means

| Descriptor | G4LDB positive mean | ChEMBL assumed-negative mean | Negative - positive |
|---|---:|---:|---:|
| rdkit_mw | 581.192 | 521.562 | -59.63 |
| rdkit_alogp | 4.501 | 4.242 | -0.259 |
| rdkit_hbd | 2.2 | 2.131 | -0.069 |
| rdkit_hba | 6.382 | 6.113 | -0.268 |
| rdkit_tpsa | 100.789 | 106.353 | 5.564 |

## Descriptor Medians

| Descriptor | G4LDB positive median | ChEMBL assumed-negative median |
|---|---:|---:|
| rdkit_mw | 520.586 | 500.629 |
| rdkit_alogp | 4.269 | 4.209 |
| rdkit_hbd | 2 | 2 |
| rdkit_hba | 6 | 6 |
| rdkit_tpsa | 81.24 | 95.4 |

## Notes

- ChEMBL label 0 rows are assumed negatives after G4LDB similarity filtering, not experimentally confirmed non-binders.
- Histogram panels clip to the 1st-99th percentile only for display readability; tabulated values use the full dataset.
