# ChEMBL Assumed-Negative Decoys

These rows are decoys/assumed negatives, not experimentally confirmed non-G4 binders.

- Generated at: 2026-06-09T09:55:26+00:00
- Positive target rows: 4672
- Selected decoy rows: 4672
- Clean ChEMBL candidate pool: 37976
- Removed as too similar to positives: 224
- Similarity threshold: Morgan/Tanimoto >= 0.7
- Positive/decoy overlap after canonicalization: 0

## Matching Residuals

- rdkit_mw: positive mean 581.1923, decoy mean 521.5619, difference -59.6304
- rdkit_alogp: positive mean 4.5008, decoy mean 4.2418, difference -0.259
- rdkit_hbd: positive mean 2.1999, decoy mean 2.1312, difference -0.0687
- rdkit_hba: positive mean 6.3816, decoy mean 6.1134, difference -0.2682
- rdkit_tpsa: positive mean 100.7888, decoy mean 106.3526, difference 5.5638

## Files

- Core decoys: `data/chembl/chembl_negative_decoys.csv`
- Decoys with metadata: `data/chembl/chembl_negative_decoys_with_metadata.csv`
- Combined modeling dataset: `data/modeling/g4ldb_chembl_binary_dataset.csv`
- Similarity-filter audit: `data/chembl/chembl_negative_decoys_removed_similar_to_g4ldb.csv`
- JSON report: `data/chembl/chembl_negative_decoy_report.json`
