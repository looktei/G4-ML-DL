# G4LDB 3.x Ligand Dataset

Retrieved: 2026-06-09T08:56:11+00:00

Source: G4LDB backend at `https://www.g4ldb.com:8088/`.

## Source endpoints

- Ligand metadata: `GET /ligand/export?n=<offset>&s=200`, paginated from offset 0 until an empty page.
- Ligand activity/details: `GET /ligand/get?ligandId=<id>`, one request per ligand.

## Files

- `g4ldb_ligands_raw.json`: raw ligand metadata list from `/ligand/export`.
- `g4ldb_ligands.csv`: metadata table with ligand identifiers, names, SMILES, PubChem IDs, formulae, molecular descriptors, and year where available.
- `g4ldb_activity_raw.jsonl`: raw activity/detail rows from `/ligand/get`, one JSON object per line.
- `g4ldb_activity.csv`: activity/detail table including ligand, assay/activity, sequence, method, value, publication, DOI, and related fields.
- `g4ldb_g4_ligand_ml_ready.csv`: positive ligand table for downstream ML use. Each row is a G4LDB ligand with `label=1` and aggregated activity, sequence, and DOI examples.

## Row counts

- Ligands: 4926
- Activity/detail records: 49585
- ML-ready positive ligand rows: 4926

## ML-ready fields

`ligandId`, `name`, `smiles`, `pubchemId`, `formula`, `mw`, `hbd`, `hba`, `alogp`, `solubility`, `year`, `n_activity_records`, `activity1_set`, `activity2_set`, `sequence_examples`, `doi_examples`, `label`.

## Intended use

This dataset is prepared for Aim 2 of the Nipah virus G-quadruplex deep learning / ligand prioritization project. It provides known G-quadruplex ligands as positive examples and ligand descriptors for model training, candidate filtering, and prioritization. It does not include negative labels; negative or background molecules should be assembled separately with clear provenance.
