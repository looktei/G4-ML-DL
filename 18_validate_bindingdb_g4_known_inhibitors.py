#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Run BindingDB known-G4 positive-control validation."""

STEP_ID = '18'
STEP_TITLE = 'Run BindingDB known-G4 positive-control validation'
EMBEDDED_SOURCE_NAME = '19_validate_bindingdb_g4_known_inhibitors.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Known G4 binder/inhibitor positive-control workflow using BindingDB records."""\n\nfrom __future__ import annotations\n\nimport argparse\nimport importlib.util\nimport json\nimport re\nimport sys\nfrom datetime import datetime\nfrom pathlib import Path\nfrom typing import Any\n\nimport matplotlib\n\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\nfrom rdkit import Chem, RDLogger\n\n\nSEEDS: list[dict[str, Any]] = [\n    {\n        "seed_key": "pyridostatin",\n        "preferred_name": "Pyridostatin",\n        "aliases": ["pyridostatin", "PDS"],\n        "g4_relevance_label": "known_g4_binder_stabilizer",\n        "curation_note": "Classic G4 ligand; present in local G4LDB training source, but not recovered from BindingDB release by seed-name scan.",\n    },\n    {\n        "seed_key": "tmpyp4",\n        "preferred_name": "TMPyP4",\n        "aliases": [\n            "TMPyP4",\n            "TMPYP",\n            "5,10,15,20-tetra(1-methyl-4-pyridyl)-porphyrin",\n            "tetrakis(1-methylpyridinium",\n        ],\n        "g4_relevance_label": "known_g4_binder_stabilizer",\n        "curation_note": "Classic cationic porphyrin G4 ligand; BindingDB rows are protein/helicase inhibition assays, not direct G4 binding constants.",\n    },\n    {\n        "seed_key": "braco19",\n        "preferred_name": "BRACO-19",\n        "aliases": ["BRACO-19", "BRACO19", "BRACO"],\n        "g4_relevance_label": "known_g4_binder_stabilizer",\n        "curation_note": "Classic G4 ligand; not recovered from BindingDB release by seed-name scan.",\n    },\n    {\n        "seed_key": "telomestatin",\n        "preferred_name": "Telomestatin",\n        "aliases": ["telomestatin", "TMS"],\n        "g4_relevance_label": "known_g4_binder_stabilizer",\n        "curation_note": "Macrocyclic G4 stabilizer; BindingDB rows are protein/helicase inhibition assays, not direct G4 binding constants.",\n    },\n    {\n        "seed_key": "rhps4",\n        "preferred_name": "RHPS4",\n        "aliases": ["RHPS4", "RHPS-4"],\n        "g4_relevance_label": "known_g4_binder_stabilizer",\n        "curation_note": "Classic G4 ligand; present in local G4LDB training source, but not recovered from BindingDB release by seed-name scan.",\n    },\n    {\n        "seed_key": "cx5461",\n        "preferred_name": "CX-5461",\n        "aliases": ["CX-5461", "CX5461"],\n        "g4_relevance_label": "reported_g4_stabilizer_inhibitor",\n        "curation_note": "Reported G4 stabilizer/RNA polymerase I inhibitor; not recovered from BindingDB release by seed-name scan.",\n    },\n    {\n        "seed_key": "quarfloxin",\n        "preferred_name": "Quarfloxin",\n        "aliases": ["quarfloxin", "CX-3543", "CX3543"],\n        "g4_relevance_label": "reported_g4_stabilizer_inhibitor",\n        "curation_note": "Reported quadruplex-targeting fluoroquinolone derivative; not recovered from BindingDB release by seed-name scan.",\n    },\n    {\n        "seed_key": "phendc3",\n        "preferred_name": "Phen-DC3",\n        "aliases": ["Phen-DC3", "PhenDC3", "phendc3", "phendc-3"],\n        "g4_relevance_label": "known_g4_binder_stabilizer",\n        "curation_note": "Classic high-affinity G4 ligand; BindingDB rows are protein/helicase inhibition assays, not direct G4 binding constants.",\n    },\n]\n\nUSECOLS = [\n    "BindingDB Reactant_set_id",\n    "Ligand SMILES",\n    "Ligand InChI",\n    "Ligand InChI Key",\n    "BindingDB MonomerID",\n    "BindingDB Ligand Name",\n    "Target Name",\n    "Target Source Organism According to Curator or DataSource",\n    "Ki (nM)",\n    "IC50 (nM)",\n    "Kd (nM)",\n    "EC50 (nM)",\n    "Curation/DataSource",\n    "Article DOI",\n    "BindingDB Entry DOI",\n    "PMID",\n    "PubChem CID",\n    "ChEMBL ID of Ligand",\n    "DrugBank ID of Ligand",\n    "UniProt (SwissProt) Recommended Name of Target Chain 1",\n    "UniProt (SwissProt) Entry Name of Target Chain 1",\n    "UniProt (SwissProt) Primary ID of Target Chain 1",\n]\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument(\n        "--bindingdb-release",\n        default="data/bindingdb/BindingDB_All_202606_tsv.zip",\n        help="Official BindingDB full TSV release or extracted TSV.",\n    )\n    parser.add_argument(\n        "--training-dataset",\n        default="data/modeling/g4ldb_chembl_binary_dataset.csv",\n        help="Locked training dataset used only for similarity/applicability-domain calculations.",\n    )\n    parser.add_argument(\n        "--checkpoint",\n        default="data/modeling/deep_learning_gnn/gnn_molecular_graph_scaffold_seed42.pt",\n        help="Locked molecular graph GNN checkpoint.",\n    )\n    parser.add_argument(\n        "--output-dir",\n        default="data/bindingdb_g4_known_inhibitors_validation",\n        help="Separate output directory for this positive-control workflow.",\n    )\n    parser.add_argument("--chunksize", type=int, default=100000)\n    parser.add_argument("--batch-size", type=int, default=256)\n    parser.add_argument("--cpu", action="store_true")\n    parser.add_argument("--skip-predict", action="store_true")\n    return parser.parse_args()\n\n\ndef root_from_script() -> Path:\n    return Path(__file__).resolve().parents[1]\n\n\ndef resolve(path: str | Path, root: Path) -> Path:\n    path = Path(path)\n    return path if path.is_absolute() else root / path\n\n\ndef load_kd_validator(root: Path):\n    validator_path = root / "scripts" / "validate_bindingdb_kd_gnn_external.py"\n    spec = importlib.util.spec_from_file_location("bindingdb_kd_validator", validator_path)\n    if spec is None or spec.loader is None:\n        raise SystemExit(f"Cannot load existing GNN validator helpers: {validator_path}")\n    module = importlib.util.module_from_spec(spec)\n    sys.modules[spec.name] = module\n    spec.loader.exec_module(module)\n    return module\n\n\ndef seed_table() -> pd.DataFrame:\n    rows = []\n    for seed in SEEDS:\n        row = dict(seed)\n        row["aliases"] = " | ".join(seed["aliases"])\n        rows.append(row)\n    return pd.DataFrame(rows)\n\n\ndef alias_patterns() -> dict[str, re.Pattern[str]]:\n    out = {}\n    for seed in SEEDS:\n        aliases = sorted(seed["aliases"], key=len, reverse=True)\n        out[seed["seed_key"]] = re.compile("|".join(re.escape(alias) for alias in aliases), flags=re.IGNORECASE)\n    return out\n\n\ndef read_bindingdb_chunks(path: Path, chunksize: int):\n    compression = "zip" if path.suffix.lower() == ".zip" else "infer"\n    return pd.read_csv(\n        path,\n        sep="\\t",\n        usecols=lambda column: column in USECOLS,\n        chunksize=chunksize,\n        compression=compression,\n        low_memory=False,\n    )\n\n\ndef scan_bindingdb_by_seed_name(path: Path, chunksize: int) -> tuple[pd.DataFrame, dict[str, Any]]:\n    patterns = alias_patterns()\n    rows = []\n    scanned = 0\n    match_counts = {key: 0 for key in patterns}\n    for chunk in read_bindingdb_chunks(path, chunksize):\n        scanned += len(chunk)\n        ligand_text = chunk["BindingDB Ligand Name"].fillna("").astype(str)\n        for seed_key, pattern in patterns.items():\n            mask = ligand_text.str.contains(pattern, na=False)\n            match_counts[seed_key] += int(mask.sum())\n            if mask.any():\n                sub = chunk.loc[mask].copy()\n                sub["seed_key"] = seed_key\n                sub["bindingdb_match_basis"] = "ligand_name_alias_scan"\n                rows.append(sub)\n    if rows:\n        raw = pd.concat(rows, ignore_index=True).drop_duplicates()\n    else:\n        raw = pd.DataFrame(columns=USECOLS + ["seed_key", "bindingdb_match_basis"])\n    counts = {"bindingdb_rows_scanned": int(scanned), "raw_seed_name_matches": int(len(raw)), **{f"raw_matches_{k}": int(v) for k, v in match_counts.items()}}\n    return raw, counts\n\n\ndef standardize_records(df: pd.DataFrame, helper) -> tuple[pd.DataFrame, dict[str, Any]]:\n    if df.empty:\n        return df.copy(), {"valid_standardized_bindingdb_records": 0}\n    RDLogger.DisableLog("rdApp.*")\n    out = df.copy()\n    standardized = out["Ligand SMILES"].apply(helper.standardize_mol)\n    out["standardized_mol_ok"] = [item[0] is not None for item in standardized]\n    out["canonical_smiles"] = [item[1] for item in standardized]\n    out["standardized_inchikey"] = [item[2] for item in standardized]\n    out["inchikey"] = out["standardized_inchikey"].fillna(out["Ligand InChI Key"])\n    seed_meta = seed_table()[["seed_key", "preferred_name", "g4_relevance_label", "curation_note"]]\n    out = out.merge(seed_meta, on="seed_key", how="left")\n    out["bindingdb_g4_evidence_label"] = np.where(\n        out["Target Name"].fillna("").str.contains("Fanconi anemia group J|DinG|helicase", case=False, regex=True),\n        "g4_relevant_helicase_inhibition_record",\n        "bindingdb_bioactivity_record_not_g4_specific",\n    )\n    out["known_g4_positive_label"] = 1\n    endpoint_cols = ["Ki (nM)", "IC50 (nM)", "Kd (nM)"]\n    out["available_endpoint_types"] = out[endpoint_cols].notna().apply(\n        lambda row: "|".join([column.split()[0] for column, present in row.items() if present]),\n        axis=1,\n    )\n    out = out[out["standardized_mol_ok"]].copy()\n    counts = {\n        "valid_standardized_bindingdb_records": int(len(out)),\n        "bindingdb_supported_seed_count": int(out["seed_key"].nunique()) if len(out) else 0,\n        "bindingdb_supported_compound_count": int(out["inchikey"].nunique()) if len(out) else 0,\n    }\n    return out.reset_index(drop=True), counts\n\n\ndef compound_summary(records: pd.DataFrame) -> pd.DataFrame:\n    if records.empty:\n        return records.copy()\n\n    def join_unique(values: pd.Series, limit: int = 20) -> str:\n        seen = []\n        for value in values.dropna():\n            text = str(value).strip()\n            if text and text not in seen:\n                seen.append(text)\n            if len(seen) >= limit:\n                break\n        return " | ".join(seen)\n\n    def endpoint_min(series: pd.Series) -> float:\n        vals = pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.extract(r"([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)")[0], errors="coerce")\n        return float(vals.min()) if vals.notna().any() else np.nan\n\n    grouped = records.groupby(["seed_key", "inchikey"], dropna=False)\n    summary = grouped.agg(\n        preferred_name=("preferred_name", "first"),\n        canonical_smiles=("canonical_smiles", "first"),\n        bindingdb_ligand_names=("BindingDB Ligand Name", join_unique),\n        g4_relevance_label=("g4_relevance_label", "first"),\n        bindingdb_g4_evidence_label=("bindingdb_g4_evidence_label", join_unique),\n        target_names=("Target Name", join_unique),\n        available_endpoint_types=("available_endpoint_types", join_unique),\n        best_ki_nm=("Ki (nM)", endpoint_min),\n        best_ic50_nm=("IC50 (nM)", endpoint_min),\n        best_kd_nm=("Kd (nM)", endpoint_min),\n        bindingdb_record_count=("BindingDB Reactant_set_id", "size"),\n        bindingdb_reactant_set_ids=("BindingDB Reactant_set_id", join_unique),\n        article_dois=("Article DOI", join_unique),\n        pmids=("PMID", join_unique),\n        pubchem_cids=("PubChem CID", join_unique),\n        chembl_ids=("ChEMBL ID of Ligand", join_unique),\n        curation_note=("curation_note", "first"),\n    ).reset_index()\n    summary["known_g4_positive_label"] = 1\n    return summary\n\n\ndef add_similarity(compounds: pd.DataFrame, training_path: Path, helper) -> pd.DataFrame:\n    training = helper.load_training_mols(training_path)\n    external = compounds.rename(columns={"known_g4_positive_label": "label"}).copy()\n    return helper.add_training_similarity(external, training, radius=2, n_bits=2048).rename(columns={"label": "known_g4_positive_label"})\n\n\ndef predict(compounds: pd.DataFrame, checkpoint: Path, batch_size: int, cpu: bool, helper) -> tuple[pd.DataFrame, str]:\n    external = compounds.rename(columns={"known_g4_positive_label": "label"}).copy()\n    predicted, status = helper.predict_with_locked_gnn(external, checkpoint, batch_size, cpu)\n    predicted = predicted.rename(columns={"label": "known_g4_positive_label"})\n    return predicted, status.message\n\n\ndef write_figures(compounds: pd.DataFrame, records: pd.DataFrame, output: Path) -> list[str]:\n    sns.set_theme(style="whitegrid", context="talk")\n    files: list[str] = []\n    if not records.empty:\n        endpoint_counts = (\n            records.melt(\n                id_vars=["seed_key"],\n                value_vars=["Ki (nM)", "IC50 (nM)", "Kd (nM)"],\n                var_name="endpoint_type",\n                value_name="endpoint_value",\n            )\n            .dropna(subset=["endpoint_value"])\n            .groupby(["seed_key", "endpoint_type"])\n            .size()\n            .reset_index(name="records")\n        )\n        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=220)\n        sns.barplot(data=endpoint_counts, x="seed_key", y="records", hue="endpoint_type", ax=ax)\n        ax.set_xlabel("Seed compound")\n        ax.set_ylabel("BindingDB records")\n        ax.set_title("BindingDB endpoint records recovered by seed")\n        plt.xticks(rotation=20, ha="right")\n        plt.tight_layout()\n        fig.savefig(output / "bindingdb_g4_known_endpoint_counts.png")\n        plt.close(fig)\n        files.append("bindingdb_g4_known_endpoint_counts.png")\n\n    if "nearest_train_similarity" in compounds.columns:\n        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=220)\n        sns.barplot(data=compounds.sort_values("nearest_train_similarity"), x="preferred_name", y="nearest_train_similarity", ax=ax, color="#4C78A8")\n        ax.axhline(0.8, color="black", linestyle="--", linewidth=1)\n        ax.axhline(0.9, color="gray", linestyle=":", linewidth=1)\n        ax.set_xlabel("Compound")\n        ax.set_ylabel("Nearest training Morgan Tanimoto")\n        ax.set_title("Similarity to locked G4 training chemistry")\n        plt.xticks(rotation=20, ha="right")\n        plt.tight_layout()\n        fig.savefig(output / "bindingdb_g4_known_training_similarity.png")\n        plt.close(fig)\n        files.append("bindingdb_g4_known_training_similarity.png")\n\n    if "gnn_probability" in compounds.columns:\n        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=220)\n        plot_df = compounds.sort_values("gnn_probability", ascending=False)\n        sns.barplot(data=plot_df, x="preferred_name", y="gnn_probability", hue="applicability_domain_bin", dodge=False, ax=ax)\n        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)\n        ax.set_xlabel("Compound")\n        ax.set_ylabel("Locked GNN G4-positive probability")\n        ax.set_title("Known G4 binders with BindingDB records")\n        plt.xticks(rotation=20, ha="right")\n        plt.tight_layout()\n        fig.savefig(output / "bindingdb_g4_known_gnn_scores.png")\n        plt.close(fig)\n        files.append("bindingdb_g4_known_gnn_scores.png")\n    return files\n\n\ndef score_summary(compounds: pd.DataFrame) -> pd.DataFrame:\n    row: dict[str, Any] = {\n        "subset": "bindingdb_supported_known_g4_positive_compounds",\n        "n": int(len(compounds)),\n        "positives": int(compounds["known_g4_positive_label"].sum()) if "known_g4_positive_label" in compounds.columns else int(len(compounds)),\n        "negatives": 0,\n        "classification_metrics_applicable": False,\n        "reason": "Only curated positive-control compounds were recovered; ROC-AUC/PR-AUC/MCC require negatives.",\n    }\n    if "gnn_probability" in compounds.columns and len(compounds):\n        row.update(\n            {\n                "gnn_score_median": float(compounds["gnn_probability"].median()),\n                "gnn_score_min": float(compounds["gnn_probability"].min()),\n                "gnn_score_max": float(compounds["gnn_probability"].max()),\n                "recovered_at_threshold_0_5": int((compounds["gnn_probability"] >= 0.5).sum()),\n            }\n        )\n    if "nearest_train_similarity" in compounds.columns and len(compounds):\n        row.update(\n            {\n                "median_nearest_train_similarity": float(compounds["nearest_train_similarity"].median()),\n                "near_or_exact_training_analog_ge_0_8": int((compounds["nearest_train_similarity"] >= 0.8).sum()),\n                "near_exact_training_analog_ge_0_9": int((compounds["nearest_train_similarity"] >= 0.9).sum()),\n            }\n        )\n    return pd.DataFrame([row])\n\n\ndef write_report(output: Path, counts: dict[str, Any], prediction_status: str, generated_files: list[str], compounds: pd.DataFrame) -> None:\n    supported = compounds["preferred_name"].tolist() if not compounds.empty else []\n    recovered = int((compounds["gnn_probability"] >= 0.5).sum()) if "gnn_probability" in compounds.columns else None\n    lines = [\n        "# BindingDB Known G4 Binder/Inhibitor Positive-Control Validation",\n        "",\n        f"Generated: {datetime.now().isoformat(timespec=\'seconds\')}",\n        "",\n        "## Scope",\n        "",\n        "This workflow tests whether the locked Molecular Graph GNN assigns high G4-positive scores to classic G4 binders/inhibitors that also have traceable BindingDB ligand-target records.",\n        "",\n        "Important limitation: BindingDB records recovered here are protein/helicase inhibition bioactivity records, not direct G-quadruplex binding constants. Therefore these results should be framed as positive-control bioactivity support, not independent external validation.",\n        "",\n        "## Seed List",\n        "",\n        "- Seeds curated: Pyridostatin/PDS, TMPyP4, BRACO-19, Telomestatin, RHPS4, CX-5461, Quarfloxin/CX-3543, Phen-DC3.",\n        f"- Seeds with BindingDB release evidence by alias/name scan: {\', \'.join(supported) if supported else \'none\'}.",\n        "",\n        "## Counts",\n        "",\n    ]\n    for key, value in counts.items():\n        lines.append(f"- {key}: {value}")\n    lines.extend(["", "## Locked GNN Result", ""])\n    if prediction_status:\n        lines.append(f"- Prediction status: {prediction_status}")\n    if recovered is not None:\n        lines.append(f"- Known G4-positive compounds scored >= 0.5: {recovered}/{len(compounds)}")\n        lines.append(f"- Median GNN probability: {compounds[\'gnn_probability\'].median():.4f}")\n    if "nearest_train_similarity" in compounds.columns and len(compounds):\n        near = int((compounds["nearest_train_similarity"] >= 0.8).sum())\n        lines.append(f"- Compounds with nearest training Tanimoto >= 0.8: {near}/{len(compounds)}")\n    lines.extend(\n        [\n            "",\n            "## Interpretation",\n            "",\n            "Use this as a supplementary positive-control check. Rows with high similarity to the training set are useful annotation support but should not be described as leakage-filtered external generalization.",\n            "",\n            "## Generated Files",\n            "",\n        ]\n    )\n    for file in generated_files + ["bindingdb_g4_known_inhibitors_validation_report.md"]:\n        lines.append(f"- `{file}`")\n    (output / "bindingdb_g4_known_inhibitors_validation_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf-8")\n\n\ndef main() -> int:\n    args = parse_args()\n    root = root_from_script()\n    output = resolve(args.output_dir, root)\n    output.mkdir(parents=True, exist_ok=True)\n    helper = load_kd_validator(root)\n    generated: list[str] = []\n    counts: dict[str, Any] = {}\n    prediction_status = ""\n\n    seeds = seed_table()\n    seeds.to_csv(output / "g4_known_inhibitor_seed_list.csv", index=False)\n    generated.append("g4_known_inhibitor_seed_list.csv")\n\n    bindingdb_path = resolve(args.bindingdb_release, root)\n    if not bindingdb_path.exists():\n        counts["blocker"] = f"BindingDB release missing: {bindingdb_path}"\n        write_report(output, counts, prediction_status, generated, pd.DataFrame())\n        return 2\n\n    raw, scan_counts = scan_bindingdb_by_seed_name(bindingdb_path, args.chunksize)\n    counts.update(scan_counts)\n    raw.to_csv(output / "bindingdb_g4_known_raw_seed_matches.csv", index=False)\n    generated.append("bindingdb_g4_known_raw_seed_matches.csv")\n\n    records, standard_counts = standardize_records(raw, helper)\n    counts.update(standard_counts)\n    records.to_csv(output / "bindingdb_g4_known_endpoint_records.csv", index=False)\n    generated.append("bindingdb_g4_known_endpoint_records.csv")\n\n    compounds = compound_summary(records)\n    compounds.to_csv(output / "bindingdb_g4_known_compound_summary.csv", index=False)\n    generated.append("bindingdb_g4_known_compound_summary.csv")\n\n    training_path = resolve(args.training_dataset, root)\n    checkpoint_path = resolve(args.checkpoint, root)\n    if compounds.empty:\n        counts["blocker"] = "No valid BindingDB-supported compounds remained after standardization."\n    elif not training_path.exists():\n        counts["blocker"] = f"Training dataset missing: {training_path}"\n    else:\n        compounds = add_similarity(compounds, training_path, helper)\n        compounds.to_csv(output / "bindingdb_g4_known_with_training_similarity.csv", index=False)\n        generated.append("bindingdb_g4_known_with_training_similarity.csv")\n        if args.skip_predict:\n            prediction_status = "Skipped by --skip-predict."\n        elif not checkpoint_path.exists():\n            prediction_status = f"Blocked: checkpoint missing: {checkpoint_path}"\n        else:\n            compounds, prediction_status = predict(compounds, checkpoint_path, args.batch_size, args.cpu, helper)\n            compounds.to_csv(output / "bindingdb_g4_known_locked_gnn_predictions.csv", index=False)\n            generated.append("bindingdb_g4_known_locked_gnn_predictions.csv")\n            score_summary(compounds).to_csv(output / "bindingdb_g4_known_score_summary.csv", index=False)\n            generated.append("bindingdb_g4_known_score_summary.csv")\n        figure_files = write_figures(compounds, records, output)\n        generated.extend(figure_files)\n\n    counts_path = output / "bindingdb_g4_known_counts.json"\n    counts_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")\n    generated.append("bindingdb_g4_known_counts.json")\n    write_report(output, counts, prediction_status, generated, compounds)\n    return 0 if "blocker" not in counts and not prediction_status.startswith("Blocked") else 2\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'


import argparse
import contextlib
import io
import json
import os
import shlex
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import TextIO


class Tee(io.TextIOBase):
    """Write text both to the live terminal stream and to a log file."""

    def __init__(self, live: TextIO, log: TextIO):
        self.live = live
        self.log = log

    def write(self, s: str) -> int:
        self.live.write(s)
        self.live.flush()
        self.log.write(s)
        self.log.flush()
        return len(s)

    def flush(self) -> None:
        self.live.flush()
        self.log.flush()


def parse_seed_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        for part in str(value).replace(',', ' ').split():
            part = part.strip()
            if not part:
                continue
            if '-' in part and part.replace('-', '').isdigit():
                a, b = part.split('-', 1)
                start, end = int(a), int(b)
                if end < start:
                    raise SystemExit(f"Invalid seed range: {part}")
                out.extend(str(i) for i in range(start, end + 1))
            else:
                int(part)
                out.append(part)
    return out


def build_forwarded_args(args: argparse.Namespace, unknown_args: list[str], seed_mode: str) -> tuple[list[str], list[str]]:
    forwarded: list[str] = []
    notes: list[str] = []

    if seed_mode == 'single':
        if args.seed is not None:
            forwarded += ['--seed', str(args.seed)]
            notes.append(f'seed={args.seed}')
    elif seed_mode == 'repeated':
        primary = args.primary_seed if args.primary_seed is not None else args.seed
        if primary is not None:
            forwarded += ['--primary-seed', str(primary)]
            notes.append(f'primary_seed={primary}')
        seeds = parse_seed_list(args.seeds)
        if args.n_seeds is not None:
            if args.n_seeds < 1:
                raise SystemExit('--n-seeds must be >= 1')
            seeds = [str(i) for i in range(args.n_seeds)]
        if seeds:
            forwarded += ['--seeds', *seeds]
            notes.append(f'repeated_seeds={" ".join(seeds)}')
    elif seed_mode in {'none', 'env'}:
        if args.seed is not None or args.primary_seed is not None or args.seeds is not None or args.n_seeds is not None:
            notes.append('Seed options were supplied but this embedded step has no seed argument; seed options were logged but not passed.')
    else:
        notes.append(f'Unknown seed_mode={seed_mode!r}; seed options were not passed automatically.')

    if args.extra_args.strip():
        extra = shlex.split(args.extra_args)
        forwarded += extra
        notes.append(f'extra_args={args.extra_args}')

    if unknown_args:
        forwarded += unknown_args
        notes.append(f'unknown_args_forwarded={" ".join(shlex.quote(x) for x in unknown_args)}')

    return forwarded, notes


def start_progress_thread(step_id: str, step_title: str, interval: float, stop_event: threading.Event) -> threading.Thread | None:
    if interval <= 0:
        return None

    def loop() -> None:
        started = time.monotonic()
        while not stop_event.wait(interval):
            elapsed = int(time.monotonic() - started)
            print(f"[RUNNING] {datetime.now().isoformat(timespec='seconds')} | {step_id} {step_title} | elapsed={elapsed}s", flush=True)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def run_embedded_step(
    *,
    step_id: str,
    step_title: str,
    embedded_source_name: str,
    seed_mode: str,
    embedded_source: str,
) -> int:
    parser = argparse.ArgumentParser(description=f"Standalone v.4 reproducibility step: {step_title}")
    parser.add_argument('--project-root', default=None, help='Output/project root. Default: this script folder. No original-script folder is required.')
    parser.add_argument('--log-dir', default=None, help='Directory for step logs. Default: <workflow folder>/logs.')
    parser.add_argument('--seed', type=int, default=None, help='Single seed for steps that support --seed. Also used as --primary-seed for repeated-seed steps unless --primary-seed is set.')
    parser.add_argument('--primary-seed', type=int, default=None, help='Primary seed for repeated-seed scripts that support --primary-seed.')
    parser.add_argument('--seeds', nargs='*', default=None, help='Repeated split seeds for scripts that support --seeds. Accepts space/comma lists or ranges, e.g. --seeds 0-19 or --seeds 0 1 2.')
    parser.add_argument('--n-seeds', type=int, default=None, help='Shortcut for --seeds 0-(N-1) on repeated-seed scripts.')
    parser.add_argument('--extra-args', default='', help='Additional arguments passed verbatim to the embedded source, e.g. "--cpu --batch-size 64".')
    parser.add_argument('--progress-interval', type=float, default=30.0, help='Seconds between [RUNNING] heartbeat messages during long steps. Use 0 to disable.')
    parser.add_argument('--dry-run', action='store_true', help='Print/log what would run without executing the embedded source.')
    args, unknown_args = parser.parse_known_args()

    wrapper_path = Path(__file__).resolve()
    workflow_dir = wrapper_path.parent
    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else workflow_dir
    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else workflow_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    forwarded_args, notes = build_forwarded_args(args, unknown_args, seed_mode)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    base = f'{step_id}_{Path(embedded_source_name).stem}_{timestamp}'
    stdout_path = log_dir / f'{base}.stdout.log'
    stderr_path = log_dir / f'{base}.stderr.log'
    status_json = log_dir / f'{base}.status.json'
    status_md = log_dir / f'{base}.status.md'
    virtual_source_path = project_root / '_embedded_source_virtual' / embedded_source_name

    status = {
        'version': 'revised_reproduce_workflow_v.4',
        'step_id': step_id,
        'step_title': step_title,
        'embedded_source': embedded_source_name,
        'workflow_dir': str(workflow_dir),
        'project_root': str(project_root),
        'argv_for_embedded_source': [str(virtual_source_path), *forwarded_args],
        'started': datetime.now().isoformat(timespec='seconds'),
        'seed_mode': seed_mode,
        'notes': notes,
        'dry_run': bool(args.dry_run),
        'stdout_log': str(stdout_path),
        'stderr_log': str(stderr_path),
        'status_json': str(status_json),
        'status_md': str(status_md),
    }

    returncode: int | None = None
    result = 'UNKNOWN'
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    stop_event = threading.Event()
    progress_thread: threading.Thread | None = None

    with stdout_path.open('w', encoding='utf-8') as stdout_log, stderr_path.open('w', encoding='utf-8') as stderr_log:
        with contextlib.redirect_stdout(Tee(sys.__stdout__, stdout_log)), contextlib.redirect_stderr(Tee(sys.__stderr__, stderr_log)):
            print(f"[START] {status['started']} | {step_id} {step_title}", flush=True)
            print(f"Workflow dir: {workflow_dir}", flush=True)
            print(f"Project root: {project_root}", flush=True)
            print(f"Embedded source: {embedded_source_name}", flush=True)
            print('Forwarded args:', ' '.join(shlex.quote(x) for x in forwarded_args) if forwarded_args else '(none)', flush=True)
            print(f"Logs: {log_dir}", flush=True)
            print(f"Stdout log: {stdout_path}", flush=True)
            print(f"Stderr log: {stderr_path}", flush=True)

            if args.dry_run:
                result = 'DRY_RUN'
                returncode = None
                print('[RUNNING] dry-run requested; embedded source not executed.', flush=True)
            else:
                print(f"[RUNNING] executing embedded source now; live output is mirrored to logs every print/flush.", flush=True)
                try:
                    project_root.mkdir(parents=True, exist_ok=True)
                    os.chdir(project_root)
                    sys.argv = [str(virtual_source_path), *forwarded_args]
                    progress_thread = start_progress_thread(step_id, step_title, args.progress_interval, stop_event)
                    namespace = {
                        '__name__': '__main__',
                        '__file__': str(virtual_source_path),
                        '__package__': None,
                        '__builtins__': __builtins__,
                    }
                    exec(compile(embedded_source, str(virtual_source_path), 'exec'), namespace)
                    returncode = 0
                except SystemExit as exc:
                    code = exc.code
                    if code is None:
                        returncode = 0
                    elif isinstance(code, int):
                        returncode = code
                    else:
                        print(code, file=sys.stderr)
                        returncode = 1
                except BaseException:
                    traceback.print_exc()
                    returncode = 1
                finally:
                    stop_event.set()
                    if progress_thread is not None:
                        progress_thread.join(timeout=1)
                    sys.argv = old_argv
                    os.chdir(old_cwd)

                result = 'PASS' if returncode == 0 else 'FAIL'

            finished = datetime.now().isoformat(timespec='seconds')
            status.update({'finished': finished, 'returncode': returncode, 'result': result})
            print(f"[DONE] {finished} | {step_id} {step_title} | result={result} | returncode={returncode}", flush=True)
            print(f"Status: {status_md}", flush=True)

    status_json.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding='utf-8')
    cmd_text = ' '.join(shlex.quote(x) for x in [Path(sys.executable).name, wrapper_path.name, *sys.argv[1:]])
    status_md.write_text(
        f"# {step_id} {step_title}\n\n"
        f"- Version: `revised_reproduce_workflow_v.4`\n"
        f"- Started: {status['started']}\n"
        f"- Finished: {status.get('finished')}\n"
        f"- Result: **{result}**\n"
        f"- Return code: `{returncode}`\n"
        f"- Embedded source: `{embedded_source_name}`\n"
        f"- Workflow dir: `{workflow_dir}`\n"
        f"- Project root: `{project_root}`\n"
        f"- Seed mode: `{seed_mode}`\n"
        f"- Notes: {', '.join(notes) if notes else 'none'}\n"
        f"- Wrapper command observed: `{cmd_text}`\n"
        f"- Forwarded args: `{(' '.join(shlex.quote(x) for x in forwarded_args) if forwarded_args else '(none)')}`\n"
        f"- Stdout log: `{stdout_path}`\n"
        f"- Stderr log: `{stderr_path}`\n"
        f"- Status JSON: `{status_json}`\n",
        encoding='utf-8',
    )

    if args.dry_run:
        return 0
    return int(returncode or 0)



if __name__ == '__main__':
    raise SystemExit(run_embedded_step(
        step_id=STEP_ID,
        step_title=STEP_TITLE,
        embedded_source_name=EMBEDDED_SOURCE_NAME,
        seed_mode=SEED_MODE,
        embedded_source=EMBEDDED_SOURCE,
    ))
