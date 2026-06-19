#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Build supplementary BindingDB positive-control support analysis."""

STEP_ID = '19'
STEP_TITLE = 'Build supplementary BindingDB positive-control support analysis'
EMBEDDED_SOURCE_NAME = '20_run_bindingdb_supplementary_positive_control.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Supplementary BindingDB positive-control bioactivity support workflow.\n\nThis script combines traceable BindingDB rows that are exact training overlap,\nnear analogs, or known G4 binder/inhibitor records. It is intentionally not an\nindependent external-validation workflow.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport math\nfrom datetime import datetime\nfrom pathlib import Path\nfrom typing import Any\n\nimport matplotlib\n\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\n\n\nDEFAULT_CHECKPOINT = "data/modeling/deep_learning_gnn/gnn_molecular_graph_scaffold_seed42.pt"\nDEFAULT_OUTPUT_DIR = "data/bindingdb_supplementary_positive_control"\nMIN_CORRELATION_N = 3\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument(\n        "--kd-predictions",\n        default="data/bindingdb_kd_external_validation/bindingdb_kd_only_locked_gnn_predictions.csv",\n        help="BindingDB Kd-only table with locked GNN scores and training similarity.",\n    )\n    parser.add_argument(\n        "--kd-similarity",\n        default="data/bindingdb_kd_external_validation/bindingdb_kd_only_with_training_similarity.csv",\n        help="Fallback BindingDB Kd-only table with training similarity if predictions are absent.",\n    )\n    parser.add_argument(\n        "--known-g4-predictions",\n        default="data/bindingdb_g4_known_inhibitors_validation/bindingdb_g4_known_locked_gnn_predictions.csv",\n        help="BindingDB-supported known G4 binder/inhibitor table with locked GNN scores.",\n    )\n    parser.add_argument(\n        "--known-g4-summary",\n        default="data/bindingdb_g4_known_inhibitors_validation/bindingdb_g4_known_with_training_similarity.csv",\n        help="Fallback BindingDB-supported known G4 binder/inhibitor table with training similarity.",\n    )\n    parser.add_argument(\n        "--checkpoint",\n        default=DEFAULT_CHECKPOINT,\n        help="Locked molecular graph GNN checkpoint. Used only as a provenance check.",\n    )\n    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)\n    return parser.parse_args()\n\n\ndef root_from_script() -> Path:\n    return Path(__file__).resolve().parents[1]\n\n\ndef resolve(path: str | Path, root: Path) -> Path:\n    path = Path(path)\n    return path if path.is_absolute() else root / path\n\n\ndef finite_float(value: Any) -> float:\n    try:\n        out = float(value)\n    except (TypeError, ValueError):\n        return np.nan\n    return out if math.isfinite(out) else np.nan\n\n\ndef pactivity_from_nm(value_nm: Any) -> float:\n    value = finite_float(value_nm)\n    if not math.isfinite(value) or value <= 0:\n        return np.nan\n    return 9.0 - math.log10(value)\n\n\ndef text_join(parts: list[Any]) -> str:\n    seen: list[str] = []\n    for part in parts:\n        if pd.isna(part):\n            continue\n        text = str(part).strip()\n        if text and text.lower() != "nan" and text not in seen:\n            seen.append(text)\n    return " | ".join(seen)\n\n\ndef gnn_class(value: Any) -> str:\n    value_float = finite_float(value)\n    if not math.isfinite(value_float):\n        return "not_available"\n    return "G4_positive_at_0.5" if value_float >= 0.5 else "not_G4_positive_at_0.5"\n\n\ndef read_existing(primary: Path, fallback: Path | None = None) -> tuple[pd.DataFrame, Path]:\n    if primary.exists():\n        return pd.read_csv(primary), primary\n    if fallback is not None and fallback.exists():\n        return pd.read_csv(fallback), fallback\n    raise FileNotFoundError(str(primary))\n\n\ndef kd_interpretation(row: pd.Series, cutoff: float) -> str:\n    exact = bool(row.get("exact_training_inchikey_match", False))\n    if exact:\n        return f"BindingDB Kd record for exact training-overlap compound; included as supplementary positive-control support at similarity >= {cutoff:.1f}."\n    if finite_float(row.get("nearest_train_similarity")) >= 0.9:\n        return f"BindingDB Kd record for near-exact training analog; included as supplementary positive-control support at similarity >= {cutoff:.1f}."\n    return f"BindingDB Kd record for near training analog; included as supplementary positive-control support at similarity >= {cutoff:.1f}."\n\n\ndef build_kd_rows(df: pd.DataFrame) -> pd.DataFrame:\n    rows: list[dict[str, Any]] = []\n    for cutoff in (0.8, 0.9):\n        subset = df[pd.to_numeric(df.get("nearest_train_similarity"), errors="coerce") >= cutoff].copy()\n        component = f"bindingdb_kd_training_similarity_ge_{str(cutoff).replace(\'.\', \'_\')}"\n        for _, row in subset.iterrows():\n            pactivity = row.get("pKd_median", np.nan)\n            if not math.isfinite(finite_float(pactivity)):\n                pactivity = pactivity_from_nm(row.get("kd_nM_median"))\n            rows.append(\n                {\n                    "source_component": component,\n                    "compound_name": row.get("bindingdb_ligand_name", ""),\n                    "canonical_smiles": row.get("canonical_smiles", ""),\n                    "inchikey": row.get("inchikey", ""),\n                    "endpoint_type": "Kd",\n                    "activity_value_nM": row.get("kd_nM_median", np.nan),\n                    "pActivity": pactivity,\n                    "bindingdb_target_text": row.get("target_name", ""),\n                    "bindingdb_assay_text": text_join([row.get("institution", ""), row.get("article_doi", ""), row.get("pmid", "")]),\n                    "target_assay_text": text_join([row.get("target_name", ""), row.get("institution", ""), row.get("article_doi", ""), row.get("pmid", "")]),\n                    "bindingdb_record_count": row.get("kd_record_count", np.nan),\n                    "nearest_training_similarity": row.get("nearest_train_similarity", np.nan),\n                    "nearest_training_smiles": row.get("nearest_train_smiles", ""),\n                    "exact_training_inchikey_match": row.get("exact_training_inchikey_match", False),\n                    "gnn_probability": row.get("gnn_probability", np.nan),\n                    "predicted_class": gnn_class(row.get("gnn_probability")),\n                    "original_predicted_label": row.get("gnn_predicted_label", np.nan),\n                    "activity_label_from_source": row.get("endpoint_zone", ""),\n                    "interpretation_label": kd_interpretation(row, cutoff),\n                    "source_row_key": f"kd::{row.get(\'inchikey\', \'\')}::ge_{cutoff:.1f}",\n                }\n            )\n    return pd.DataFrame(rows)\n\n\ndef known_g4_interpretation(row: pd.Series) -> str:\n    exact = bool(row.get("exact_training_inchikey_match", False))\n    similarity = finite_float(row.get("nearest_train_similarity"))\n    if exact:\n        overlap = "exact training-overlap known G4 binder/inhibitor"\n    elif math.isfinite(similarity) and similarity >= 0.9:\n        overlap = "near-exact known G4 binder/inhibitor analog"\n    elif math.isfinite(similarity) and similarity >= 0.8:\n        overlap = "near-analog known G4 binder/inhibitor"\n    else:\n        overlap = "known G4 binder/inhibitor"\n    return f"BindingDB-supported {overlap}; use as supplementary positive-control bioactivity support, not independent external validation."\n\n\ndef build_known_g4_rows(df: pd.DataFrame) -> pd.DataFrame:\n    rows: list[dict[str, Any]] = []\n    endpoint_specs = [("Ki", "best_ki_nm"), ("IC50", "best_ic50_nm"), ("Kd", "best_kd_nm")]\n    for _, row in df.iterrows():\n        emitted = False\n        for endpoint_type, column in endpoint_specs:\n            value = finite_float(row.get(column))\n            if not math.isfinite(value):\n                continue\n            emitted = True\n            rows.append(\n                {\n                    "source_component": "bindingdb_supported_known_g4_binder_inhibitor",\n                    "compound_name": row.get("preferred_name", row.get("bindingdb_ligand_names", "")),\n                    "canonical_smiles": row.get("canonical_smiles", ""),\n                    "inchikey": row.get("inchikey", ""),\n                    "endpoint_type": endpoint_type,\n                    "activity_value_nM": value,\n                    "pActivity": pactivity_from_nm(value),\n                    "bindingdb_target_text": row.get("target_names", ""),\n                    "bindingdb_assay_text": text_join([row.get("article_dois", ""), row.get("pmids", ""), row.get("curation_note", "")]),\n                    "target_assay_text": text_join([row.get("target_names", ""), row.get("article_dois", ""), row.get("pmids", ""), row.get("curation_note", "")]),\n                    "bindingdb_record_count": row.get("bindingdb_record_count", np.nan),\n                    "nearest_training_similarity": row.get("nearest_train_similarity", np.nan),\n                    "nearest_training_smiles": row.get("nearest_train_smiles", ""),\n                    "exact_training_inchikey_match": row.get("exact_training_inchikey_match", False),\n                    "gnn_probability": row.get("gnn_probability", np.nan),\n                    "predicted_class": gnn_class(row.get("gnn_probability")),\n                    "original_predicted_label": row.get("gnn_predicted_label", np.nan),\n                    "activity_label_from_source": row.get("g4_relevance_label", ""),\n                    "interpretation_label": known_g4_interpretation(row),\n                    "source_row_key": f"known_g4::{row.get(\'inchikey\', \'\')}::{endpoint_type}",\n                }\n            )\n        if not emitted:\n            rows.append(\n                {\n                    "source_component": "bindingdb_supported_known_g4_binder_inhibitor",\n                    "compound_name": row.get("preferred_name", row.get("bindingdb_ligand_names", "")),\n                    "canonical_smiles": row.get("canonical_smiles", ""),\n                    "inchikey": row.get("inchikey", ""),\n                    "endpoint_type": text_join([row.get("available_endpoint_types", "")]) or "BindingDB_bioactivity",\n                    "activity_value_nM": np.nan,\n                    "pActivity": np.nan,\n                    "bindingdb_target_text": row.get("target_names", ""),\n                    "bindingdb_assay_text": text_join([row.get("article_dois", ""), row.get("pmids", ""), row.get("curation_note", "")]),\n                    "target_assay_text": text_join([row.get("target_names", ""), row.get("article_dois", ""), row.get("pmids", ""), row.get("curation_note", "")]),\n                    "bindingdb_record_count": row.get("bindingdb_record_count", np.nan),\n                    "nearest_training_similarity": row.get("nearest_train_similarity", np.nan),\n                    "nearest_training_smiles": row.get("nearest_train_smiles", ""),\n                    "exact_training_inchikey_match": row.get("exact_training_inchikey_match", False),\n                    "gnn_probability": row.get("gnn_probability", np.nan),\n                    "predicted_class": gnn_class(row.get("gnn_probability")),\n                    "original_predicted_label": row.get("gnn_predicted_label", np.nan),\n                    "activity_label_from_source": row.get("g4_relevance_label", ""),\n                    "interpretation_label": known_g4_interpretation(row),\n                    "source_row_key": f"known_g4::{row.get(\'inchikey\', \'\')}::no_numeric_endpoint",\n                }\n            )\n    return pd.DataFrame(rows)\n\n\ndef summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:\n    summary_rows: list[dict[str, Any]] = []\n    for component, group in df.groupby("source_component", dropna=False):\n        scores = pd.to_numeric(group["gnn_probability"], errors="coerce")\n        sims = pd.to_numeric(group["nearest_training_similarity"], errors="coerce")\n        summary_rows.append(\n            {\n                "source_component": component,\n                "rows": int(len(group)),\n                "unique_compounds": int(group["inchikey"].nunique()),\n                "endpoint_types": text_join(sorted(group["endpoint_type"].dropna().astype(str).unique().tolist())),\n                "finite_pActivity_rows": int(pd.to_numeric(group["pActivity"], errors="coerce").notna().sum()),\n                "gnn_probability_median": float(scores.median()) if scores.notna().any() else np.nan,\n                "gnn_probability_min": float(scores.min()) if scores.notna().any() else np.nan,\n                "gnn_probability_max": float(scores.max()) if scores.notna().any() else np.nan,\n                "predicted_positive_rows_at_0_5": int((scores >= 0.5).sum()),\n                "median_nearest_training_similarity": float(sims.median()) if sims.notna().any() else np.nan,\n                "rows_with_similarity_ge_0_8": int((sims >= 0.8).sum()),\n                "rows_with_similarity_ge_0_9": int((sims >= 0.9).sum()),\n                "exact_training_inchikey_match_rows": int(group["exact_training_inchikey_match"].fillna(False).astype(bool).sum()),\n            }\n        )\n    summary = pd.DataFrame(summary_rows)\n    endpoint_counts = (\n        df.groupby(["source_component", "endpoint_type"], dropna=False)\n        .size()\n        .reset_index(name="rows")\n        .sort_values(["source_component", "endpoint_type"])\n    )\n    overall = {\n        "rows": int(len(df)),\n        "unique_compounds": int(df["inchikey"].nunique()) if len(df) else 0,\n        "unique_source_components": int(df["source_component"].nunique()) if len(df) else 0,\n        "predicted_positive_rows_at_0_5": int((pd.to_numeric(df["gnn_probability"], errors="coerce") >= 0.5).sum()) if len(df) else 0,\n        "near_analog_rows_ge_0_8": int((pd.to_numeric(df["nearest_training_similarity"], errors="coerce") >= 0.8).sum()) if len(df) else 0,\n        "near_exact_rows_ge_0_9": int((pd.to_numeric(df["nearest_training_similarity"], errors="coerce") >= 0.9).sum()) if len(df) else 0,\n    }\n    return summary, endpoint_counts, overall\n\n\ndef correlation_rows(df: pd.DataFrame) -> pd.DataFrame:\n    rows: list[dict[str, Any]] = []\n    all_unique = (\n        df.groupby(["inchikey", "endpoint_type"], dropna=False)\n        .agg(\n            gnn_probability=("gnn_probability", "max"),\n            pActivity=("pActivity", "max"),\n            source_component=("source_component", lambda x: text_join(sorted(set(map(str, x))))),\n        )\n        .reset_index()\n    )\n    scopes: list[tuple[str, pd.DataFrame]] = [("all_components_deduplicated_by_compound_endpoint", all_unique)]\n    scopes.extend((f"component::{name}", group) for name, group in df.groupby("source_component", dropna=False))\n    for scope, scoped in scopes:\n        for endpoint_type, group in scoped.groupby("endpoint_type", dropna=False):\n            clean = group[["gnn_probability", "pActivity"]].apply(pd.to_numeric, errors="coerce").dropna()\n            row: dict[str, Any] = {\n                "scope": scope,\n                "endpoint_type": endpoint_type,\n                "n": int(len(clean)),\n                "spearman_rho": np.nan,\n                "status": "",\n            }\n            if len(clean) < MIN_CORRELATION_N:\n                row["status"] = f"not_computed: requires at least {MIN_CORRELATION_N} finite score/pActivity pairs"\n            elif clean["gnn_probability"].nunique() < 2 or clean["pActivity"].nunique() < 2:\n                row["status"] = "not_computed: score or pActivity has fewer than two unique values"\n            else:\n                row["spearman_rho"] = float(clean["gnn_probability"].corr(clean["pActivity"], method="spearman"))\n                row["status"] = "computed"\n            rows.append(row)\n    return pd.DataFrame(rows).sort_values(["scope", "endpoint_type"]).reset_index(drop=True)\n\n\ndef write_plots(df: pd.DataFrame, output: Path) -> list[str]:\n    sns.set_theme(style="whitegrid", context="talk")\n    generated: list[str] = []\n    plot_df = df.copy()\n    plot_df["gnn_probability"] = pd.to_numeric(plot_df["gnn_probability"], errors="coerce")\n    plot_df["pActivity"] = pd.to_numeric(plot_df["pActivity"], errors="coerce")\n    plot_df["nearest_training_similarity"] = pd.to_numeric(plot_df["nearest_training_similarity"], errors="coerce")\n\n    if plot_df["gnn_probability"].notna().any():\n        fig, ax = plt.subplots(figsize=(10, 6), dpi=220)\n        sns.stripplot(data=plot_df, x="source_component", y="gnn_probability", hue="endpoint_type", dodge=True, size=8, ax=ax)\n        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)\n        ax.set_xlabel("Supplementary source component")\n        ax.set_ylabel("Locked GNN G4-positive probability")\n        ax.set_title("BindingDB supplementary positive-control GNN scores")\n        plt.xticks(rotation=18, ha="right")\n        plt.tight_layout()\n        fig.savefig(output / "supplementary_positive_control_gnn_scores.png")\n        plt.close(fig)\n        generated.append("supplementary_positive_control_gnn_scores.png")\n\n    scatter = plot_df.dropna(subset=["gnn_probability", "pActivity"])\n    if not scatter.empty:\n        fig, ax = plt.subplots(figsize=(9, 6), dpi=220)\n        sns.scatterplot(\n            data=scatter,\n            x="pActivity",\n            y="gnn_probability",\n            hue="source_component",\n            style="endpoint_type",\n            s=95,\n            ax=ax,\n        )\n        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)\n        ax.set_xlabel("pActivity from BindingDB endpoint")\n        ax.set_ylabel("Locked GNN G4-positive probability")\n        ax.set_title("BindingDB pActivity versus locked GNN score")\n        plt.tight_layout()\n        fig.savefig(output / "supplementary_positive_control_pactivity_vs_score.png")\n        plt.close(fig)\n        generated.append("supplementary_positive_control_pactivity_vs_score.png")\n\n    if plot_df["nearest_training_similarity"].notna().any():\n        fig, ax = plt.subplots(figsize=(9, 6), dpi=220)\n        sns.scatterplot(\n            data=plot_df,\n            x="nearest_training_similarity",\n            y="gnn_probability",\n            hue="source_component",\n            style="predicted_class",\n            s=95,\n            ax=ax,\n        )\n        ax.axvline(0.8, color="black", linestyle="--", linewidth=1)\n        ax.axvline(0.9, color="gray", linestyle=":", linewidth=1)\n        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)\n        ax.set_xlabel("Nearest training Morgan Tanimoto")\n        ax.set_ylabel("Locked GNN G4-positive probability")\n        ax.set_title("Training similarity and GNN score")\n        plt.tight_layout()\n        fig.savefig(output / "supplementary_positive_control_similarity_vs_score.png")\n        plt.close(fig)\n        generated.append("supplementary_positive_control_similarity_vs_score.png")\n\n    return generated\n\n\ndef write_report(\n    output: Path,\n    inputs: dict[str, str],\n    overall: dict[str, Any],\n    summary: pd.DataFrame,\n    correlations: pd.DataFrame,\n    generated_files: list[str],\n) -> None:\n    positive_components = summary.set_index("source_component")["predicted_positive_rows_at_0_5"].to_dict() if not summary.empty else {}\n    lines = [\n        "# BindingDB Supplementary Positive-Control Bioactivity Support",\n        "",\n        f"Generated: {datetime.now().isoformat(timespec=\'seconds\')}",\n        "",\n        "## Scope",\n        "",\n        "This workflow combines BindingDB compounds that are exact training overlap, near analogs, or known G4 binder/inhibitor records. It is supplementary positive-control bioactivity support for the locked Molecular Graph GNN checkpoint, not independent external validation.",\n        "",\n        "No retraining, probability-threshold tuning, or model refitting is performed. Existing locked-checkpoint prediction artifacts are consumed, and the required checkpoint path is checked for provenance.",\n        "",\n        "## Inputs",\n        "",\n    ]\n    for key, value in inputs.items():\n        lines.append(f"- {key}: `{value}`")\n    lines.extend(\n        [\n            "",\n            "## Counts",\n            "",\n            f"- Combined component-specific rows: {overall[\'rows\']}",\n            f"- Unique compounds: {overall[\'unique_compounds\']}",\n            f"- Rows with nearest training similarity >= 0.8: {overall[\'near_analog_rows_ge_0_8\']}",\n            f"- Rows with nearest training similarity >= 0.9: {overall[\'near_exact_rows_ge_0_9\']}",\n            f"- Rows scored G4-positive at the locked 0.5 threshold: {overall[\'predicted_positive_rows_at_0_5\']}",\n            "",\n            "## Component Summary",\n            "",\n        ]\n    )\n    for row in summary.to_dict("records"):\n        lines.append(\n            "- "\n            f"{row[\'source_component\']}: {row[\'rows\']} rows, {row[\'unique_compounds\']} unique compounds, "\n            f"{row[\'predicted_positive_rows_at_0_5\']} rows scored positive, "\n            f"median score {row[\'gnn_probability_median\']:.4f}."\n        )\n    lines.extend(["", "## Score versus pActivity", ""])\n    computed = correlations[correlations["status"] == "computed"]\n    if computed.empty:\n        lines.append("- No Spearman correlations were computed because no endpoint scope met the minimum data requirements.")\n    else:\n        for row in computed.to_dict("records"):\n            lines.append(f"- {row[\'scope\']} / {row[\'endpoint_type\']}: Spearman rho {row[\'spearman_rho\']:.3f} (n={row[\'n\']}).")\n    skipped = correlations[correlations["status"] != "computed"]\n    if not skipped.empty:\n        lines.append("- Skipped endpoint scopes are documented in `supplementary_positive_control_spearman_correlations.csv` with the reason.")\n    lines.extend(\n        [\n            "",\n            "## Findings",\n            "",\n            f"- BindingDB-supported known G4 binder/inhibitor rows show {positive_components.get(\'bindingdb_supported_known_g4_binder_inhibitor\', 0)} positive calls at threshold 0.5.",\n            "- High-similarity Kd-only BindingDB rows are retained as training-overlap/analog support and should not be used to claim leakage-filtered generalization.",\n            "- Potency endpoints differ across Kd and IC50 records; pActivity summaries are descriptive only.",\n            "",\n            "## Limitations",\n            "",\n            "- These rows are selected because they overlap with, resemble, or support known G4-positive chemistry. This enriches for positive-control behavior and is not an unbiased external benchmark.",\n            "- BindingDB-supported known-G4 rows recovered here are protein/helicase inhibition bioactivity records, not direct G-quadruplex binding constants.",\n            "- The Kd-only high-similarity component is small, so endpoint correlations are underpowered where n is below the configured minimum.",\n            "",\n            "## Suggested Manuscript Wording",\n            "",\n            "As supplementary positive-control bioactivity support, we evaluated BindingDB compounds that either overlapped with or were close analogs of the locked training chemistry, together with BindingDB-supported records for known G-quadruplex binders/inhibitors. The locked scaffold-split Molecular Graph GNN checkpoint was used without retraining or threshold tuning. These analyses are reported as traceability and positive-control support only, not as independent external validation.",\n            "",\n            "## Generated Files",\n            "",\n        ]\n    )\n    for file_name in generated_files:\n        lines.append(f"- `{file_name}`")\n    (output / "bindingdb_supplementary_positive_control_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf-8")\n\n\ndef main() -> int:\n    args = parse_args()\n    root = root_from_script()\n    output = resolve(args.output_dir, root)\n    output.mkdir(parents=True, exist_ok=True)\n    checkpoint = resolve(args.checkpoint, root)\n    if checkpoint != resolve(DEFAULT_CHECKPOINT, root):\n        raise SystemExit(f"Use the locked scaffold checkpoint only: {DEFAULT_CHECKPOINT}")\n    if not checkpoint.exists():\n        raise SystemExit(f"Locked checkpoint missing: {checkpoint}")\n\n    kd_predictions = resolve(args.kd_predictions, root)\n    kd_similarity = resolve(args.kd_similarity, root)\n    known_predictions = resolve(args.known_g4_predictions, root)\n    known_summary = resolve(args.known_g4_summary, root)\n\n    kd_df, kd_input = read_existing(kd_predictions, kd_similarity)\n    known_df, known_input = read_existing(known_predictions, known_summary)\n\n    kd_rows = build_kd_rows(kd_df)\n    known_rows = build_known_g4_rows(known_df)\n    combined = pd.concat([kd_rows, known_rows], ignore_index=True)\n    if combined.empty:\n        raise SystemExit("No supplementary positive-control rows were assembled.")\n\n    columns = [\n        "source_component",\n        "compound_name",\n        "canonical_smiles",\n        "inchikey",\n        "endpoint_type",\n        "activity_value_nM",\n        "pActivity",\n        "bindingdb_target_text",\n        "bindingdb_assay_text",\n        "target_assay_text",\n        "bindingdb_record_count",\n        "nearest_training_similarity",\n        "nearest_training_smiles",\n        "exact_training_inchikey_match",\n        "gnn_probability",\n        "predicted_class",\n        "original_predicted_label",\n        "activity_label_from_source",\n        "interpretation_label",\n        "source_row_key",\n    ]\n    combined = combined.reindex(columns=columns).sort_values(["source_component", "gnn_probability"], ascending=[True, False])\n    combined.to_csv(output / "supplementary_positive_control_combined.csv", index=False)\n\n    compound_level = (\n        combined.groupby("inchikey", dropna=False)\n        .agg(\n            compound_name=("compound_name", "first"),\n            canonical_smiles=("canonical_smiles", "first"),\n            source_components=("source_component", lambda x: text_join(sorted(set(map(str, x))))),\n            endpoint_types=("endpoint_type", lambda x: text_join(sorted(set(map(str, x))))),\n            best_pActivity=("pActivity", "max"),\n            max_gnn_probability=("gnn_probability", "max"),\n            predicted_class=("predicted_class", lambda x: "G4_positive_at_0.5" if "G4_positive_at_0.5" in set(map(str, x)) else "not_G4_positive_at_0.5"),\n            max_nearest_training_similarity=("nearest_training_similarity", "max"),\n            exact_training_inchikey_match=("exact_training_inchikey_match", "max"),\n            interpretation_labels=("interpretation_label", lambda x: text_join(list(x))),\n        )\n        .reset_index()\n        .sort_values("max_gnn_probability", ascending=False)\n    )\n    compound_level.to_csv(output / "supplementary_positive_control_compound_level.csv", index=False)\n\n    summary, endpoint_counts, overall = summarize(combined)\n    correlations = correlation_rows(combined)\n    summary.to_csv(output / "supplementary_positive_control_component_summary.csv", index=False)\n    endpoint_counts.to_csv(output / "supplementary_positive_control_endpoint_counts.csv", index=False)\n    correlations.to_csv(output / "supplementary_positive_control_spearman_correlations.csv", index=False)\n    (output / "supplementary_positive_control_summary.json").write_text(\n        json.dumps(\n            {\n                "generated": datetime.now().isoformat(timespec="seconds"),\n                "scope": "supplementary_positive_control_bioactivity_support_not_independent_external_validation",\n                "locked_checkpoint": str(checkpoint.relative_to(root)),\n                "inputs": {\n                    "kd_component": str(kd_input.relative_to(root)),\n                    "known_g4_component": str(known_input.relative_to(root)),\n                },\n                "overall": overall,\n            },\n            indent=2,\n        )\n        + "\\n",\n        encoding="utf-8",\n    )\n    generated = [\n        "supplementary_positive_control_combined.csv",\n        "supplementary_positive_control_compound_level.csv",\n        "supplementary_positive_control_component_summary.csv",\n        "supplementary_positive_control_endpoint_counts.csv",\n        "supplementary_positive_control_spearman_correlations.csv",\n        "supplementary_positive_control_summary.json",\n    ]\n    generated.extend(write_plots(combined, output))\n    generated.append("bindingdb_supplementary_positive_control_report.md")\n    write_report(\n        output,\n        {\n            "locked_checkpoint": str(checkpoint.relative_to(root)),\n            "kd_component": str(kd_input.relative_to(root)),\n            "known_g4_component": str(known_input.relative_to(root)),\n        },\n        overall,\n        summary,\n        correlations,\n        generated,\n    )\n    print(json.dumps({"output_dir": str(output), "overall": overall}, indent=2))\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'


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
