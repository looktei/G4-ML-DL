#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Run BindingDB Kd-only locked-GNN external validation."""

STEP_ID = '17'
STEP_TITLE = 'Run BindingDB Kd-only locked-GNN external validation'
EMBEDDED_SOURCE_NAME = '18_validate_bindingdb_kd_gnn_external.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""BindingDB Kd-only external validation for the locked molecular graph GNN."""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport math\nimport re\nimport sys\nfrom dataclasses import dataclass\nfrom datetime import datetime\nfrom pathlib import Path\nfrom typing import Any\n\nimport matplotlib\n\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\nfrom rdkit import Chem, DataStructs, RDLogger\nfrom rdkit.Chem import rdFingerprintGenerator\nfrom rdkit.Chem.MolStandardize import rdMolStandardize\nfrom rdkit.Chem.Scaffolds import MurckoScaffold\nfrom sklearn.metrics import (\n    average_precision_score,\n    balanced_accuracy_score,\n    brier_score_loss,\n    confusion_matrix,\n    matthews_corrcoef,\n    precision_recall_curve,\n    recall_score,\n    roc_auc_score,\n    roc_curve,\n)\n\n\nACTIVE_PKD = 6.0\nWEAK_PKD = 5.0\nPRIMARY_LEAKAGE_CUTOFF = 0.8\nSENSITIVITY_LEAKAGE_CUTOFF = 0.9\nDEFAULT_RAW_CANDIDATES = [\n    "data/bindingdb/BindingDB_All.tsv",\n    "data/bindingdb/BindingDB_All_2026m5.tsv",\n    "data/bindingdb/BindingDB_Kd.tsv",\n    "data/bindingdb/bindingdb_kd.tsv",\n    "data/bindingdb/bindingdb_kd.csv",\n]\nSMILES_COLUMNS = [\n    "Ligand SMILES",\n    "Ligand_SMILES",\n    "SMILES",\n    "smiles",\n    "canonical_smiles",\n    "Canonical SMILES",\n]\nKD_COLUMNS = [\n    "Kd (nM)",\n    "Kd(nM)",\n    "Kd_nM",\n    "Kd",\n    "kd",\n    "kd_nM",\n]\nTARGET_COLUMNS = [\n    "Target Name",\n    "Target Source Organism According to Curator or DataSource",\n    "UniProt (SwissProt) Primary ID of Target Chain",\n    "BindingDB Target Chain Sequence",\n    "target_name",\n    "target",\n]\nPASSTHROUGH_COLUMNS = [\n    "BindingDB MonomerID",\n    "BindingDB Reactant_set_id",\n    "BindingDB Ligand Name",\n    "Target Name",\n    "UniProt (SwissProt) Primary ID of Target Chain",\n    "Article DOI",\n    "PMID",\n    "Institution",\n    "Source",\n    "Assay Description",\n]\n\n\n@dataclass(frozen=True)\nclass Status:\n    ok: bool\n    message: str\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument(\n        "--raw-bindingdb",\n        default=None,\n        help="Raw BindingDB table. If omitted, common data/bindingdb paths are probed.",\n    )\n    parser.add_argument(\n        "--training-dataset",\n        default="data/modeling/g4ldb_chembl_binary_dataset.csv",\n        help="Locked GNN training dataset used only for similarity/leakage calculations.",\n    )\n    parser.add_argument(\n        "--checkpoint",\n        default="data/modeling/deep_learning_gnn/gnn_molecular_graph_scaffold_seed42.pt",\n        help="Locked molecular graph GNN checkpoint. Defaults to scaffold seed 42.",\n    )\n    parser.add_argument(\n        "--output-dir",\n        default="data/bindingdb_kd_external_validation",\n        help="Output directory for tables, figures, and report.",\n    )\n    parser.add_argument("--smiles-column", default=None, help="Override raw BindingDB SMILES column.")\n    parser.add_argument("--kd-column", default=None, help="Override raw BindingDB Kd column.")\n    parser.add_argument(\n        "--target-query",\n        default=None,\n        help="Optional case-insensitive regex applied across target columns before processing.",\n    )\n    parser.add_argument("--batch-size", type=int, default=256, help="Prediction batch size.")\n    parser.add_argument("--cpu", action="store_true", help="Force CPU prediction if torch is available.")\n    parser.add_argument("--skip-predict", action="store_true", help="Build tables/figures without GNN prediction.")\n    parser.add_argument(\n        "--sample-per-class",\n        type=int,\n        default=None,\n        help="Optional balanced sampling size per class after Kd labeling and deduplication.",\n    )\n    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for optional balanced sampling.")\n    parser.add_argument("--fingerprint-radius", type=int, default=2)\n    parser.add_argument("--fingerprint-bits", type=int, default=2048)\n    return parser.parse_args()\n\n\ndef resolve_path(path: str | Path, root: Path) -> Path:\n    path = Path(path)\n    return path if path.is_absolute() else root / path\n\n\ndef find_raw_bindingdb(root: Path, raw_arg: str | None) -> Path | None:\n    if raw_arg:\n        return resolve_path(raw_arg, root)\n    for candidate in DEFAULT_RAW_CANDIDATES:\n        path = root / candidate\n        if path.exists():\n            return path\n    return None\n\n\ndef read_table(path: Path) -> pd.DataFrame:\n    suffix = path.suffix.lower()\n    if suffix in {".csv"}:\n        return pd.read_csv(path, low_memory=False)\n    return pd.read_csv(path, sep="\\t", low_memory=False)\n\n\ndef pick_column(df: pd.DataFrame, preferred: str | None, candidates: list[str], label: str) -> str:\n    if preferred:\n        if preferred not in df.columns:\n            raise SystemExit(f"{label} column not found: {preferred}")\n        return preferred\n    for column in candidates:\n        if column in df.columns:\n            return column\n    lower = {column.lower(): column for column in df.columns}\n    for column in candidates:\n        if column.lower() in lower:\n            return lower[column.lower()]\n    raise SystemExit(f"Could not infer {label} column. Available columns include: {list(df.columns[:20])}")\n\n\ndef parse_kd_to_nm(value: Any, column_name: str) -> tuple[float, str]:\n    if pd.isna(value):\n        return np.nan, ""\n    text = str(value).strip()\n    if not text or text.lower() in {"nan", "none", "n/a", "na"}:\n        return np.nan, ""\n    relation = ""\n    relation_match = re.match(r"^(<=|>=|<|>|=|~|approx\\.?|ca\\.?)\\s*", text, flags=re.IGNORECASE)\n    if relation_match:\n        relation = relation_match.group(1)\n    cleaned = text.replace(",", "")\n    number_match = re.search(r"[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?", cleaned)\n    if not number_match:\n        return np.nan, relation\n    value_float = float(number_match.group(0))\n    unit_text = f"{column_name} {text}".lower()\n    if "pm" in unit_text and "npm" not in unit_text:\n        factor = 1e-3\n    elif "um" in unit_text or "microm" in unit_text or "micro" in unit_text or "u m" in unit_text:\n        factor = 1e3\n    elif "mm" in unit_text and "pm" not in unit_text:\n        factor = 1e6\n    elif re.search(r"(^|[^a-z])m([^a-z]|$)", unit_text) and "nm" not in unit_text:\n        factor = 1e9\n    else:\n        factor = 1.0\n    return value_float * factor, relation\n\n\ndef standardize_mol(smiles: Any) -> tuple[Chem.Mol | None, str | None, str | None]:\n    if pd.isna(smiles):\n        return None, None, None\n    mol = Chem.MolFromSmiles(str(smiles))\n    if mol is None:\n        return None, None, None\n    try:\n        mol = rdMolStandardize.Cleanup(mol)\n        mol = rdMolStandardize.FragmentParent(mol)\n        mol = rdMolStandardize.Uncharger().uncharge(mol)\n    except Exception:\n        pass\n    try:\n        Chem.SanitizeMol(mol)\n    except Exception:\n        return None, None, None\n    canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)\n    inchikey = Chem.MolToInchiKey(mol)\n    return mol, canonical, inchikey\n\n\ndef first_nonempty(values: pd.Series) -> str:\n    for value in values.dropna():\n        text = str(value).strip()\n        if text:\n            return text\n    return ""\n\n\ndef join_unique(values: pd.Series, limit: int = 12) -> str:\n    seen: list[str] = []\n    for value in values.dropna():\n        text = str(value).strip()\n        if text and text not in seen:\n            seen.append(text)\n        if len(seen) >= limit:\n            break\n    return " | ".join(seen)\n\n\ndef build_kd_table(raw: pd.DataFrame, smiles_col: str, kd_col: str, target_query: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:\n    df = raw.copy()\n    counts: dict[str, Any] = {"raw_rows": int(len(df)), "raw_columns": list(raw.columns)}\n    if target_query:\n        target_cols = [column for column in TARGET_COLUMNS if column in df.columns]\n        if not target_cols:\n            raise SystemExit("--target-query was supplied, but no target metadata columns were found.")\n        pattern = re.compile(target_query, flags=re.IGNORECASE)\n        target_text = df[target_cols].fillna("").astype(str).agg(" ".join, axis=1)\n        df = df[target_text.str.contains(pattern, na=False)].copy()\n        counts["after_target_query_rows"] = int(len(df))\n        counts["target_query"] = target_query\n        counts["target_query_columns"] = target_cols\n\n    parsed = df[kd_col].apply(lambda value: parse_kd_to_nm(value, kd_col))\n    df["kd_nM"] = [item[0] for item in parsed]\n    df["kd_relation"] = [item[1] for item in parsed]\n    df = df[np.isfinite(df["kd_nM"]) & (df["kd_nM"] > 0)].copy()\n    counts["after_valid_kd_rows"] = int(len(df))\n\n    RDLogger.DisableLog("rdApp.*")\n    standardized = df[smiles_col].apply(standardize_mol)\n    df["canonical_smiles"] = [item[1] for item in standardized]\n    df["inchikey"] = [item[2] for item in standardized]\n    df = df.dropna(subset=["canonical_smiles", "inchikey"]).copy()\n    counts["after_valid_smiles_rows"] = int(len(df))\n\n    df["pKd"] = 9.0 - np.log10(df["kd_nM"].astype(float))\n    aggregate: dict[str, Any] = {\n        "canonical_smiles": ("canonical_smiles", first_nonempty),\n        "kd_nM_median": ("kd_nM", "median"),\n        "kd_nM_min": ("kd_nM", "min"),\n        "kd_nM_max": ("kd_nM", "max"),\n        "kd_record_count": ("kd_nM", "size"),\n        "pKd_median": ("pKd", "median"),\n        "pKd_best": ("pKd", "max"),\n        "kd_relation_examples": ("kd_relation", join_unique),\n    }\n    for column in PASSTHROUGH_COLUMNS:\n        if column in df.columns:\n            safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", column).strip("_").lower()\n            aggregate[safe_name] = (column, join_unique)\n    dedup = df.groupby("inchikey", as_index=False).agg(**aggregate)\n    dedup["label"] = np.select(\n        [dedup["pKd_median"] >= ACTIVE_PKD, dedup["pKd_median"] <= WEAK_PKD],\n        [1, 0],\n        default=np.nan,\n    )\n    dedup["endpoint_zone"] = np.select(\n        [dedup["pKd_median"] >= ACTIVE_PKD, dedup["pKd_median"] <= WEAK_PKD],\n        ["active_kd_le_1uM", "weak_kd_ge_10uM"],\n        default="ambiguous_1_to_10uM",\n    )\n    labeled = dedup.dropna(subset=["label"]).copy()\n    labeled["label"] = labeled["label"].astype(int)\n    counts["deduplicated_compounds"] = int(len(dedup))\n    counts["labeled_nonambiguous_compounds"] = int(len(labeled))\n    counts["active_compounds"] = int((labeled["label"] == 1).sum())\n    counts["weak_inactive_like_compounds"] = int((labeled["label"] == 0).sum())\n    counts["ambiguous_compounds_excluded"] = int((dedup["endpoint_zone"] == "ambiguous_1_to_10uM").sum())\n    return labeled.reset_index(drop=True), counts\n\n\ndef balanced_sample_labeled(labeled: pd.DataFrame, sample_per_class: int | None, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:\n    if sample_per_class is None:\n        return labeled, {}\n    rows = []\n    counts: dict[str, Any] = {\n        "requested_sample_per_class": int(sample_per_class),\n        "sample_seed": int(seed),\n    }\n    for label, label_name in [(1, "active"), (0, "weak_inactive_like")]:\n        group = labeled[labeled["label"] == label].copy()\n        counts[f"{label_name}_available_before_sampling"] = int(len(group))\n        take = min(sample_per_class, len(group))\n        counts[f"{label_name}_sampled"] = int(take)\n        if take:\n            rows.append(group.sample(n=take, random_state=seed))\n    if not rows:\n        return labeled.iloc[0:0].copy(), counts\n    sampled = pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)\n    counts["sampled_labeled_nonambiguous_compounds"] = int(len(sampled))\n    return sampled, counts\n\n\ndef load_training_mols(path: Path) -> pd.DataFrame:\n    df = pd.read_csv(path)\n    if "canonical_smiles" not in df.columns:\n        raise SystemExit(f"Training dataset missing canonical_smiles: {path}")\n    rows = []\n    for i, smiles in enumerate(df["canonical_smiles"]):\n        mol, canonical, inchikey = standardize_mol(smiles)\n        if mol is None or canonical is None or inchikey is None:\n            continue\n        rows.append({"training_index": i, "training_canonical_smiles": canonical, "training_inchikey": inchikey, "mol": mol})\n    if not rows:\n        raise SystemExit("No parseable training molecules for similarity calculation.")\n    return pd.DataFrame(rows)\n\n\ndef add_training_similarity(\n    external: pd.DataFrame,\n    training: pd.DataFrame,\n    radius: int,\n    n_bits: int,\n) -> pd.DataFrame:\n    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)\n    train_fps = [generator.GetFingerprint(mol) for mol in training["mol"]]\n    train_inchikeys = training["training_inchikey"].tolist()\n    train_smiles = training["training_canonical_smiles"].tolist()\n    nearest_similarity = []\n    nearest_index = []\n    nearest_smiles = []\n    exact_match = []\n    external_mols = []\n    for smiles, inchikey in zip(external["canonical_smiles"], external["inchikey"]):\n        mol = Chem.MolFromSmiles(smiles)\n        external_mols.append(mol)\n        fp = generator.GetFingerprint(mol)\n        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)\n        best_idx = int(np.argmax(sims))\n        nearest_similarity.append(float(sims[best_idx]))\n        nearest_index.append(best_idx)\n        nearest_smiles.append(train_smiles[best_idx])\n        exact_match.append(bool(inchikey in train_inchikeys))\n    out = external.copy()\n    out["nearest_train_similarity"] = nearest_similarity\n    out["nearest_train_index"] = nearest_index\n    out["nearest_train_smiles"] = nearest_smiles\n    out["exact_training_inchikey_match"] = exact_match\n    out["leakage_keep_tanimoto_lt_0_8"] = out["nearest_train_similarity"] < PRIMARY_LEAKAGE_CUTOFF\n    out["leakage_keep_tanimoto_lt_0_9"] = out["nearest_train_similarity"] < SENSITIVITY_LEAKAGE_CUTOFF\n    out["applicability_domain_bin"] = pd.cut(\n        out["nearest_train_similarity"],\n        bins=[-0.001, 0.3, 0.5, 0.8, 0.9, 1.0],\n        labels=["remote_lt_0_3", "low_0_3_to_0_5", "moderate_0_5_to_0_8", "analog_0_8_to_0_9", "near_exact_ge_0_9"],\n    ).astype(str)\n    out["murcko_scaffold"] = [\n        MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol is not None else ""\n        for mol in external_mols\n    ]\n    out["murcko_scaffold"] = np.where(out["murcko_scaffold"] == "", "acyclic::" + out["canonical_smiles"], out["murcko_scaffold"])\n    return out\n\n\ndef evaluate_predictions(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:\n    prediction = (probability >= 0.5).astype(int)\n    labels = np.unique(y_true)\n    out: dict[str, Any] = {\n        "n": int(len(y_true)),\n        "positives": int(y_true.sum()),\n        "negatives": int(len(y_true) - y_true.sum()),\n        "threshold": 0.5,\n    }\n    if len(labels) == 2:\n        tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()\n        out.update(\n            {\n                "roc_auc": float(roc_auc_score(y_true, probability)),\n                "pr_auc": float(average_precision_score(y_true, probability)),\n                "brier_score": float(brier_score_loss(y_true, probability)),\n                "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),\n                "sensitivity_recall": float(recall_score(y_true, prediction, zero_division=0)),\n                "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,\n                "mcc": float(matthews_corrcoef(y_true, prediction)),\n                "tn": int(tn),\n                "fp": int(fp),\n                "fn": int(fn),\n                "tp": int(tp),\n            }\n        )\n    return out\n\n\ndef add_bootstrap_ci(y_true: np.ndarray, probability: np.ndarray, seed: int = 42, n_bootstrap: int = 1000) -> dict[str, Any]:\n    if len(np.unique(y_true)) < 2:\n        return {}\n    rng = np.random.default_rng(seed)\n    metrics: dict[str, list[float]] = {"roc_auc": [], "pr_auc": [], "mcc": [], "balanced_accuracy": [], "brier_score": []}\n    for _ in range(n_bootstrap):\n        idx = rng.integers(0, len(y_true), len(y_true))\n        if len(np.unique(y_true[idx])) < 2:\n            continue\n        row = evaluate_predictions(y_true[idx], probability[idx])\n        for metric in metrics:\n            if metric in row and np.isfinite(row[metric]):\n                metrics[metric].append(float(row[metric]))\n    ci = {}\n    for metric, values in metrics.items():\n        if values:\n            ci[f"{metric}_ci_low"] = float(np.percentile(values, 2.5))\n            ci[f"{metric}_ci_high"] = float(np.percentile(values, 97.5))\n    return ci\n\n\ndef plot_kd_distribution(df: pd.DataFrame, output: Path) -> None:\n    sns.set_theme(style="whitegrid", context="talk")\n    fig, ax = plt.subplots(figsize=(9, 6), dpi=220)\n    sns.histplot(data=df, x="pKd_median", hue="endpoint_zone", bins=30, multiple="stack", ax=ax)\n    ax.axvline(WEAK_PKD, color="black", linestyle="--", linewidth=1)\n    ax.axvline(ACTIVE_PKD, color="black", linestyle="--", linewidth=1)\n    ax.set_xlabel("Median pKd")\n    ax.set_ylabel("Compounds")\n    ax.set_title("BindingDB Kd-only endpoint distribution")\n    plt.tight_layout()\n    plt.savefig(output / "bindingdb_kd_pkd_distribution.png")\n    plt.close()\n\n\ndef plot_similarity(df: pd.DataFrame, output: Path) -> None:\n    sns.set_theme(style="whitegrid", context="talk")\n    fig, ax = plt.subplots(figsize=(9, 6), dpi=220)\n    sns.histplot(data=df, x="nearest_train_similarity", hue="endpoint_zone", bins=30, multiple="stack", ax=ax)\n    ax.axvline(PRIMARY_LEAKAGE_CUTOFF, color="black", linestyle="--", linewidth=1, label="0.8 cutoff")\n    ax.axvline(SENSITIVITY_LEAKAGE_CUTOFF, color="gray", linestyle=":", linewidth=1, label="0.9 cutoff")\n    ax.set_xlabel("Nearest training Morgan Tanimoto")\n    ax.set_ylabel("Compounds")\n    ax.set_title("External-set similarity to locked training set")\n    ax.legend()\n    plt.tight_layout()\n    plt.savefig(output / "bindingdb_kd_training_similarity.png")\n    plt.close()\n\n\ndef plot_prediction_figures(df: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:\n    if "gnn_probability" not in df.columns:\n        return\n    sns.set_theme(style="whitegrid", context="talk")\n    fig, ax = plt.subplots(figsize=(9, 6), dpi=220)\n    sns.boxplot(data=df, x="endpoint_zone", y="gnn_probability", ax=ax)\n    sns.stripplot(data=df, x="endpoint_zone", y="gnn_probability", color="black", alpha=0.35, size=3, ax=ax)\n    ax.set_xlabel("")\n    ax.set_ylabel("Locked GNN positive probability")\n    ax.set_title("GNN scores by BindingDB Kd label")\n    plt.xticks(rotation=10)\n    plt.tight_layout()\n    plt.savefig(output / "bindingdb_kd_gnn_score_distribution.png")\n    plt.close()\n\n    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)\n    sns.scatterplot(data=df, x="pKd_median", y="gnn_probability", hue="nearest_train_similarity", palette="viridis", ax=ax)\n    ax.axvline(WEAK_PKD, color="black", linestyle="--", linewidth=1)\n    ax.axvline(ACTIVE_PKD, color="black", linestyle="--", linewidth=1)\n    ax.set_xlabel("Median pKd")\n    ax.set_ylabel("Locked GNN positive probability")\n    ax.set_title("BindingDB potency versus GNN score")\n    plt.tight_layout()\n    plt.savefig(output / "bindingdb_kd_pkd_vs_gnn_score.png")\n    plt.close()\n\n    primary = df[df["leakage_keep_tanimoto_lt_0_8"]].copy()\n    if len(primary) and primary["label"].nunique() == 2:\n        y = primary["label"].to_numpy()\n        p = primary["gnn_probability"].to_numpy()\n        fpr, tpr, _ = roc_curve(y, p)\n        precision, recall, _ = precision_recall_curve(y, p)\n        row = metrics[metrics["subset"] == "leakage_filtered_tanimoto_lt_0_8"].iloc[0]\n        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=220)\n        axes[0].plot(fpr, tpr, linewidth=2, label=f"ROC-AUC={row[\'roc_auc\']:.3f}")\n        axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)\n        axes[0].set_xlabel("False positive rate")\n        axes[0].set_ylabel("True positive rate")\n        axes[0].set_title("Primary leakage-filtered ROC")\n        axes[0].legend()\n        axes[1].plot(recall, precision, linewidth=2, label=f"PR-AUC={row[\'pr_auc\']:.3f}")\n        axes[1].set_xlabel("Recall")\n        axes[1].set_ylabel("Precision")\n        axes[1].set_title("Primary leakage-filtered PR")\n        axes[1].legend()\n        plt.tight_layout()\n        plt.savefig(output / "bindingdb_kd_primary_roc_pr_curves.png")\n        plt.close()\n\n\ndef import_torch_stack() -> tuple[Status, Any, Any, Any, Any, Any, Any]:\n    try:\n        import torch\n        from torch import nn\n        from torch_geometric.data import Data\n        from torch_geometric.loader import DataLoader\n        from torch_geometric.nn import BatchNorm, GINEConv, global_max_pool, global_mean_pool\n    except Exception as exc:\n        return Status(False, f"Missing GNN runtime dependency: {type(exc).__name__}: {exc}"), None, None, None, None, None, None\n    return Status(True, "GNN runtime dependencies available."), torch, nn, Data, DataLoader, (BatchNorm, GINEConv, global_max_pool, global_mean_pool), None\n\n\ndef predict_with_locked_gnn(df: pd.DataFrame, checkpoint: Path, batch_size: int, force_cpu: bool) -> tuple[pd.DataFrame, Status]:\n    status, torch, nn, Data, DataLoader, pyg_items, _ = import_torch_stack()\n    if not status.ok:\n        return df, status\n    if not checkpoint.exists():\n        return df, Status(False, f"Locked GNN checkpoint not found: {checkpoint}")\n    BatchNorm, GINEConv, global_max_pool, global_mean_pool = pyg_items\n    atom_feature_dims = [119, 8, 7, 6, 8, 6, 2, 2]\n    bond_feature_dims = [5, 2, 2, 7]\n\n    class CategoricalFeatureEncoder(nn.Module):\n        def __init__(self, feature_dims: list[int], hidden_dim: int) -> None:\n            super().__init__()\n            self.embeddings = nn.ModuleList([nn.Embedding(dim, hidden_dim) for dim in feature_dims])\n\n        def forward(self, features: Any) -> Any:\n            features = features.long()\n            out = 0\n            for column, embedding in enumerate(self.embeddings):\n                values = features[:, column].clamp(min=0, max=embedding.num_embeddings - 1)\n                out = out + embedding(values)\n            return out\n\n    class MolecularGNN(nn.Module):\n        def __init__(self, hidden_dim: int, layers: int, dense_hidden: int, dropout: float) -> None:\n            super().__init__()\n            self.atom_encoder = CategoricalFeatureEncoder(atom_feature_dims, hidden_dim)\n            self.bond_encoder = CategoricalFeatureEncoder(bond_feature_dims, hidden_dim)\n            self.convs = nn.ModuleList()\n            self.norms = nn.ModuleList()\n            for _ in range(layers):\n                mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim))\n                self.convs.append(GINEConv(mlp, train_eps=True))\n                self.norms.append(BatchNorm(hidden_dim))\n            self.dropout = nn.Dropout(dropout)\n            self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2, dense_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dense_hidden, 1))\n\n        def forward(self, data: Any) -> Any:\n            x = self.atom_encoder(data.x)\n            edge_attr = self.bond_encoder(data.edge_attr)\n            for conv, norm in zip(self.convs, self.norms):\n                residual = x\n                x = conv(x, data.edge_index, edge_attr)\n                x = norm(x)\n                x = torch.relu(x)\n                x = self.dropout(x)\n                x = x + residual\n            pooled = torch.cat([global_mean_pool(x, data.batch), global_max_pool(x, data.batch)], dim=1)\n            return self.classifier(pooled).squeeze(1)\n\n    def bounded_lookup(value: int, allowed: list[int]) -> int:\n        return allowed.index(value) if value in allowed else len(allowed)\n\n    def atom_features(atom: Chem.Atom) -> list[int]:\n        charge_values = [-3, -2, -1, 0, 1, 2]\n        hybrid_values = [\n            Chem.rdchem.HybridizationType.S,\n            Chem.rdchem.HybridizationType.SP,\n            Chem.rdchem.HybridizationType.SP2,\n            Chem.rdchem.HybridizationType.SP3,\n            Chem.rdchem.HybridizationType.SP3D,\n            Chem.rdchem.HybridizationType.SP3D2,\n            Chem.rdchem.HybridizationType.UNSPECIFIED,\n        ]\n        return [\n            min(atom.GetAtomicNum(), 118),\n            min(atom.GetTotalDegree(), 7),\n            bounded_lookup(atom.GetFormalCharge(), charge_values),\n            min(int(atom.GetTotalNumHs()), 5),\n            hybrid_values.index(atom.GetHybridization()) if atom.GetHybridization() in hybrid_values else 7,\n            min(int(atom.GetTotalValence()), 5),\n            int(atom.GetIsAromatic()),\n            int(atom.IsInRing()),\n        ]\n\n    def bond_features(bond: Chem.Bond) -> list[int]:\n        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]\n        stereo_values = [\n            Chem.rdchem.BondStereo.STEREONONE,\n            Chem.rdchem.BondStereo.STEREOANY,\n            Chem.rdchem.BondStereo.STEREOZ,\n            Chem.rdchem.BondStereo.STEREOE,\n            Chem.rdchem.BondStereo.STEREOCIS,\n            Chem.rdchem.BondStereo.STEREOTRANS,\n        ]\n        return [\n            bond_types.index(bond.GetBondType()) if bond.GetBondType() in bond_types else 4,\n            int(bond.GetIsConjugated()),\n            int(bond.IsInRing()),\n            stereo_values.index(bond.GetStereo()) if bond.GetStereo() in stereo_values else 6,\n        ]\n\n    def mol_to_graph(mol: Chem.Mol, index: int) -> Any:\n        atom_matrix = [atom_features(atom) for atom in mol.GetAtoms()]\n        edges: list[list[int]] = []\n        edge_features: list[list[int]] = []\n        for bond in mol.GetBonds():\n            begin = bond.GetBeginAtomIdx()\n            end = bond.GetEndAtomIdx()\n            features = bond_features(bond)\n            edges.extend([[begin, end], [end, begin]])\n            edge_features.extend([features, features])\n        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)\n        edge_attr = torch.tensor(edge_features, dtype=torch.long) if edges else torch.empty((0, len(bond_feature_dims)), dtype=torch.long)\n        graph = Data(x=torch.tensor(atom_matrix, dtype=torch.long), edge_index=edge_index, edge_attr=edge_attr)\n        graph.sample_id = torch.tensor([index], dtype=torch.long)\n        return graph\n\n    checkpoint_data = torch.load(checkpoint, map_location="cpu")\n    hidden_dim = int(checkpoint_data.get("hidden_dim", 96))\n    layers = int(checkpoint_data.get("layers", 3))\n    dense_hidden = int(checkpoint_data.get("dense_hidden", 256))\n    dropout = float(checkpoint_data.get("dropout", 0.2))\n    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")\n    model = MolecularGNN(hidden_dim, layers, dense_hidden, dropout).to(device)\n    model.load_state_dict(checkpoint_data["model_state_dict"])\n    model.eval()\n    graphs = [mol_to_graph(Chem.MolFromSmiles(smiles), i) for i, smiles in enumerate(df["canonical_smiles"])]\n    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)\n    probabilities = np.zeros(len(df), dtype=float)\n    with torch.no_grad():\n        for batch in loader:\n            batch = batch.to(device)\n            logits = model(batch)\n            probs = torch.sigmoid(logits).detach().cpu().numpy()\n            indices = batch.sample_id.view(-1).detach().cpu().numpy()\n            probabilities[indices] = probs\n    out = df.copy()\n    out["gnn_probability"] = probabilities\n    out["gnn_predicted_label"] = (out["gnn_probability"] >= 0.5).astype(int)\n    return out, Status(True, f"Predicted with locked checkpoint {checkpoint} on {device}.")\n\n\ndef metrics_for_subsets(df: pd.DataFrame) -> pd.DataFrame:\n    rows = []\n    subsets = {\n        "full_kd_only_nonambiguous": df,\n        "leakage_filtered_tanimoto_lt_0_8": df[df["leakage_keep_tanimoto_lt_0_8"]],\n        "sensitivity_tanimoto_lt_0_9": df[df["leakage_keep_tanimoto_lt_0_9"]],\n        "overlap_or_analog_ge_0_8_supplementary": df[~df["leakage_keep_tanimoto_lt_0_8"]],\n    }\n    for name, group in subsets.items():\n        if group.empty:\n            rows.append({"subset": name, "n": 0, "positives": 0, "negatives": 0})\n            continue\n        y = group["label"].to_numpy(dtype=int)\n        p = group["gnn_probability"].to_numpy(dtype=float)\n        row = {"subset": name, **evaluate_predictions(y, p)}\n        row.update(add_bootstrap_ci(y, p))\n        rows.append(row)\n    return pd.DataFrame(rows)\n\n\ndef write_report(output: Path, statuses: list[Status], counts: dict[str, Any], metrics: pd.DataFrame | None, files: list[str]) -> None:\n    lines = [\n        "# BindingDB Kd-Only External Validation Status",\n        "",\n        f"Generated: {datetime.now().isoformat(timespec=\'seconds\')}",\n        "",\n        "## Locked Decisions",\n        "",\n        "- Endpoint: BindingDB Kd-only.",\n        "- Active: Kd <= 1 uM (pKd >= 6).",\n        "- Weak/inactive-like: Kd >= 10 uM (pKd <= 5).",\n        "- Ambiguous 1-10 uM zone excluded.",\n        "- Primary leakage filter: Morgan Tanimoto < 0.8 to the locked training set.",\n        "- Sensitivity leakage filter: Morgan Tanimoto < 0.9.",\n        "- Exact/analog overlap is supplementary positive-control support only.",\n        "- Prediction model: locked Molecular Graph GNN checkpoint; no retraining or tuning.",\n        "",\n        "## Status",\n        "",\n    ]\n    for status in statuses:\n        marker = "OK" if status.ok else "BLOCKED"\n        lines.append(f"- {marker}: {status.message}")\n    if counts:\n        lines.extend(["", "## Counts", ""])\n        for key, value in counts.items():\n            if key == "raw_columns":\n                continue\n            lines.append(f"- {key}: {value}")\n    if metrics is not None and not metrics.empty:\n        lines.extend(["", "## Metrics", "", metrics.to_markdown(index=False, floatfmt=".4f")])\n    if files:\n        lines.extend(["", "## Generated Files", ""])\n        for file in files:\n            lines.append(f"- `{file}`")\n    lines.extend(\n        [\n            "",\n            "## Notes",\n            "",\n            "Do not treat overlap/analog rows as independent external validation. The primary manuscript result should use the Tanimoto < 0.8 subset if it contains enough active and weak/inactive-like records.",\n        ]\n    )\n    (output / "bindingdb_kd_external_validation_status.md").write_text("\\n".join(lines) + "\\n", encoding="utf-8")\n\n\ndef main() -> int:\n    args = parse_args()\n    root = Path(__file__).resolve().parents[1]\n    output = resolve_path(args.output_dir, root)\n    output.mkdir(parents=True, exist_ok=True)\n    statuses: list[Status] = []\n    files: list[str] = []\n    counts: dict[str, Any] = {}\n\n    raw_path = find_raw_bindingdb(root, args.raw_bindingdb)\n    if raw_path is None or not raw_path.exists():\n        statuses.append(\n            Status(\n                False,\n                "Raw BindingDB table not found. Supply --raw-bindingdb or place a table at data/bindingdb/BindingDB_All.tsv, data/bindingdb/BindingDB_Kd.tsv, or data/bindingdb/bindingdb_kd.csv.",\n            )\n        )\n        checkpoint = resolve_path(args.checkpoint, root)\n        training = resolve_path(args.training_dataset, root)\n        statuses.append(Status(checkpoint.exists(), f"Locked GNN checkpoint {\'found\' if checkpoint.exists() else \'missing\'}: {checkpoint}"))\n        statuses.append(Status(training.exists(), f"Training dataset {\'found\' if training.exists() else \'missing\'}: {training}"))\n        if not args.skip_predict:\n            runtime_status, *_ = import_torch_stack()\n            statuses.append(runtime_status)\n        write_report(output, statuses, counts, None, files)\n        return 2\n\n    statuses.append(Status(True, f"Raw BindingDB table found: {raw_path}"))\n    raw = read_table(raw_path)\n    smiles_col = pick_column(raw, args.smiles_column, SMILES_COLUMNS, "SMILES")\n    kd_col = pick_column(raw, args.kd_column, KD_COLUMNS, "Kd")\n    labeled, counts = build_kd_table(raw, smiles_col, kd_col, args.target_query)\n    labeled, sample_counts = balanced_sample_labeled(labeled, args.sample_per_class, args.sample_seed)\n    counts.update(sample_counts)\n    labeled.to_csv(output / "bindingdb_kd_only_labeled_deduplicated.csv", index=False)\n    files.append("bindingdb_kd_only_labeled_deduplicated.csv")\n    statuses.append(Status(True, f"Built Kd-only nonambiguous table with {len(labeled)} deduplicated compounds."))\n    if labeled.empty:\n        statuses.append(Status(False, "No nonambiguous Kd-only compounds remain after pKd thresholding."))\n        write_report(output, statuses, counts, None, files)\n        return 2\n\n    training_path = resolve_path(args.training_dataset, root)\n    if not training_path.exists():\n        statuses.append(Status(False, f"Training dataset not found for similarity filtering: {training_path}"))\n        write_report(output, statuses, counts, None, files)\n        return 2\n    training = load_training_mols(training_path)\n    scored = add_training_similarity(labeled, training, args.fingerprint_radius, args.fingerprint_bits)\n    scored.to_csv(output / "bindingdb_kd_only_with_training_similarity.csv", index=False)\n    files.append("bindingdb_kd_only_with_training_similarity.csv")\n    plot_kd_distribution(scored, output)\n    plot_similarity(scored, output)\n    files.extend(["bindingdb_kd_pkd_distribution.png", "bindingdb_kd_training_similarity.png"])\n\n    predicted = scored\n    metrics = None\n    if args.skip_predict:\n        statuses.append(Status(True, "Skipped locked GNN prediction by request."))\n    else:\n        checkpoint = resolve_path(args.checkpoint, root)\n        predicted, predict_status = predict_with_locked_gnn(scored, checkpoint, args.batch_size, args.cpu)\n        statuses.append(predict_status)\n        if predict_status.ok:\n            predicted.to_csv(output / "bindingdb_kd_only_locked_gnn_predictions.csv", index=False)\n            files.append("bindingdb_kd_only_locked_gnn_predictions.csv")\n            metrics = metrics_for_subsets(predicted)\n            metrics.to_csv(output / "bindingdb_kd_only_locked_gnn_metrics.csv", index=False)\n            files.append("bindingdb_kd_only_locked_gnn_metrics.csv")\n            top_active = predicted[predicted["label"] == 1].sort_values("gnn_probability", ascending=False).head(50)\n            top_active.to_csv(output / "bindingdb_kd_top_ranked_kd_actives.csv", index=False)\n            files.append("bindingdb_kd_top_ranked_kd_actives.csv")\n            plot_prediction_figures(predicted, metrics, output)\n            files.extend(\n                [\n                    "bindingdb_kd_gnn_score_distribution.png",\n                    "bindingdb_kd_pkd_vs_gnn_score.png",\n                ]\n            )\n            if (output / "bindingdb_kd_primary_roc_pr_curves.png").exists():\n                files.append("bindingdb_kd_primary_roc_pr_curves.png")\n\n    write_report(output, statuses, counts, metrics, files)\n    return 0 if all(status.ok for status in statuses) else 2\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'


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
