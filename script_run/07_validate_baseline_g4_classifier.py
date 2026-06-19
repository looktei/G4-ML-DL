#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Validate Morgan RF classical baseline with random/scaffold repeated splits."""

STEP_ID = '07'
STEP_TITLE = 'Validate Morgan RF classical baseline with random/scaffold repeated splits'
EMBEDDED_SOURCE_NAME = '08_validate_baseline_g4_classifier.py'
SEED_MODE = 'repeated'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Validate and interpret the Phase 1B Morgan+RF G4 ligand baseline."""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport warnings\nfrom dataclasses import dataclass\nfrom datetime import datetime\nfrom pathlib import Path\n\nimport joblib\nimport matplotlib\n\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\nfrom rdkit import Chem, DataStructs\nfrom rdkit import RDLogger\nfrom rdkit.Chem import AllChem, Draw, rdFingerprintGenerator\nfrom rdkit.Chem.Scaffolds import MurckoScaffold\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import (\n    accuracy_score,\n    average_precision_score,\n    balanced_accuracy_score,\n    brier_score_loss,\n    confusion_matrix,\n    f1_score,\n    matthews_corrcoef,\n    precision_score,\n    recall_score,\n    roc_auc_score,\n)\nfrom sklearn.model_selection import train_test_split\n\n\nDEFAULT_SEEDS = [0, 1, 2, 3, 4]\n\n\n@dataclass(frozen=True)\nclass SplitResult:\n    split: str\n    seed: int\n    train_idx: np.ndarray\n    test_idx: np.ndarray\n    model: RandomForestClassifier\n    probability: np.ndarray\n    metrics: dict[str, float | int | str]\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument(\n        "--dataset",\n        default="data/modeling/g4ldb_chembl_binary_dataset.csv",\n        help="Binary G4LDB-positive vs ChEMBL assumed-negative dataset.",\n    )\n    parser.add_argument(\n        "--training-dir",\n        default="data/modeling/baseline_training",\n        help="Existing baseline training output directory used for context only.",\n    )\n    parser.add_argument(\n        "--output-dir",\n        default="data/modeling/baseline_validation",\n        help="New validation output directory.",\n    )\n    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction.")\n    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Repeated split seeds.")\n    parser.add_argument("--primary-seed", type=int, default=42, help="Seed matching the training report.")\n    parser.add_argument("--n-jobs", type=int, default=-1, help="Random forest parallel jobs.")\n    parser.add_argument("--rf-trees", type=int, default=None, help="Random forest tree count.")\n    parser.add_argument("--fingerprint-bits", type=int, default=2048, help="Morgan fingerprint size.")\n    parser.add_argument("--fingerprint-radius", type=int, default=2, help="Morgan fingerprint radius.")\n    parser.add_argument("--top-bits", type=int, default=30, help="Number of top RF bits to report.")\n    parser.add_argument("--in-domain-threshold", type=float, default=0.50, help="Max Tanimoto threshold for in-domain.")\n    parser.add_argument(\n        "--borderline-threshold",\n        type=float,\n        default=0.35,\n        help="Max Tanimoto threshold for borderline domain; lower is out-of-domain.",\n    )\n    return parser.parse_args()\n\n\ndef resolve_path(path: str | Path, root: Path) -> Path:\n    path = Path(path)\n    return path if path.is_absolute() else root / path\n\n\ndef load_dataset(path: Path) -> pd.DataFrame:\n    df = pd.read_csv(path)\n    required = {"canonical_smiles", "label"}\n    missing = sorted(required - set(df.columns))\n    if missing:\n        raise SystemExit(f"Missing required column(s): {\', \'.join(missing)}")\n    df = df.copy()\n    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)\n    before = len(df)\n    df = df.dropna(subset=["canonical_smiles", "label"]).reset_index(drop=True)\n    if len(df) != before:\n        print(f"Dropped {before - len(df)} rows with missing validation fields.")\n    if not set(df["label"].unique()).issubset({0, 1}):\n        raise SystemExit("Labels must be binary 0/1.")\n    return df\n\n\ndef mols_from_smiles(smiles: pd.Series) -> list[Chem.Mol]:\n    mols: list[Chem.Mol] = []\n    bad = []\n    for i, value in enumerate(smiles):\n        mol = Chem.MolFromSmiles(str(value))\n        if mol is None:\n            bad.append(i)\n        else:\n            mols.append(mol)\n    if bad:\n        raise SystemExit(f"Could not parse {len(bad)} canonical SMILES; first bad row index: {bad[0]}")\n    return mols\n\n\ndef morgan_matrix(mols: list[Chem.Mol], radius: int, n_bits: int) -> np.ndarray:\n    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)\n    matrix = np.zeros((len(mols), n_bits), dtype=np.float32)\n    for i, mol in enumerate(mols):\n        arr = np.zeros((n_bits,), dtype=np.int8)\n        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), arr)\n        matrix[i, :] = arr\n    return matrix\n\n\ndef morgan_bitvectors(mols: list[Chem.Mol], radius: int, n_bits: int) -> list[DataStructs.ExplicitBitVect]:\n    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)\n    return [generator.GetFingerprint(mol) for mol in mols]\n\n\ndef murcko_scaffolds(mols: list[Chem.Mol], smiles: pd.Series) -> list[str]:\n    scaffolds = []\n    for mol, smi in zip(mols, smiles):\n        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)\n        scaffolds.append(scaffold if scaffold else f"acyclic::{smi}")\n    return scaffolds\n\n\ndef scaffold_split_indices(scaffolds: list[str], y: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:\n    rng = np.random.default_rng(seed)\n    groups: dict[str, list[int]] = {}\n    for index, scaffold in enumerate(scaffolds):\n        groups.setdefault(scaffold, []).append(index)\n\n    records = []\n    for scaffold, indices in groups.items():\n        idx = np.array(indices, dtype=int)\n        records.append(\n            {\n                "scaffold": scaffold,\n                "indices": idx,\n                "n": len(idx),\n                "pos": int(y[idx].sum()),\n                "shuffle": rng.random(),\n            }\n        )\n    records.sort(key=lambda item: (-item["n"], item["shuffle"]))\n\n    target_n = int(round(len(y) * test_size))\n    target_pos = y.mean()\n    test_indices: list[int] = []\n    test_pos = 0\n\n    for item in records:\n        if not test_indices:\n            take = True\n        else:\n            current_n = len(test_indices)\n            current_score = abs(current_n - target_n) / len(y)\n            current_score += abs((test_pos / max(current_n, 1)) - target_pos)\n            new_n = current_n + item["n"]\n            new_pos = test_pos + item["pos"]\n            new_score = abs(new_n - target_n) / len(y)\n            new_score += abs((new_pos / max(new_n, 1)) - target_pos)\n            take = new_score <= current_score or current_n < target_n * 0.75\n        if take:\n            test_indices.extend(item["indices"].tolist())\n            test_pos += item["pos"]\n        if len(test_indices) >= target_n:\n            break\n\n    test = np.array(sorted(test_indices), dtype=int)\n    train = np.array(sorted(set(range(len(y))) - set(test.tolist())), dtype=int)\n    if len(np.unique(y[test])) < 2:\n        raise RuntimeError(f"Scaffold split seed {seed} has only one test class.")\n    return train, test\n\n\ndef make_rf(seed: int, n_jobs: int, trees: int) -> RandomForestClassifier:\n    return RandomForestClassifier(\n        n_estimators=trees,\n        random_state=seed,\n        n_jobs=n_jobs,\n        class_weight="balanced_subsample",\n        min_samples_leaf=2,\n        max_features="sqrt",\n    )\n\n\ndef predict_probability(model: RandomForestClassifier, x_test: np.ndarray) -> np.ndarray:\n    return model.predict_proba(x_test)[:, 1]\n\n\ndef evaluate(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:\n    prediction = (probability >= 0.5).astype(int)\n    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()\n    return {\n        "roc_auc": float(roc_auc_score(y_true, probability)),\n        "pr_auc": float(average_precision_score(y_true, probability)),\n        "brier_score": float(brier_score_loss(y_true, probability)),\n        "accuracy": float(accuracy_score(y_true, prediction)),\n        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),\n        "sensitivity_recall": float(recall_score(y_true, prediction, zero_division=0)),\n        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,\n        "precision": float(precision_score(y_true, prediction, zero_division=0)),\n        "f1": float(f1_score(y_true, prediction, zero_division=0)),\n        "mcc": float(matthews_corrcoef(y_true, prediction)),\n        "tn": int(tn),\n        "fp": int(fp),\n        "fn": int(fn),\n        "tp": int(tp),\n    }\n\n\ndef calibration_bins(y_true: np.ndarray, probability: np.ndarray, n_bins: int = 10) -> pd.DataFrame:\n    bins = np.linspace(0, 1, n_bins + 1)\n    ids = np.clip(np.digitize(probability, bins, right=True) - 1, 0, n_bins - 1)\n    rows = []\n    for bin_id in range(n_bins):\n        mask = ids == bin_id\n        if not mask.any():\n            continue\n        rows.append(\n            {\n                "bin": bin_id,\n                "bin_left": float(bins[bin_id]),\n                "bin_right": float(bins[bin_id + 1]),\n                "n": int(mask.sum()),\n                "mean_predicted_probability": float(probability[mask].mean()),\n                "observed_positive_fraction": float(y_true[mask].mean()),\n            }\n        )\n    return pd.DataFrame(rows)\n\n\ndef plot_calibration(calibration: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:\n    sns.set_theme(style="whitegrid", context="talk")\n    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=220)\n    for split_name, group in calibration.groupby("split"):\n        row = metrics[(metrics["split"] == split_name) & (metrics["seed"] == 42)].iloc[0]\n        label = f"{split_name} (Brier={row[\'brier_score\']:.3f})"\n        axes[0].plot(\n            group["mean_predicted_probability"],\n            group["observed_positive_fraction"],\n            marker="o",\n            linewidth=2,\n            label=label,\n        )\n    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)\n    axes[0].set_xlabel("Mean predicted probability")\n    axes[0].set_ylabel("Observed positive fraction")\n    axes[0].set_title("Calibration curve")\n    axes[0].legend(fontsize=9)\n\n    sns.barplot(data=metrics[metrics["seed"] == 42], x="split", y="brier_score", ax=axes[1], palette="Set2")\n    axes[1].set_xlabel("")\n    axes[1].set_ylabel("Brier score")\n    axes[1].set_title("Lower is better")\n    plt.tight_layout()\n    plt.savefig(output / "baseline_calibration_curves.png")\n    plt.close()\n\n\ndef plot_robustness(repeated: pd.DataFrame, output: Path) -> None:\n    sns.set_theme(style="whitegrid", context="talk")\n    metrics = ["roc_auc", "pr_auc", "mcc", "brier_score"]\n    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=220)\n    for ax, metric in zip(axes.ravel(), metrics):\n        sns.stripplot(data=repeated, x="split", y=metric, hue="split", ax=ax, jitter=0.12, size=8, palette="Set2", legend=False)\n        grouped = repeated.groupby("split")[metric].agg(["mean", "std"]).reindex(["random", "scaffold"])\n        ax.errorbar(\n            np.arange(len(grouped)),\n            grouped["mean"],\n            yerr=grouped["std"].fillna(0),\n            fmt="_",\n            color="black",\n            markersize=28,\n            linewidth=2,\n            capsize=6,\n        )\n        ax.set_xlabel("")\n        ax.set_title(metric.replace("_", " ").upper())\n    plt.tight_layout()\n    plt.savefig(output / "baseline_repeated_split_robustness.png")\n    plt.close()\n\n\ndef max_similarities(\n    query_fps: list[DataStructs.ExplicitBitVect],\n    train_fps: list[DataStructs.ExplicitBitVect],\n    train_y: np.ndarray,\n) -> tuple[np.ndarray, np.ndarray, np.ndarray]:\n    pos_mask = train_y == 1\n    neg_mask = train_y == 0\n    max_all, max_pos, max_neg = [], [], []\n    for fp in query_fps:\n        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps), dtype=np.float32)\n        max_all.append(float(sims.max()) if sims.size else 0.0)\n        max_pos.append(float(sims[pos_mask].max()) if pos_mask.any() else 0.0)\n        max_neg.append(float(sims[neg_mask].max()) if neg_mask.any() else 0.0)\n    return np.asarray(max_all), np.asarray(max_pos), np.asarray(max_neg)\n\n\ndef domain_label(similarity: float, in_threshold: float, borderline_threshold: float) -> str:\n    if similarity >= in_threshold:\n        return "in-domain"\n    if similarity >= borderline_threshold:\n        return "borderline"\n    return "out-of-domain"\n\n\ndef applicability_domain_table(\n    df: pd.DataFrame,\n    fps: list[DataStructs.ExplicitBitVect],\n    y: np.ndarray,\n    result: SplitResult,\n    in_threshold: float,\n    borderline_threshold: float,\n) -> pd.DataFrame:\n    train_fps = [fps[i] for i in result.train_idx]\n    test_fps = [fps[i] for i in result.test_idx]\n    train_y = y[result.train_idx]\n    max_all, max_pos, max_neg = max_similarities(test_fps, train_fps, train_y)\n    source = df.iloc[result.test_idx].reset_index(drop=True)\n    table = pd.DataFrame(\n        {\n            "dataset_index": result.test_idx,\n            "source": source.get("source", pd.Series([""] * len(result.test_idx))).to_numpy(),\n            "compound_id": source.get("compound_id", pd.Series([""] * len(result.test_idx))).to_numpy(),\n            "canonical_smiles": source["canonical_smiles"].to_numpy(),\n            "label": y[result.test_idx],\n            "probability": result.probability,\n            "prediction": (result.probability >= 0.5).astype(int),\n            "max_tanimoto_train": max_all,\n            "max_tanimoto_positive_train": max_pos,\n            "max_tanimoto_negative_train": max_neg,\n        }\n    )\n    table["domain"] = table["max_tanimoto_train"].map(lambda value: domain_label(value, in_threshold, borderline_threshold))\n    table["correct"] = table["label"] == table["prediction"]\n    return table\n\n\ndef summarize_domain(ad: pd.DataFrame) -> pd.DataFrame:\n    rows = []\n    for domain, group in ad.groupby("domain", sort=False):\n        metrics = evaluate(group["label"].to_numpy(), group["probability"].to_numpy()) if len(group["label"].unique()) == 2 else {}\n        rows.append(\n            {\n                "domain": domain,\n                "n": int(len(group)),\n                "positive_rate": float(group["label"].mean()),\n                "mean_max_tanimoto_train": float(group["max_tanimoto_train"].mean()),\n                "accuracy": float((group["label"] == group["prediction"]).mean()),\n                "roc_auc": metrics.get("roc_auc", np.nan),\n                "pr_auc": metrics.get("pr_auc", np.nan),\n                "brier_score": metrics.get("brier_score", np.nan),\n                "mcc": metrics.get("mcc", np.nan),\n            }\n        )\n    order = {"in-domain": 0, "borderline": 1, "out-of-domain": 2}\n    return pd.DataFrame(rows).sort_values("domain", key=lambda s: s.map(order)).reset_index(drop=True)\n\n\ndef plot_applicability(ad: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:\n    sns.set_theme(style="whitegrid", context="talk")\n    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=220)\n    sns.histplot(data=ad, x="max_tanimoto_train", hue="label", bins=30, stat="count", common_norm=False, ax=axes[0])\n    axes[0].set_xlabel("Max Tanimoto similarity to training set")\n    axes[0].set_title("Primary scaffold test applicability")\n    sns.barplot(data=summary, x="domain", y="accuracy", ax=axes[1], palette="Set2")\n    axes[1].set_ylim(0, 1)\n    axes[1].set_xlabel("")\n    axes[1].set_ylabel("Accuracy")\n    axes[1].set_title("Performance by domain")\n    plt.tight_layout()\n    plt.savefig(output / "baseline_applicability_domain.png")\n    plt.close()\n\n\ndef representative_bit_substructure(mols: list[Chem.Mol], smiles: pd.Series, bit: int, radius: int, n_bits: int) -> dict[str, object]:\n    for idx, mol in enumerate(mols):\n        bit_info: dict[int, list[tuple[int, int]]] = {}\n        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bit_info)\n        if bit not in bit_info:\n            continue\n        atom_id, bit_radius = bit_info[bit][0]\n        if bit_radius == 0:\n            atom = mol.GetAtomWithIdx(atom_id)\n            return {\n                "bit": bit,\n                "representative_index": idx,\n                "representative_smiles": smiles.iloc[idx],\n                "bit_radius": bit_radius,\n                "representative_smarts": atom.GetSmarts(),\n                "highlight_atoms": [atom_id],\n                "highlight_bonds": [],\n            }\n        bond_ids = list(Chem.FindAtomEnvironmentOfRadiusN(mol, bit_radius, atom_id))\n        atom_ids: set[int] = {atom_id}\n        for bond_id in bond_ids:\n            bond = mol.GetBondWithIdx(bond_id)\n            atom_ids.add(bond.GetBeginAtomIdx())\n            atom_ids.add(bond.GetEndAtomIdx())\n        try:\n            smarts = Chem.MolFragmentToSmarts(mol, atomsToUse=sorted(atom_ids), bondsToUse=bond_ids)\n        except Exception:\n            smarts = ""\n        return {\n            "bit": bit,\n            "representative_index": idx,\n            "representative_smiles": smiles.iloc[idx],\n            "bit_radius": bit_radius,\n            "representative_smarts": smarts,\n            "highlight_atoms": sorted(atom_ids),\n            "highlight_bonds": bond_ids,\n        }\n    return {\n        "bit": bit,\n        "representative_index": None,\n        "representative_smiles": "",\n        "bit_radius": None,\n        "representative_smarts": "",\n        "highlight_atoms": [],\n        "highlight_bonds": [],\n    }\n\n\ndef bit_importance_outputs(\n    model: RandomForestClassifier,\n    mols: list[Chem.Mol],\n    smiles: pd.Series,\n    radius: int,\n    n_bits: int,\n    top_n: int,\n    output: Path,\n) -> pd.DataFrame:\n    importances = np.asarray(model.feature_importances_, dtype=float)\n    order = np.argsort(importances)[::-1][:top_n]\n    rows = []\n    draw_mols = []\n    legends = []\n    highlight_atoms = []\n    highlight_bonds = []\n    for rank, bit in enumerate(order, start=1):\n        rep = representative_bit_substructure(mols, smiles, int(bit), radius, n_bits)\n        rows.append(\n            {\n                "rank": rank,\n                "morgan_bit": int(bit),\n                "importance": float(importances[bit]),\n                "representative_index": rep["representative_index"],\n                "bit_radius": rep["bit_radius"],\n                "representative_smarts": rep["representative_smarts"],\n                "representative_smiles": rep["representative_smiles"],\n            }\n        )\n        if rep["representative_index"] is not None and len(draw_mols) < 12:\n            mol = mols[int(rep["representative_index"])]\n            draw_mols.append(mol)\n            legends.append(f"#{rank} bit {bit}\\nimp {importances[bit]:.4f}")\n            highlight_atoms.append(rep["highlight_atoms"])\n            highlight_bonds.append(rep["highlight_bonds"])\n    bit_df = pd.DataFrame(rows)\n    bit_df.to_csv(output / "baseline_rf_morgan_bit_importance.csv", index=False)\n    if draw_mols:\n        image = Draw.MolsToGridImage(\n            draw_mols,\n            molsPerRow=3,\n            subImgSize=(320, 240),\n            legends=legends,\n            highlightAtomLists=highlight_atoms,\n            highlightBondLists=highlight_bonds,\n            useSVG=False,\n        )\n        image.save(output / "baseline_rf_top_morgan_bits.png")\n    return bit_df\n\n\ndef train_evaluate_split(\n    split_name: str,\n    seed: int,\n    train_idx: np.ndarray,\n    test_idx: np.ndarray,\n    x: np.ndarray,\n    y: np.ndarray,\n    n_jobs: int,\n    trees: int,\n) -> SplitResult:\n    model = make_rf(seed=seed, n_jobs=n_jobs, trees=trees)\n    model.fit(x[train_idx], y[train_idx])\n    probability = predict_probability(model, x[test_idx])\n    metrics = {\n        "split": split_name,\n        "seed": seed,\n        "train_n": int(len(train_idx)),\n        "test_n": int(len(test_idx)),\n        "train_positive": int(y[train_idx].sum()),\n        "test_positive": int(y[test_idx].sum()),\n        **evaluate(y[test_idx], probability),\n    }\n    return SplitResult(split_name, seed, train_idx, test_idx, model, probability, metrics)\n\n\ndef metrics_summary(repeated: pd.DataFrame) -> pd.DataFrame:\n    metrics = ["roc_auc", "pr_auc", "balanced_accuracy", "mcc", "brier_score"]\n    summary = repeated.groupby("split")[metrics].agg(["mean", "std", "min", "max"])\n    summary.columns = ["_".join(col).strip() for col in summary.columns.to_flat_index()]\n    return summary.reset_index()\n\n\ndef read_training_context(training_dir: Path) -> dict[str, object]:\n    metadata_path = training_dir / "baseline_training_metadata.json"\n    performance_path = training_dir / "baseline_model_performance.csv"\n    context: dict[str, object] = {"training_dir": str(training_dir)}\n    if metadata_path.exists():\n        context["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))\n    if performance_path.exists():\n        perf = pd.read_csv(performance_path)\n        candidate = perf[(perf["feature_set"] == "morgan") & (perf["model"] == "random_forest")]\n        context["morgan_random_forest_training_rows"] = candidate.to_dict(orient="records")\n    return context\n\n\ndef write_reports(\n    output: Path,\n    dataset_path: Path,\n    training_context: dict[str, object],\n    df: pd.DataFrame,\n    primary_metrics: pd.DataFrame,\n    repeated: pd.DataFrame,\n    repeated_summary: pd.DataFrame,\n    ad_summary: pd.DataFrame,\n    bit_df: pd.DataFrame,\n    args: argparse.Namespace,\n) -> None:\n    generated = datetime.now().isoformat(timespec="seconds")\n    training_rows = training_context.get("morgan_random_forest_training_rows", [])\n    repeated_display = repeated[["split", "seed", "roc_auc", "pr_auc", "mcc", "brier_score", "balanced_accuracy"]]\n    primary_display = primary_metrics[["split", "roc_auc", "pr_auc", "mcc", "brier_score", "balanced_accuracy", "sensitivity_recall", "specificity"]]\n\n    lines = [\n        "# Phase 1B Baseline Validation Summary",\n        "",\n        f"Generated: {generated}",\n        "",\n        "## Scope",\n        "",\n        "- Candidate validated: `Morgan fingerprint + RandomForestClassifier`.",\n        f"- Dataset: `{dataset_path}`",\n        f"- Rows: {len(df):,}; positives: {int(df[\'label\'].sum()):,}; negatives: {int((df[\'label\'] == 0).sum()):,}.",\n        f"- Morgan fingerprint: radius {args.fingerprint_radius}, {args.fingerprint_bits} bits.",\n        f"- Random forest trees: {args.rf_trees}; min_samples_leaf=2; class_weight=`balanced_subsample`; max_features=`sqrt`.",\n        "- Negative-label caveat remains: ChEMBL examples are assumed-negative decoys.",\n        "",\n        "## Primary Validation",\n        "",\n        primary_display.to_markdown(index=False, floatfmt=".3f"),\n        "",\n        "## Repeated Split Robustness",\n        "",\n        repeated_summary.to_markdown(index=False, floatfmt=".3f"),\n        "",\n        "Detailed repeated split rows:",\n        "",\n        repeated_display.to_markdown(index=False, floatfmt=".3f"),\n        "",\n        "## Calibration",\n        "",\n        "- Brier score is included for both the random and scaffold primary evaluations; lower values indicate better probability calibration.",\n        "- Use `baseline_calibration_curves.png` to inspect whether predicted probabilities are over- or under-confident.",\n        "",\n        "## Applicability Domain",\n        "",\n        f"- In-domain: max Morgan Tanimoto similarity to the training set >= {args.in_domain_threshold:.2f}.",\n        f"- Borderline: >= {args.borderline_threshold:.2f} and < {args.in_domain_threshold:.2f}.",\n        f"- Out-of-domain: < {args.borderline_threshold:.2f}.",\n        "",\n        ad_summary.to_markdown(index=False, floatfmt=".3f"),\n        "",\n        "## Top Morgan Bits",\n        "",\n        bit_df.head(12)[["rank", "morgan_bit", "importance", "bit_radius", "representative_smarts"]].to_markdown(\n            index=False, floatfmt=".5f"\n        ),\n        "",\n        "## Generated Files",\n        "",\n        "- `baseline_validation_metrics.csv`",\n        "- `baseline_repeated_split_metrics.csv`",\n        "- `baseline_repeated_split_summary.csv`",\n        "- `baseline_calibration_bins.csv`",\n        "- `baseline_calibration_curves.png`",\n        "- `baseline_applicability_domain_primary_scaffold.csv`",\n        "- `baseline_applicability_domain_summary.csv`",\n        "- `baseline_applicability_domain.png`",\n        "- `baseline_rf_morgan_bit_importance.csv`",\n        "- `baseline_rf_top_morgan_bits.png`",\n        "- `primary_random_morgan_rf_model.joblib`",\n        "- `primary_scaffold_morgan_rf_model.joblib`",\n    ]\n    (output / "baseline_validation_summary.md").write_text("\\n".join(lines) + "\\n", encoding="utf-8")\n\n    model_lines = [\n        "# Phase 1B Model Selection Report",\n        "",\n        f"Generated: {generated}",\n        "",\n        "## Decision",\n        "",\n        "Retain `Morgan fingerprint + Random Forest` as the controlled Phase 1 classical baseline.",\n        "",\n        "## Evidence From Phase 1 Training",\n        "",\n    ]\n    if training_rows:\n        train_df = pd.DataFrame(training_rows)\n        model_lines.extend(\n            [\n                train_df[["split", "roc_auc", "pr_auc", "mcc", "balanced_accuracy"]].to_markdown(\n                    index=False, floatfmt=".3f"\n                ),\n                "",\n            ]\n        )\n    else:\n        model_lines.extend(["- Existing training rows were not found in the expected performance CSV.", ""])\n    scaffold_primary = primary_metrics[primary_metrics["split"] == "scaffold"].iloc[0]\n    random_primary = primary_metrics[primary_metrics["split"] == "random"].iloc[0]\n    scaffold_repeat = repeated_summary[repeated_summary["split"] == "scaffold"].iloc[0]\n    random_repeat = repeated_summary[repeated_summary["split"] == "random"].iloc[0]\n    model_lines.extend(\n        [\n            "## Phase 1B Validation Evidence",\n            "",\n            f"- Primary random split: ROC-AUC {random_primary[\'roc_auc\']:.3f}, PR-AUC {random_primary[\'pr_auc\']:.3f}, MCC {random_primary[\'mcc\']:.3f}, Brier {random_primary[\'brier_score\']:.3f}.",\n            f"- Primary scaffold split: ROC-AUC {scaffold_primary[\'roc_auc\']:.3f}, PR-AUC {scaffold_primary[\'pr_auc\']:.3f}, MCC {scaffold_primary[\'mcc\']:.3f}, Brier {scaffold_primary[\'brier_score\']:.3f}.",\n            f"- {len(args.seeds)}-seed random split ROC-AUC mean {random_repeat[\'roc_auc_mean\']:.3f} +/- {random_repeat[\'roc_auc_std\']:.3f}.",\n            f"- {len(args.seeds)}-seed scaffold split ROC-AUC mean {scaffold_repeat[\'roc_auc_mean\']:.3f} +/- {scaffold_repeat[\'roc_auc_std\']:.3f}.",\n            "",\n            "## Interpretation",\n            "",\n            "- Scaffold performance remains high, so the candidate is not only memorizing close random-split analogues.",\n            "- Repeated splits show whether the result is sensitive to seed choice; inspect the CSV for any weak seeds before publication.",\n            "- Probability outputs are usable for ranking only with the reported calibration caveat. If absolute probability thresholds become important, calibrate the model with a held-out calibration set.",\n            "- Applicability-domain labels should accompany any future Nipah G4 screening output, especially for compounds below the borderline threshold.",\n            "- Top Morgan bit mappings are representative hashed environments, not unique medicinal-chemistry explanations; collisions are possible with hashed fingerprints.",\n            "",\n            "## Recommended Next Use",\n            "",\n            "Use this model as the Phase 1 screening baseline and report scores with scaffold-split metrics, Brier score, and applicability-domain tags. Do not describe predictions as confirmed Nipah G4 binding activity without experimental validation.",\n        ]\n    )\n    (output / "model_selection_report.md").write_text("\\n".join(model_lines) + "\\n", encoding="utf-8")\n\n\ndef main() -> None:\n    warnings.filterwarnings("ignore", category=UserWarning)\n    warnings.filterwarnings("ignore", category=DeprecationWarning)\n    RDLogger.DisableLog("rdApp.warning")\n    args = parse_args()\n    project_root = Path.cwd()\n    dataset_path = resolve_path(args.dataset, project_root)\n    training_dir = resolve_path(args.training_dir, project_root)\n    output_dir = resolve_path(args.output_dir, project_root)\n    output_dir.mkdir(parents=True, exist_ok=True)\n\n    training_context = read_training_context(training_dir)\n    metadata = training_context.get("metadata", {})\n    if args.rf_trees is None:\n        args.rf_trees = int(metadata.get("args", {}).get("rf_trees", 250)) if isinstance(metadata, dict) else 250\n\n    print(f"Loading dataset: {dataset_path}", flush=True)\n    df = load_dataset(dataset_path)\n    y = df["label"].to_numpy(dtype=int)\n    print(f"Rows: {len(df):,}; positives: {int(y.sum()):,}; negatives: {int((y == 0).sum()):,}", flush=True)\n\n    print("Parsing molecules and computing Morgan fingerprints...", flush=True)\n    mols = mols_from_smiles(df["canonical_smiles"])\n    x = morgan_matrix(mols, args.fingerprint_radius, args.fingerprint_bits)\n    fps = morgan_bitvectors(mols, args.fingerprint_radius, args.fingerprint_bits)\n    scaffolds = murcko_scaffolds(mols, df["canonical_smiles"])\n\n    all_results: list[SplitResult] = []\n    primary_results: list[SplitResult] = []\n    primary_seed = args.primary_seed\n    split_jobs: list[tuple[str, int, np.ndarray, np.ndarray]] = []\n\n    random_train, random_test = train_test_split(\n        np.arange(len(y)), test_size=args.test_size, stratify=y, random_state=primary_seed\n    )\n    scaffold_train, scaffold_test = scaffold_split_indices(scaffolds, y, args.test_size, primary_seed)\n    split_jobs.extend(\n        [\n            ("random", primary_seed, np.array(sorted(random_train)), np.array(sorted(random_test))),\n            ("scaffold", primary_seed, scaffold_train, scaffold_test),\n        ]\n    )\n    for seed in args.seeds:\n        random_train, random_test = train_test_split(\n            np.arange(len(y)), test_size=args.test_size, stratify=y, random_state=seed\n        )\n        scaffold_train, scaffold_test = scaffold_split_indices(scaffolds, y, args.test_size, seed)\n        split_jobs.extend(\n            [\n                ("random", seed, np.array(sorted(random_train)), np.array(sorted(random_test))),\n                ("scaffold", seed, scaffold_train, scaffold_test),\n            ]\n        )\n\n    seen: set[tuple[str, int]] = set()\n    for split_name, seed, train_idx, test_idx in split_jobs:\n        if (split_name, seed) in seen:\n            continue\n        seen.add((split_name, seed))\n        print(f"Training Morgan+RF split={split_name} seed={seed} train={len(train_idx):,} test={len(test_idx):,}", flush=True)\n        result = train_evaluate_split(split_name, seed, train_idx, test_idx, x, y, args.n_jobs, args.rf_trees)\n        all_results.append(result)\n        if seed == primary_seed:\n            primary_results.append(result)\n\n    metrics = pd.DataFrame([result.metrics for result in all_results]).sort_values(["split", "seed"])\n    primary_metrics = metrics[metrics["seed"] == primary_seed].sort_values("split").reset_index(drop=True)\n    repeated = metrics[metrics["seed"].isin(args.seeds)].sort_values(["split", "seed"]).reset_index(drop=True)\n    repeated_summary = metrics_summary(repeated)\n\n    primary_metrics.to_csv(output_dir / "baseline_validation_metrics.csv", index=False)\n    repeated.to_csv(output_dir / "baseline_repeated_split_metrics.csv", index=False)\n    repeated_summary.to_csv(output_dir / "baseline_repeated_split_summary.csv", index=False)\n\n    calibration_rows = []\n    for result in primary_results:\n        y_test = y[result.test_idx]\n        cal = calibration_bins(y_test, result.probability)\n        cal.insert(0, "seed", result.seed)\n        cal.insert(0, "split", result.split)\n        calibration_rows.append(cal)\n        joblib.dump(result.model, output_dir / f"primary_{result.split}_morgan_rf_model.joblib")\n    calibration = pd.concat(calibration_rows, ignore_index=True)\n    calibration.to_csv(output_dir / "baseline_calibration_bins.csv", index=False)\n    print("Plotting calibration and repeated-split robustness...", flush=True)\n    plot_calibration(calibration, metrics, output_dir)\n    plot_robustness(repeated, output_dir)\n\n    print("Computing applicability-domain similarities for the primary scaffold split...", flush=True)\n    scaffold_primary = next(result for result in primary_results if result.split == "scaffold")\n    ad = applicability_domain_table(\n        df,\n        fps,\n        y,\n        scaffold_primary,\n        args.in_domain_threshold,\n        args.borderline_threshold,\n    )\n    ad_summary = summarize_domain(ad)\n    ad.to_csv(output_dir / "baseline_applicability_domain_primary_scaffold.csv", index=False)\n    ad_summary.to_csv(output_dir / "baseline_applicability_domain_summary.csv", index=False)\n    plot_applicability(ad, ad_summary, output_dir)\n\n    print("Mapping top Morgan fingerprint bits to representative environments...", flush=True)\n    bit_df = bit_importance_outputs(\n        scaffold_primary.model,\n        mols,\n        df["canonical_smiles"],\n        args.fingerprint_radius,\n        args.fingerprint_bits,\n        args.top_bits,\n        output_dir,\n    )\n\n    metadata_out = {\n        "generated": datetime.now().isoformat(timespec="seconds"),\n        "dataset": str(dataset_path),\n        "training_dir": str(training_dir),\n        "output_dir": str(output_dir),\n        "args": vars(args),\n        "training_context": training_context,\n    }\n    (output_dir / "baseline_validation_metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")\n\n    write_reports(\n        output_dir,\n        dataset_path,\n        training_context,\n        df,\n        primary_metrics,\n        repeated,\n        repeated_summary,\n        ad_summary,\n        bit_df,\n        args,\n    )\n\n    print("\\nPrimary validation:", flush=True)\n    print(primary_metrics[["split", "roc_auc", "pr_auc", "mcc", "brier_score", "balanced_accuracy"]].to_string(index=False))\n    print("\\nRepeated split summary:", flush=True)\n    print(repeated_summary.to_string(index=False))\n    print(f"\\nOutputs: {output_dir}")\n\n\nif __name__ == "__main__":\n    main()\n'


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

    # MODIFIED (Kai, 2026-06-09): forward --candidates so the embedded source
    # can validate multiple models in one run.
    if getattr(args, 'candidates', None):
        forwarded += ['--candidates', *args.candidates]
        notes.append(f'candidates={" ".join(args.candidates)}')

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
    parser.add_argument('--candidates', nargs='*', default=None, help='Space-separated list of "<feature_set>:<model>" candidates to forward to the embedded source. For step 07: descriptor/morgan/combined × random_forest/logistic_regression/lgbm/xgboost. Example: --candidates morgan:random_forest combined:random_forest')
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
                    # MODIFIED (Kai, 2026-06-09): prefer a real file at virtual_source_path
                    # so the embedded source can be edited and version-controlled like a normal
                    # script. Fall back to the inline EMBEDDED_SOURCE string when the file
                    # is not present (e.g. on a fresh checkout that only has the wrapper).
                    if virtual_source_path.is_file():
                        print(f"[RUNNING] loading embedded source from file: {virtual_source_path}", flush=True)
                        exec(compile(virtual_source_path.read_text(encoding='utf-8'), str(virtual_source_path), 'exec'), namespace)
                    else:
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
