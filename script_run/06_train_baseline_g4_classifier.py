#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Train classical ML baseline models."""

STEP_ID = '06'
STEP_TITLE = 'Train classical ML baseline models'
EMBEDDED_SOURCE_NAME = '07_train_baseline_g4_classifier.py'
SEED_MODE = 'repeated'  # MODIFIED (Kai, 2026-06-09): enable repeated-seed mode for n=20 replicates
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Train Phase 1 classical baseline models for the G4 ligand classifier."""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport warnings\nfrom dataclasses import dataclass\nfrom datetime import datetime\nfrom pathlib import Path\n\nimport joblib\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\nfrom rdkit import Chem, DataStructs\nfrom rdkit.Chem import rdFingerprintGenerator\nfrom rdkit.Chem.Scaffolds import MurckoScaffold\nfrom sklearn.base import clone\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import (\n    accuracy_score,\n    average_precision_score,\n    balanced_accuracy_score,\n    confusion_matrix,\n    f1_score,\n    matthews_corrcoef,\n    precision_score,\n    precision_recall_curve,\n    recall_score,\n    roc_auc_score,\n    roc_curve,\n)\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\n\ntry:\n    from lightgbm import LGBMClassifier\nexcept ImportError:  # pragma: no cover\n    LGBMClassifier = None\n\ntry:\n    from xgboost import XGBClassifier\nexcept ImportError:  # pragma: no cover\n    XGBClassifier = None\n\n\nDESCRIPTOR_COLUMNS = ["rdkit_mw", "rdkit_alogp", "rdkit_hbd", "rdkit_hba", "rdkit_tpsa"]\nDESCRIPTOR_LABELS = {\n    "rdkit_mw": "Molecular weight",\n    "rdkit_alogp": "AlogP",\n    "rdkit_hbd": "HBD",\n    "rdkit_hba": "HBA",\n    "rdkit_tpsa": "TPSA",\n}\n\n\n@dataclass\nclass FeatureSet:\n    name: str\n    matrix: np.ndarray\n    feature_names: list[str]\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument(\n        "--dataset",\n        default="data/modeling/g4ldb_chembl_binary_dataset.csv",\n        help="Binary G4LDB-positive vs ChEMBL assumed-negative dataset.",\n    )\n    parser.add_argument(\n        "--output-dir",\n        default="data/modeling/baseline_training",\n        help="Directory for training outputs.",\n    )\n    parser.add_argument("--seed", type=int, default=42, help="Random seed.")\n    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction.")\n    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for supported models.")\n    parser.add_argument("--rf-trees", type=int, default=300, help="Random forest tree count.")\n    parser.add_argument("--boost-rounds", type=int, default=300, help="Boosting estimator count.")\n    parser.add_argument("--fingerprint-bits", type=int, default=2048, help="Morgan fingerprint size.")\n    parser.add_argument("--fingerprint-radius", type=int, default=2, help="Morgan fingerprint radius.")\n    return parser.parse_args()\n\n\ndef resolve_path(path: str | Path, root: Path) -> Path:\n    path = Path(path)\n    if path.is_absolute():\n        return path\n    return root / path\n\n\ndef load_dataset(path: Path) -> pd.DataFrame:\n    df = pd.read_csv(path)\n    required = {"canonical_smiles", "label", *DESCRIPTOR_COLUMNS}\n    missing = sorted(required - set(df.columns))\n    if missing:\n        raise SystemExit(f"Missing required column(s): {\', \'.join(missing)}")\n\n    df = df.copy()\n    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)\n    for column in DESCRIPTOR_COLUMNS:\n        df[column] = pd.to_numeric(df[column], errors="coerce")\n    before = len(df)\n    df = df.dropna(subset=["canonical_smiles", "label", *DESCRIPTOR_COLUMNS]).reset_index(drop=True)\n    if len(df) != before:\n        print(f"Dropped {before - len(df)} rows with missing training fields.")\n    if not set(df["label"].unique()).issubset({0, 1}):\n        raise SystemExit("Labels must be binary 0/1.")\n    return df\n\n\ndef mols_from_smiles(smiles: pd.Series) -> list[Chem.Mol]:\n    mols: list[Chem.Mol] = []\n    bad = 0\n    for value in smiles:\n        mol = Chem.MolFromSmiles(str(value))\n        if mol is None:\n            bad += 1\n            mol = Chem.MolFromSmiles("")\n        mols.append(mol)\n    if bad:\n        raise SystemExit(f"Could not parse {bad} canonical SMILES.")\n    return mols\n\n\ndef morgan_fingerprints(mols: list[Chem.Mol], radius: int, n_bits: int) -> np.ndarray:\n    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)\n    matrix = np.zeros((len(mols), n_bits), dtype=np.float32)\n    for i, mol in enumerate(mols):\n        fp = generator.GetFingerprint(mol)\n        arr = np.zeros((n_bits,), dtype=np.int8)\n        DataStructs.ConvertToNumpyArray(fp, arr)\n        matrix[i, :] = arr\n    return matrix\n\n\ndef murcko_scaffolds(mols: list[Chem.Mol], smiles: pd.Series) -> list[str]:\n    scaffolds = []\n    for mol, smi in zip(mols, smiles):\n        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)\n        if not scaffold:\n            scaffold = f"acyclic::{smi}"\n        scaffolds.append(scaffold)\n    return scaffolds\n\n\ndef scaffold_split_indices(\n    scaffolds: list[str], y: np.ndarray, test_size: float, seed: int\n) -> tuple[np.ndarray, np.ndarray]:\n    rng = np.random.default_rng(seed)\n    groups: dict[str, list[int]] = {}\n    for index, scaffold in enumerate(scaffolds):\n        groups.setdefault(scaffold, []).append(index)\n\n    records = []\n    for scaffold, indices in groups.items():\n        idx = np.array(indices, dtype=int)\n        records.append(\n            {\n                "scaffold": scaffold,\n                "indices": idx,\n                "n": len(idx),\n                "pos": int(y[idx].sum()),\n                "shuffle": rng.random(),\n            }\n        )\n    records.sort(key=lambda item: (-item["n"], item["shuffle"]))\n\n    target_n = int(round(len(y) * test_size))\n    target_pos = y.mean()\n    test_indices: list[int] = []\n    test_pos = 0\n\n    for item in records:\n        if not test_indices:\n            take = True\n        else:\n            current_n = len(test_indices)\n            current_score = abs(current_n - target_n) / len(y)\n            current_score += abs((test_pos / max(current_n, 1)) - target_pos)\n            new_n = current_n + item["n"]\n            new_pos = test_pos + item["pos"]\n            new_score = abs(new_n - target_n) / len(y)\n            new_score += abs((new_pos / max(new_n, 1)) - target_pos)\n            take = new_score <= current_score or current_n < target_n * 0.75\n        if take:\n            test_indices.extend(item["indices"].tolist())\n            test_pos += item["pos"]\n        if len(test_indices) >= target_n:\n            break\n\n    test = np.array(sorted(test_indices), dtype=int)\n    train = np.array(sorted(set(range(len(y))) - set(test.tolist())), dtype=int)\n    if len(np.unique(y[test])) < 2:\n        raise SystemExit("Scaffold split test set has only one class; adjust seed or split strategy.")\n    return train, test\n\n\ndef split_summary(name: str, train_idx: np.ndarray, test_idx: np.ndarray, y: np.ndarray) -> dict[str, int | float | str]:\n    return {\n        "split": name,\n        "train_n": int(len(train_idx)),\n        "test_n": int(len(test_idx)),\n        "train_positive": int(y[train_idx].sum()),\n        "train_negative": int(len(train_idx) - y[train_idx].sum()),\n        "test_positive": int(y[test_idx].sum()),\n        "test_negative": int(len(test_idx) - y[test_idx].sum()),\n        "test_positive_rate": float(y[test_idx].mean()),\n    }\n\n\ndef make_feature_sets(df: pd.DataFrame, mols: list[Chem.Mol], args: argparse.Namespace) -> list[FeatureSet]:\n    descriptor_matrix = df[DESCRIPTOR_COLUMNS].to_numpy(dtype=np.float32)\n    fp_matrix = morgan_fingerprints(mols, args.fingerprint_radius, args.fingerprint_bits)\n    combined = np.hstack([fp_matrix, descriptor_matrix]).astype(np.float32)\n    return [\n        FeatureSet("descriptor", descriptor_matrix, DESCRIPTOR_COLUMNS),\n        FeatureSet("morgan", fp_matrix, [f"morgan_bit_{i}" for i in range(args.fingerprint_bits)]),\n        FeatureSet(\n            "combined",\n            combined,\n            [f"morgan_bit_{i}" for i in range(args.fingerprint_bits)] + DESCRIPTOR_COLUMNS,\n        ),\n    ]\n\n\ndef make_models(seed: int, n_jobs: int, rf_trees: int, boost_rounds: int) -> dict[str, object]:\n    models: dict[str, object] = {\n        "logistic_regression": Pipeline(\n            [\n                ("scaler", StandardScaler()),\n                (\n                    "model",\n                    LogisticRegression(\n                        max_iter=3000,\n                        class_weight="balanced",\n                        solver="lbfgs",\n                        random_state=seed,\n                    ),\n                ),\n            ]\n        ),\n        "random_forest": RandomForestClassifier(\n            n_estimators=rf_trees,\n            random_state=seed,\n            n_jobs=n_jobs,\n            class_weight="balanced_subsample",\n            min_samples_leaf=2,\n            max_features="sqrt",\n        ),\n    }\n    if LGBMClassifier is not None:\n        models["lightgbm"] = LGBMClassifier(\n            n_estimators=boost_rounds,\n            learning_rate=0.03,\n            num_leaves=31,\n            subsample=0.9,\n            colsample_bytree=0.8,\n            random_state=seed,\n            n_jobs=n_jobs,\n            objective="binary",\n            verbose=-1,\n        )\n    if XGBClassifier is not None:\n        models["xgboost"] = XGBClassifier(\n            n_estimators=boost_rounds,\n            learning_rate=0.03,\n            max_depth=5,\n            subsample=0.9,\n            colsample_bytree=0.8,\n            objective="binary:logistic",\n            eval_metric="logloss",\n            tree_method="hist",\n            random_state=seed,\n            n_jobs=n_jobs,\n        )\n    return models\n\n\ndef predict_probability(model: object, x_test: np.ndarray) -> np.ndarray:\n    if hasattr(model, "predict_proba"):\n        return model.predict_proba(x_test)[:, 1]\n    if hasattr(model, "decision_function"):\n        scores = model.decision_function(x_test)\n        return 1 / (1 + np.exp(-scores))\n    raise TypeError(f"Model lacks probability output: {type(model)!r}")\n\n\ndef evaluate(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:\n    prediction = (probability >= 0.5).astype(int)\n    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()\n    return {\n        "roc_auc": float(roc_auc_score(y_true, probability)),\n        "pr_auc": float(average_precision_score(y_true, probability)),\n        "accuracy": float(accuracy_score(y_true, prediction)),\n        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),\n        "sensitivity_recall": float(recall_score(y_true, prediction, zero_division=0)),\n        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,\n        "precision": float(precision_score(y_true, prediction, zero_division=0)),\n        "f1": float(f1_score(y_true, prediction, zero_division=0)),\n        "mcc": float(matthews_corrcoef(y_true, prediction)),\n        "tn": int(tn),\n        "fp": int(fp),\n        "fn": int(fn),\n        "tp": int(tp),\n    }\n\n\ndef feature_importance(model: object, feature_names: list[str], top_n: int = 30) -> pd.DataFrame:\n    estimator = model\n    if isinstance(model, Pipeline):\n        estimator = model.named_steps["model"]\n    values = None\n    if hasattr(estimator, "feature_importances_"):\n        values = np.asarray(estimator.feature_importances_, dtype=float)\n    elif hasattr(estimator, "coef_"):\n        values = np.abs(np.asarray(estimator.coef_).ravel())\n    if values is None or len(values) != len(feature_names):\n        return pd.DataFrame()\n    order = np.argsort(values)[::-1][:top_n]\n    return pd.DataFrame(\n        {\n            "feature": [feature_names[i] for i in order],\n            "importance": values[order],\n            "feature_label": [DESCRIPTOR_LABELS.get(feature_names[i], feature_names[i]) for i in order],\n        }\n    )\n\n\ndef plot_metric_bars(performance: pd.DataFrame, output: Path) -> None:\n    sns.set_theme(style="whitegrid", context="talk")\n    for metric in ["roc_auc", "pr_auc", "mcc", "balanced_accuracy"]:\n        plt.figure(figsize=(13, 7), dpi=220)\n        plot_df = performance.copy()\n        plot_df["model_label"] = plot_df["feature_set"] + " / " + plot_df["model"]\n        sns.barplot(data=plot_df, y="model_label", x=metric, hue="split", palette="Set2")\n        plt.xlabel(metric.replace("_", " ").upper())\n        plt.ylabel("")\n        plt.xlim(0, 1 if metric != "mcc" else max(1, plot_df[metric].max() + 0.05))\n        plt.title(f"Baseline model comparison: {metric.replace(\'_\', \' \').upper()}")\n        plt.tight_layout()\n        plt.savefig(output / f"baseline_{metric}_comparison.png")\n        plt.close()\n\n\ndef plot_best_curves(curve_rows: dict[tuple[str, str, str], dict[str, np.ndarray]], performance: pd.DataFrame, output: Path) -> None:\n    best = (\n        performance.sort_values(["split", "roc_auc", "pr_auc", "mcc"], ascending=[True, False, False, False])\n        .groupby("split")\n        .head(2)\n    )\n    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=220)\n    for _, row in best.iterrows():\n        key = (row["split"], row["feature_set"], row["model"])\n        data = curve_rows[key]\n        label = f"{row[\'split\']} | {row[\'feature_set\']} | {row[\'model\']}"\n        axes[0].plot(data["fpr"], data["tpr"], label=f"{label} (AUC={row[\'roc_auc\']:.3f})")\n        axes[1].plot(data["recall"], data["precision"], label=f"{label} (AP={row[\'pr_auc\']:.3f})")\n    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)\n    axes[0].set_xlabel("False positive rate")\n    axes[0].set_ylabel("True positive rate")\n    axes[0].set_title("ROC curves")\n    axes[1].set_xlabel("Recall")\n    axes[1].set_ylabel("Precision")\n    axes[1].set_title("Precision-recall curves")\n    for ax in axes:\n        ax.legend(fontsize=8)\n        ax.grid(True, alpha=0.3)\n    plt.tight_layout()\n    plt.savefig(output / "best_baseline_roc_pr_curves.png")\n    plt.close()\n\n\ndef plot_confusions(performance: pd.DataFrame, output: Path) -> None:\n    best = performance.sort_values("roc_auc", ascending=False).groupby("split").head(1)\n    fig, axes = plt.subplots(1, len(best), figsize=(6 * len(best), 5), dpi=220)\n    if len(best) == 1:\n        axes = [axes]\n    for ax, (_, row) in zip(axes, best.iterrows()):\n        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)\n        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)\n        ax.set_xticklabels(["Pred 0", "Pred 1"])\n        ax.set_yticklabels(["True 0", "True 1"], rotation=0)\n        ax.set_title(f"{row[\'split\']}: {row[\'feature_set\']} / {row[\'model\']}")\n    plt.tight_layout()\n    plt.savefig(output / "best_baseline_confusion_matrices.png")\n    plt.close()\n\n\ndef write_report(\n    output: Path,\n    dataset_path: Path,\n    df: pd.DataFrame,\n    split_df: pd.DataFrame,\n    performance: pd.DataFrame,\n    importance: pd.DataFrame,\n    args: argparse.Namespace,\n) -> None:\n    best_random = performance[performance["split"] == "random"].sort_values("roc_auc", ascending=False).iloc[0]\n    best_scaffold = performance[performance["split"] == "scaffold"].sort_values("roc_auc", ascending=False).iloc[0]\n    descriptor_best = performance[performance["feature_set"] == "descriptor"].sort_values("roc_auc", ascending=False).iloc[0]\n    top_lines = performance.sort_values("roc_auc", ascending=False).head(8)\n\n    lines = [\n        "# Phase 1 Classical Baseline Training Summary",\n        "",\n        f"Generated: {datetime.now().isoformat(timespec=\'seconds\')}",\n        "",\n        "## Dataset",\n        "",\n        f"- Dataset: `{dataset_path}`",\n        f"- Rows: {len(df):,}",\n        f"- Positive: {int(df[\'label\'].sum()):,}",\n        f"- Negative: {int((df[\'label\'] == 0).sum()):,}",\n        "- Framing: G4LDB-positive vs ChEMBL assumed-negative classifier.",\n        "- Caveat: ChEMBL negatives are assumed-negative, not experimentally verified non-G4 binders.",\n        "",\n        "## Training Setup",\n        "",\n        f"- Random seed: {args.seed}",\n        f"- Test size: {args.test_size}",\n        f"- Morgan fingerprint: radius {args.fingerprint_radius}, {args.fingerprint_bits} bits",\n        "- Feature sets: descriptor, Morgan fingerprint, combined.",\n        "- Models: logistic regression, random forest, LightGBM if available, XGBoost if available.",\n        "",\n        "## Split Summary",\n        "",\n        split_df.to_markdown(index=False),\n        "",\n        "## Top Models By ROC-AUC",\n        "",\n        top_lines[\n            [\n                "split",\n                "feature_set",\n                "model",\n                "roc_auc",\n                "pr_auc",\n                "balanced_accuracy",\n                "mcc",\n                "sensitivity_recall",\n                "specificity",\n            ]\n        ].to_markdown(index=False, floatfmt=".3f"),\n        "",\n        "## Key Results",\n        "",\n        f"- Best random split model: `{best_random[\'feature_set\']} / {best_random[\'model\']}` with ROC-AUC {best_random[\'roc_auc\']:.3f}, PR-AUC {best_random[\'pr_auc\']:.3f}, MCC {best_random[\'mcc\']:.3f}.",\n        f"- Best scaffold split model: `{best_scaffold[\'feature_set\']} / {best_scaffold[\'model\']}` with ROC-AUC {best_scaffold[\'roc_auc\']:.3f}, PR-AUC {best_scaffold[\'pr_auc\']:.3f}, MCC {best_scaffold[\'mcc\']:.3f}.",\n        f"- Best descriptor-only baseline: `{descriptor_best[\'model\']}` on `{descriptor_best[\'split\']}` split with ROC-AUC {descriptor_best[\'roc_auc\']:.3f}.",\n        "",\n        "## Initial Interpretation",\n        "",\n        "- Descriptor-only performance should be treated as a bias indicator. If it is high, simple physicochemical properties are carrying classification signal.",\n        "- Morgan and combined models should be judged more strongly on scaffold split than random split, because scaffold split better approximates generalization to new chemotypes.",\n        "- This Phase 1 run is intended for model triage and bias checking before downstream Nipah G4 compound prioritization.",\n    ]\n    if not importance.empty:\n        lines.extend(\n            [\n                "",\n                "## Top Feature Importance Rows",\n                "",\n                importance.head(20)[["split", "feature_set", "model", "feature_label", "importance"]].to_markdown(\n                    index=False, floatfmt=".5f"\n                ),\n            ]\n        )\n    lines.extend(\n        [\n            "",\n            "## Output Files",\n            "",\n            "- `baseline_model_performance.csv`",\n            "- `baseline_split_summary.csv`",\n            "- `baseline_feature_importance_top.csv`",\n            "- `baseline_training_metadata.json`",\n            "- `baseline_*_comparison.png`",\n            "- `best_baseline_roc_pr_curves.png`",\n            "- `best_baseline_confusion_matrices.png`",\n            "- `best_scaffold_model.joblib`",\n        ]\n    )\n    (output / "baseline_training_summary.md").write_text("\\n".join(lines) + "\\n", encoding="utf-8")\n\n\ndef main() -> None:\n    warnings.filterwarnings("ignore", category=UserWarning)\n    args = parse_args()\n    project_root = Path.cwd()\n    dataset_path = resolve_path(args.dataset, project_root)\n    output_dir = resolve_path(args.output_dir, project_root)\n    output_dir.mkdir(parents=True, exist_ok=True)\n\n    print(f"Loading dataset: {dataset_path}")\n    df = load_dataset(dataset_path)\n    y = df["label"].to_numpy(dtype=int)\n    print(f"Rows: {len(df):,}; positives: {y.sum():,}; negatives: {(y == 0).sum():,}")\n\n    print("Parsing molecules and generating features...")\n    mols = mols_from_smiles(df["canonical_smiles"])\n    scaffolds = murcko_scaffolds(mols, df["canonical_smiles"])\n    feature_sets = make_feature_sets(df, mols, args)\n\n    random_train, random_test = train_test_split(\n        np.arange(len(y)), test_size=args.test_size, stratify=y, random_state=args.seed\n    )\n    scaffold_train, scaffold_test = scaffold_split_indices(scaffolds, y, args.test_size, args.seed)\n    splits = {\n        "random": (np.array(sorted(random_train)), np.array(sorted(random_test))),\n        "scaffold": (scaffold_train, scaffold_test),\n    }\n    split_df = pd.DataFrame([split_summary(name, train, test, y) for name, (train, test) in splits.items()])\n    split_df.to_csv(output_dir / "baseline_split_summary.csv", index=False)\n\n    models = make_models(args.seed, args.n_jobs, args.rf_trees, args.boost_rounds)\n    rows = []\n    importance_rows = []\n    curve_rows: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}\n    fitted_models: dict[tuple[str, str, str], object] = {}\n\n    for split_name, (train_idx, test_idx) in splits.items():\n        print(f"\\nSplit: {split_name} train={len(train_idx):,} test={len(test_idx):,}")\n        for feature_set in feature_sets:\n            x_train = feature_set.matrix[train_idx]\n            x_test = feature_set.matrix[test_idx]\n            y_train = y[train_idx]\n            y_test = y[test_idx]\n            for model_name, template in models.items():\n                print(f"  Training {feature_set.name} / {model_name}...")\n                model = clone(template)\n                model.fit(x_train, y_train)\n                probability = predict_probability(model, x_test)\n                metrics = evaluate(y_test, probability)\n                rows.append(\n                    {\n                        "split": split_name,\n                        "feature_set": feature_set.name,\n                        "model": model_name,\n                        "train_n": int(len(train_idx)),\n                        "test_n": int(len(test_idx)),\n                        **metrics,\n                    }\n                )\n                fpr, tpr, _ = roc_curve(y_test, probability)\n                precision, recall, _ = precision_recall_curve(y_test, probability)\n                curve_rows[(split_name, feature_set.name, model_name)] = {\n                    "fpr": fpr,\n                    "tpr": tpr,\n                    "precision": precision,\n                    "recall": recall,\n                }\n                imp = feature_importance(model, feature_set.feature_names)\n                if not imp.empty:\n                    imp.insert(0, "model", model_name)\n                    imp.insert(0, "feature_set", feature_set.name)\n                    imp.insert(0, "split", split_name)\n                    importance_rows.append(imp)\n                fitted_models[(split_name, feature_set.name, model_name)] = model\n\n    performance = pd.DataFrame(rows).sort_values(["split", "roc_auc"], ascending=[True, False])\n    performance.to_csv(output_dir / "baseline_model_performance.csv", index=False)\n    importance = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame()\n    importance.to_csv(output_dir / "baseline_feature_importance_top.csv", index=False)\n\n    plot_metric_bars(performance, output_dir)\n    plot_best_curves(curve_rows, performance, output_dir)\n    plot_confusions(performance, output_dir)\n\n    best_scaffold = performance[performance["split"] == "scaffold"].sort_values(\n        ["roc_auc", "pr_auc", "mcc"], ascending=False\n    ).iloc[0]\n    best_key = (best_scaffold["split"], best_scaffold["feature_set"], best_scaffold["model"])\n    joblib.dump(fitted_models[best_key], output_dir / "best_scaffold_model.joblib")\n\n    metadata = {\n        "generated": datetime.now().isoformat(timespec="seconds"),\n        "dataset": str(dataset_path),\n        "rows": int(len(df)),\n        "positive": int(y.sum()),\n        "negative": int((y == 0).sum()),\n        "args": vars(args),\n        "best_scaffold_model": {\n            "split": best_scaffold["split"],\n            "feature_set": best_scaffold["feature_set"],\n            "model": best_scaffold["model"],\n            "roc_auc": float(best_scaffold["roc_auc"]),\n            "pr_auc": float(best_scaffold["pr_auc"]),\n            "mcc": float(best_scaffold["mcc"]),\n        },\n    }\n    (output_dir / "baseline_training_metadata.json").write_text(\n        json.dumps(metadata, indent=2), encoding="utf-8"\n    )\n    write_report(output_dir, dataset_path, df, split_df, performance, importance, args)\n    print("\\nDone.")\n    print(performance[["split", "feature_set", "model", "roc_auc", "pr_auc", "mcc"]].head(12).to_string(index=False))\n    print(f"\\nOutputs: {output_dir}")\n\n\nif __name__ == "__main__":\n    main()\n'


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
