#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Train Morgan-fingerprint MLP deep-learning baseline."""

STEP_ID = '08'
STEP_TITLE = 'Train Morgan-fingerprint MLP deep-learning baseline'
EMBEDDED_SOURCE_NAME = '09_train_mlp_morgan_g4_classifier.py'
SEED_MODE = 'repeated'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Train Phase 3A-1 MLP deep learning baseline on Morgan fingerprints."""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport random\nimport time\nfrom dataclasses import dataclass\nfrom datetime import datetime\nfrom pathlib import Path\n\nimport joblib\nimport matplotlib\n\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\nimport torch\nfrom rdkit import Chem, DataStructs, RDLogger\nfrom rdkit.Chem import rdFingerprintGenerator\nfrom rdkit.Chem.Scaffolds import MurckoScaffold\nfrom sklearn.metrics import (\n    accuracy_score,\n    average_precision_score,\n    balanced_accuracy_score,\n    brier_score_loss,\n    confusion_matrix,\n    f1_score,\n    matthews_corrcoef,\n    precision_recall_curve,\n    precision_score,\n    recall_score,\n    roc_auc_score,\n    roc_curve,\n)\nfrom sklearn.model_selection import train_test_split\nfrom torch import nn\nfrom torch.utils.data import DataLoader, TensorDataset\n\n\nDEFAULT_SEEDS = [0, 1, 2, 3, 4]\n\n\n@dataclass(frozen=True)\nclass SplitResult:\n    split: str\n    seed: int\n    train_idx: np.ndarray\n    val_idx: np.ndarray\n    test_idx: np.ndarray\n    probability: np.ndarray\n    metrics: dict[str, float | int | str]\n    history: pd.DataFrame\n    model_path: str\n\n\nclass MorganMLP(nn.Module):\n    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:\n        super().__init__()\n        layers: list[nn.Module] = []\n        previous = input_dim\n        for hidden in hidden_dims:\n            layers.extend(\n                [\n                    nn.Linear(previous, hidden),\n                    nn.BatchNorm1d(hidden),\n                    nn.ReLU(),\n                    nn.Dropout(dropout),\n                ]\n            )\n            previous = hidden\n        layers.append(nn.Linear(previous, 1))\n        self.network = nn.Sequential(*layers)\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        return self.network(x).squeeze(1)\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--dataset", default="data/modeling/g4ldb_chembl_binary_dataset.csv")\n    parser.add_argument("--output-dir", default="data/modeling/deep_learning_mlp")\n    parser.add_argument("--test-size", type=float, default=0.2)\n    parser.add_argument("--val-size", type=float, default=0.1)\n    parser.add_argument("--primary-seed", type=int, default=42)\n    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)\n    parser.add_argument("--fingerprint-bits", type=int, default=2048)\n    parser.add_argument("--fingerprint-radius", type=int, default=2)\n    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 256])\n    parser.add_argument("--dropout", type=float, default=0.25)\n    parser.add_argument("--batch-size", type=int, default=256)\n    parser.add_argument("--epochs", type=int, default=80)\n    parser.add_argument("--patience", type=int, default=12)\n    parser.add_argument("--learning-rate", type=float, default=1e-3)\n    parser.add_argument("--weight-decay", type=float, default=1e-4)\n    parser.add_argument("--num-workers", type=int, default=0)\n    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")\n    return parser.parse_args()\n\n\ndef resolve_path(path: str | Path, root: Path) -> Path:\n    path = Path(path)\n    return path if path.is_absolute() else root / path\n\n\ndef set_seed(seed: int) -> None:\n    random.seed(seed)\n    np.random.seed(seed)\n    torch.manual_seed(seed)\n    torch.cuda.manual_seed_all(seed)\n    torch.backends.cudnn.benchmark = False\n    torch.backends.cudnn.deterministic = True\n\n\ndef load_dataset(path: Path) -> pd.DataFrame:\n    df = pd.read_csv(path)\n    required = {"canonical_smiles", "label"}\n    missing = sorted(required - set(df.columns))\n    if missing:\n        raise SystemExit(f"Missing required column(s): {\', \'.join(missing)}")\n    df = df.copy()\n    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)\n    before = len(df)\n    df = df.dropna(subset=["canonical_smiles", "label"]).reset_index(drop=True)\n    if len(df) != before:\n        print(f"Dropped {before - len(df)} rows with missing required fields.")\n    if not set(df["label"].unique()).issubset({0, 1}):\n        raise SystemExit("Labels must be binary 0/1.")\n    return df\n\n\ndef mols_from_smiles(smiles: pd.Series) -> list[Chem.Mol]:\n    mols = []\n    bad = []\n    for i, value in enumerate(smiles):\n        mol = Chem.MolFromSmiles(str(value))\n        if mol is None:\n            bad.append(i)\n        else:\n            mols.append(mol)\n    if bad:\n        raise SystemExit(f"Could not parse {len(bad)} canonical SMILES; first bad row index: {bad[0]}")\n    return mols\n\n\ndef morgan_matrix(mols: list[Chem.Mol], radius: int, n_bits: int) -> np.ndarray:\n    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)\n    matrix = np.zeros((len(mols), n_bits), dtype=np.float32)\n    for i, mol in enumerate(mols):\n        arr = np.zeros((n_bits,), dtype=np.int8)\n        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), arr)\n        matrix[i, :] = arr\n    return matrix\n\n\ndef murcko_scaffolds(mols: list[Chem.Mol], smiles: pd.Series) -> list[str]:\n    scaffolds = []\n    for mol, smi in zip(mols, smiles):\n        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)\n        scaffolds.append(scaffold if scaffold else f"acyclic::{smi}")\n    return scaffolds\n\n\ndef scaffold_split_indices(scaffolds: list[str], y: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:\n    rng = np.random.default_rng(seed)\n    groups: dict[str, list[int]] = {}\n    for index, scaffold in enumerate(scaffolds):\n        groups.setdefault(scaffold, []).append(index)\n\n    records = []\n    for scaffold, indices in groups.items():\n        idx = np.array(indices, dtype=int)\n        records.append({"indices": idx, "n": len(idx), "pos": int(y[idx].sum()), "shuffle": rng.random()})\n    records.sort(key=lambda item: (-item["n"], item["shuffle"]))\n\n    target_n = int(round(len(y) * test_size))\n    target_pos = y.mean()\n    test_indices: list[int] = []\n    test_pos = 0\n    for item in records:\n        if not test_indices:\n            take = True\n        else:\n            current_n = len(test_indices)\n            current_score = abs(current_n - target_n) / len(y)\n            current_score += abs((test_pos / max(current_n, 1)) - target_pos)\n            new_n = current_n + item["n"]\n            new_pos = test_pos + item["pos"]\n            new_score = abs(new_n - target_n) / len(y)\n            new_score += abs((new_pos / max(new_n, 1)) - target_pos)\n            take = new_score <= current_score or current_n < target_n * 0.75\n        if take:\n            test_indices.extend(item["indices"].tolist())\n            test_pos += item["pos"]\n        if len(test_indices) >= target_n:\n            break\n\n    test = np.array(sorted(test_indices), dtype=int)\n    train = np.array(sorted(set(range(len(y))) - set(test.tolist())), dtype=int)\n    if len(np.unique(y[test])) < 2:\n        raise RuntimeError(f"Scaffold split seed {seed} has only one test class.")\n    return train, test\n\n\ndef validation_split(train_idx: np.ndarray, y: np.ndarray, val_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:\n    train_sub, val = train_test_split(\n        train_idx,\n        test_size=val_size,\n        random_state=seed,\n        stratify=y[train_idx],\n    )\n    return np.array(sorted(train_sub), dtype=int), np.array(sorted(val), dtype=int)\n\n\ndef make_loader(x: np.ndarray, y: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool, workers: int) -> DataLoader:\n    dataset = TensorDataset(\n        torch.from_numpy(x[indices]).float(),\n        torch.from_numpy(y[indices]).float(),\n    )\n    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available())\n\n\ndef evaluate(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:\n    prediction = (probability >= 0.5).astype(int)\n    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()\n    return {\n        "roc_auc": float(roc_auc_score(y_true, probability)),\n        "pr_auc": float(average_precision_score(y_true, probability)),\n        "brier_score": float(brier_score_loss(y_true, probability)),\n        "accuracy": float(accuracy_score(y_true, prediction)),\n        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),\n        "sensitivity_recall": float(recall_score(y_true, prediction, zero_division=0)),\n        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,\n        "precision": float(precision_score(y_true, prediction, zero_division=0)),\n        "f1": float(f1_score(y_true, prediction, zero_division=0)),\n        "mcc": float(matthews_corrcoef(y_true, prediction)),\n        "tn": int(tn),\n        "fp": int(fp),\n        "fn": int(fn),\n        "tp": int(tp),\n    }\n\n\n@torch.no_grad()\ndef predict(model: nn.Module, x: np.ndarray, indices: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:\n    model.eval()\n    probabilities = []\n    for start in range(0, len(indices), batch_size):\n        batch_idx = indices[start : start + batch_size]\n        xb = torch.from_numpy(x[batch_idx]).float().to(device)\n        logits = model(xb)\n        probabilities.append(torch.sigmoid(logits).cpu().numpy())\n    return np.concatenate(probabilities)\n\n\ndef train_one_split(\n    split: str,\n    seed: int,\n    train_idx: np.ndarray,\n    test_idx: np.ndarray,\n    x: np.ndarray,\n    y: np.ndarray,\n    args: argparse.Namespace,\n    device: torch.device,\n    output: Path,\n    save_model: bool,\n) -> SplitResult:\n    set_seed(seed)\n    train_sub_idx, val_idx = validation_split(train_idx, y, args.val_size, seed)\n    train_loader = make_loader(x, y, train_sub_idx, args.batch_size, True, args.num_workers)\n    val_loader = make_loader(x, y, val_idx, args.batch_size, False, args.num_workers)\n\n    model = MorganMLP(x.shape[1], args.hidden_dims, args.dropout).to(device)\n    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)\n    positives = float(y[train_sub_idx].sum())\n    negatives = float(len(train_sub_idx) - positives)\n    pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)\n    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)\n\n    best_state = None\n    best_val_auc = -np.inf\n    best_epoch = 0\n    stale_epochs = 0\n    rows = []\n\n    for epoch in range(1, args.epochs + 1):\n        model.train()\n        train_losses = []\n        for xb, yb in train_loader:\n            xb = xb.to(device, non_blocking=True)\n            yb = yb.to(device, non_blocking=True)\n            optimizer.zero_grad(set_to_none=True)\n            loss = criterion(model(xb), yb)\n            loss.backward()\n            optimizer.step()\n            train_losses.append(float(loss.detach().cpu()))\n\n        model.eval()\n        val_losses = []\n        with torch.no_grad():\n            for xb, yb in val_loader:\n                xb = xb.to(device, non_blocking=True)\n                yb = yb.to(device, non_blocking=True)\n                val_losses.append(float(criterion(model(xb), yb).detach().cpu()))\n        val_probability = predict(model, x, val_idx, device, args.batch_size)\n        val_auc = roc_auc_score(y[val_idx], val_probability)\n        val_pr_auc = average_precision_score(y[val_idx], val_probability)\n        rows.append(\n            {\n                "split": split,\n                "seed": seed,\n                "epoch": epoch,\n                "train_loss": float(np.mean(train_losses)),\n                "val_loss": float(np.mean(val_losses)),\n                "val_roc_auc": float(val_auc),\n                "val_pr_auc": float(val_pr_auc),\n            }\n        )\n        if val_auc > best_val_auc:\n            best_val_auc = val_auc\n            best_epoch = epoch\n            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}\n            stale_epochs = 0\n        else:\n            stale_epochs += 1\n        if stale_epochs >= args.patience:\n            break\n\n    if best_state is not None:\n        model.load_state_dict(best_state)\n\n    probability = predict(model, x, test_idx, device, args.batch_size)\n    metrics = {\n        "split": split,\n        "seed": seed,\n        "train_n": int(len(train_idx)),\n        "val_n": int(len(val_idx)),\n        "test_n": int(len(test_idx)),\n        "train_positive": int(y[train_idx].sum()),\n        "val_positive": int(y[val_idx].sum()),\n        "test_positive": int(y[test_idx].sum()),\n        "best_epoch": int(best_epoch),\n        "best_val_roc_auc": float(best_val_auc),\n        **evaluate(y[test_idx], probability),\n    }\n    model_path = ""\n    if save_model:\n        model_path = str(output / f"mlp_morgan_{split}_seed{seed}.pt")\n        torch.save(\n            {\n                "model_state_dict": model.state_dict(),\n                "input_dim": x.shape[1],\n                "hidden_dims": args.hidden_dims,\n                "dropout": args.dropout,\n                "seed": seed,\n                "split": split,\n                "metrics": metrics,\n            },\n            model_path,\n        )\n\n    return SplitResult(split, seed, train_idx, val_idx, test_idx, probability, metrics, pd.DataFrame(rows), model_path)\n\n\ndef calibration_bins(y_true: np.ndarray, probability: np.ndarray, split: str, seed: int, n_bins: int = 10) -> pd.DataFrame:\n    bins = np.linspace(0, 1, n_bins + 1)\n    ids = np.clip(np.digitize(probability, bins, right=True) - 1, 0, n_bins - 1)\n    rows = []\n    for bin_id in range(n_bins):\n        mask = ids == bin_id\n        if not mask.any():\n            continue\n        rows.append(\n            {\n                "split": split,\n                "seed": seed,\n                "bin": bin_id,\n                "bin_left": float(bins[bin_id]),\n                "bin_right": float(bins[bin_id + 1]),\n                "n": int(mask.sum()),\n                "mean_predicted_probability": float(probability[mask].mean()),\n                "observed_positive_fraction": float(y_true[mask].mean()),\n            }\n        )\n    return pd.DataFrame(rows)\n\n\ndef plot_learning_curves(history: pd.DataFrame, output: Path) -> None:\n    sns.set_theme(style="whitegrid", context="talk")\n    primary = history[history["seed"] == 42].copy()\n    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=220)\n    for split, group in primary.groupby("split"):\n        axes[0].plot(group["epoch"], group["train_loss"], label=f"{split} train")\n        axes[0].plot(group["epoch"], group["val_loss"], linestyle="--", label=f"{split} val")\n        axes[1].plot(group["epoch"], group["val_roc_auc"], label=f"{split} ROC-AUC")\n        axes[1].plot(group["epoch"], group["val_pr_auc"], linestyle="--", label=f"{split} PR-AUC")\n    axes[0].set_xlabel("Epoch")\n    axes[0].set_ylabel("BCE loss")\n    axes[0].set_title("Primary learning curves")\n    axes[1].set_xlabel("Epoch")\n    axes[1].set_ylabel("Validation score")\n    axes[1].set_title("Validation metrics")\n    for ax in axes:\n        ax.legend(fontsize=9)\n    plt.tight_layout()\n    plt.savefig(output / "mlp_morgan_learning_curves.png")\n    plt.close()\n\n\ndef plot_roc_pr(results: list[SplitResult], y: np.ndarray, output: Path) -> None:\n    sns.set_theme(style="whitegrid", context="talk")\n    primary = [result for result in results if result.seed == 42]\n    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=220)\n    for result in primary:\n        y_test = y[result.test_idx]\n        fpr, tpr, _ = roc_curve(y_test, result.probability)\n        precision, recall, _ = precision_recall_curve(y_test, result.probability)\n        axes[0].plot(fpr, tpr, linewidth=2, label=f"{result.split} AUC={result.metrics[\'roc_auc\']:.3f}")\n        axes[1].plot(recall, precision, linewidth=2, label=f"{result.split} AUC={result.metrics[\'pr_auc\']:.3f}")\n    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)\n    axes[0].set_xlabel("False positive rate")\n    axes[0].set_ylabel("True positive rate")\n    axes[0].set_title("Primary ROC curves")\n    axes[1].set_xlabel("Recall")\n    axes[1].set_ylabel("Precision")\n    axes[1].set_title("Primary PR curves")\n    for ax in axes:\n        ax.legend(fontsize=9)\n    plt.tight_layout()\n    plt.savefig(output / "mlp_morgan_primary_roc_pr_curves.png")\n    plt.close()\n\n\ndef plot_confusion(results: list[SplitResult], y: np.ndarray, output: Path) -> None:\n    primary = [result for result in results if result.seed == 42]\n    fig, axes = plt.subplots(1, len(primary), figsize=(7 * len(primary), 6), dpi=220)\n    if len(primary) == 1:\n        axes = [axes]\n    for ax, result in zip(axes, primary):\n        prediction = (result.probability >= 0.5).astype(int)\n        cm = confusion_matrix(y[result.test_idx], prediction, labels=[0, 1])\n        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])\n        ax.set_title(f"{result.split} confusion matrix")\n        ax.set_xlabel("Predicted")\n        ax.set_ylabel("True")\n    plt.tight_layout()\n    plt.savefig(output / "mlp_morgan_primary_confusion_matrices.png")\n    plt.close()\n\n\ndef plot_calibration(calibration: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:\n    sns.set_theme(style="whitegrid", context="talk")\n    primary_cal = calibration[calibration["seed"] == 42]\n    primary_metrics = metrics[metrics["seed"] == 42]\n    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=220)\n    for split, group in primary_cal.groupby("split"):\n        row = primary_metrics[primary_metrics["split"] == split].iloc[0]\n        axes[0].plot(\n            group["mean_predicted_probability"],\n            group["observed_positive_fraction"],\n            marker="o",\n            linewidth=2,\n            label=f"{split} (Brier={row[\'brier_score\']:.3f})",\n        )\n    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)\n    axes[0].set_xlabel("Mean predicted probability")\n    axes[0].set_ylabel("Observed positive fraction")\n    axes[0].set_title("Primary calibration")\n    axes[0].legend(fontsize=9)\n    sns.barplot(data=primary_metrics, x="split", y="brier_score", ax=axes[1], palette="Set2")\n    axes[1].set_xlabel("")\n    axes[1].set_ylabel("Brier score")\n    axes[1].set_title("Lower is better")\n    plt.tight_layout()\n    plt.savefig(output / "mlp_morgan_calibration_curves.png")\n    plt.close()\n\n\ndef plot_robustness(metrics: pd.DataFrame, output: Path, seeds: list[int]) -> None:\n    sns.set_theme(style="whitegrid", context="talk")\n    repeated = metrics[metrics["seed"].isin(seeds)]\n    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=220)\n    for ax, metric in zip(axes.ravel(), ["roc_auc", "pr_auc", "mcc", "brier_score"]):\n        sns.stripplot(data=repeated, x="split", y=metric, hue="split", jitter=0.12, size=8, palette="Set2", legend=False, ax=ax)\n        grouped = repeated.groupby("split")[metric].agg(["mean", "std"]).reindex(["random", "scaffold"])\n        ax.errorbar(np.arange(len(grouped)), grouped["mean"], yerr=grouped["std"].fillna(0), fmt="_", color="black", markersize=28, linewidth=2, capsize=6)\n        ax.set_xlabel("")\n        ax.set_title(metric.replace("_", " ").upper())\n    plt.tight_layout()\n    plt.savefig(output / "mlp_morgan_repeated_split_robustness.png")\n    plt.close()\n\n\ndef summarize_repeated(metrics: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:\n    repeated = metrics[metrics["seed"].isin(seeds)]\n    summary = repeated.groupby("split").agg(\n        roc_auc_mean=("roc_auc", "mean"),\n        roc_auc_std=("roc_auc", "std"),\n        pr_auc_mean=("pr_auc", "mean"),\n        pr_auc_std=("pr_auc", "std"),\n        balanced_accuracy_mean=("balanced_accuracy", "mean"),\n        balanced_accuracy_std=("balanced_accuracy", "std"),\n        mcc_mean=("mcc", "mean"),\n        mcc_std=("mcc", "std"),\n        brier_score_mean=("brier_score", "mean"),\n        brier_score_std=("brier_score", "std"),\n    )\n    return summary.reset_index()\n\n\ndef read_rf_context(root: Path) -> pd.DataFrame:\n    path = root / "data/modeling/baseline_validation/baseline_validation_metrics.csv"\n    if not path.exists():\n        return pd.DataFrame()\n    rf = pd.read_csv(path)\n    return rf[(rf["seed"] == 42) & (rf["split"].isin(["random", "scaffold"]))].copy()\n\n\ndef write_summary(\n    output: Path,\n    dataset: Path,\n    df: pd.DataFrame,\n    args: argparse.Namespace,\n    device: torch.device,\n    metrics: pd.DataFrame,\n    repeated_summary: pd.DataFrame,\n    rf_context: pd.DataFrame,\n    elapsed: float,\n) -> None:\n    primary = metrics[metrics["seed"] == 42].copy()\n    lines = [\n        "# Phase 3A-1 MLP Morgan Fingerprint Baseline",\n        "",\n        f"Generated: {datetime.now().isoformat(timespec=\'seconds\')}",\n        "",\n        "## Scope",\n        "",\n        "- Model: PyTorch MLP on Morgan fingerprints.",\n        f"- Dataset: `{dataset}`",\n        f"- Rows: {len(df):,}; positives: {int(df[\'label\'].sum()):,}; negatives: {int((1 - df[\'label\']).sum()):,}.",\n        f"- Morgan fingerprint: radius {args.fingerprint_radius}, {args.fingerprint_bits} bits.",\n        f"- Architecture: hidden dims {args.hidden_dims}, dropout {args.dropout}.",\n        f"- Device: `{device}`.",\n        f"- Runtime: {elapsed / 60:.1f} minutes.",\n        "- Negative-label caveat remains: ChEMBL examples are assumed-negative decoys.",\n        "",\n        "## Primary Results",\n        "",\n        primary[["split", "roc_auc", "pr_auc", "mcc", "brier_score", "balanced_accuracy", "sensitivity_recall", "specificity", "best_epoch"]].to_markdown(index=False, floatfmt=".3f"),\n        "",\n        "## Repeated Split Robustness",\n        "",\n        repeated_summary.to_markdown(index=False, floatfmt=".3f"),\n        "",\n    ]\n    if not rf_context.empty:\n        comparison = primary[["split", "roc_auc", "pr_auc", "mcc", "brier_score", "balanced_accuracy"]].merge(\n            rf_context[["split", "roc_auc", "pr_auc", "mcc", "brier_score", "balanced_accuracy"]],\n            on="split",\n            suffixes=("_mlp", "_rf"),\n        )\n        for metric in ["roc_auc", "pr_auc", "mcc", "brier_score", "balanced_accuracy"]:\n            comparison[f"delta_{metric}_mlp_minus_rf"] = comparison[f"{metric}_mlp"] - comparison[f"{metric}_rf"]\n        lines.extend(\n            [\n                "## Comparison With Phase 1B Random Forest",\n                "",\n                comparison.to_markdown(index=False, floatfmt=".4f"),\n                "",\n                "Interpretation: positive deltas favor the MLP except for Brier score, where lower is better.",\n                "",\n            ]\n        )\n    lines.extend(\n        [\n            "## Generated Files",\n            "",\n            "- `mlp_morgan_metrics.csv`",\n            "- `mlp_morgan_repeated_split_summary.csv`",\n            "- `mlp_morgan_training_history.csv`",\n            "- `mlp_morgan_calibration_bins.csv`",\n            "- `mlp_morgan_learning_curves.png`",\n            "- `mlp_morgan_primary_roc_pr_curves.png`",\n            "- `mlp_morgan_primary_confusion_matrices.png`",\n            "- `mlp_morgan_calibration_curves.png`",\n            "- `mlp_morgan_repeated_split_robustness.png`",\n            "- `mlp_morgan_random_seed42.pt`",\n            "- `mlp_morgan_scaffold_seed42.pt`",\n            "- `mlp_morgan_metadata.json`",\n            "",\n            "## Caution",\n            "",\n            "This MLP is a controlled deep learning baseline using the same Morgan representation as the classical Random Forest. If it does not clearly outperform the RF baseline, that is still useful: it suggests that gains may require representation learning from SMILES or molecular graphs rather than only swapping the classifier.",\n            "",\n        ]\n    )\n    (output / "mlp_morgan_summary.md").write_text("\\n".join(lines), encoding="utf-8")\n\n\ndef main() -> None:\n    args = parse_args()\n    root = Path(__file__).resolve().parents[1]\n    dataset = resolve_path(args.dataset, root)\n    output = resolve_path(args.output_dir, root)\n    output.mkdir(parents=True, exist_ok=True)\n    RDLogger.DisableLog("rdApp.warning")\n\n    start = time.time()\n    df = load_dataset(dataset)\n    y = df["label"].to_numpy(dtype=np.int64)\n    mols = mols_from_smiles(df["canonical_smiles"])\n    x = morgan_matrix(mols, args.fingerprint_radius, args.fingerprint_bits)\n    scaffolds = murcko_scaffolds(mols, df["canonical_smiles"])\n\n    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")\n    all_seeds = sorted(set([args.primary_seed, *args.seeds]))\n    results: list[SplitResult] = []\n\n    for seed in all_seeds:\n        random_train, random_test = train_test_split(\n            np.arange(len(y)),\n            test_size=args.test_size,\n            random_state=seed,\n            stratify=y,\n        )\n        scaffold_train, scaffold_test = scaffold_split_indices(scaffolds, y, args.test_size, seed)\n        for split, train_idx, test_idx in [\n            ("random", np.array(sorted(random_train), dtype=int), np.array(sorted(random_test), dtype=int)),\n            ("scaffold", scaffold_train, scaffold_test),\n        ]:\n            print(f"Training {split} split seed={seed} on {device}...")\n            save_model = seed == args.primary_seed\n            result = train_one_split(split, seed, train_idx, test_idx, x, y, args, device, output, save_model)\n            results.append(result)\n            print(\n                f"  {split} seed={seed}: ROC-AUC={result.metrics[\'roc_auc\']:.4f}, "\n                f"PR-AUC={result.metrics[\'pr_auc\']:.4f}, MCC={result.metrics[\'mcc\']:.4f}, "\n                f"best_epoch={result.metrics[\'best_epoch\']}"\n            )\n\n    metrics = pd.DataFrame([result.metrics | {"model_path": result.model_path} for result in results])\n    metrics.to_csv(output / "mlp_morgan_metrics.csv", index=False)\n    history = pd.concat([result.history for result in results], ignore_index=True)\n    history.to_csv(output / "mlp_morgan_training_history.csv", index=False)\n    repeated_summary = summarize_repeated(metrics, args.seeds)\n    repeated_summary.to_csv(output / "mlp_morgan_repeated_split_summary.csv", index=False)\n\n    calibration = pd.concat(\n        [calibration_bins(y[result.test_idx], result.probability, result.split, result.seed) for result in results],\n        ignore_index=True,\n    )\n    calibration.to_csv(output / "mlp_morgan_calibration_bins.csv", index=False)\n\n    plot_learning_curves(history, output)\n    plot_roc_pr(results, y, output)\n    plot_confusion(results, y, output)\n    plot_calibration(calibration, metrics, output)\n    plot_robustness(metrics, output, args.seeds)\n\n    rf_context = read_rf_context(root)\n    elapsed = time.time() - start\n    write_summary(output, dataset, df, args, device, metrics, repeated_summary, rf_context, elapsed)\n    metadata = {\n        "generated": datetime.now().isoformat(timespec="seconds"),\n        "dataset": str(dataset),\n        "output_dir": str(output),\n        "rows": int(len(df)),\n        "positives": int(y.sum()),\n        "negatives": int(len(y) - y.sum()),\n        "fingerprint_radius": args.fingerprint_radius,\n        "fingerprint_bits": args.fingerprint_bits,\n        "hidden_dims": args.hidden_dims,\n        "dropout": args.dropout,\n        "batch_size": args.batch_size,\n        "epochs": args.epochs,\n        "patience": args.patience,\n        "learning_rate": args.learning_rate,\n        "weight_decay": args.weight_decay,\n        "device": str(device),\n        "torch_version": torch.__version__,\n        "cuda_available": bool(torch.cuda.is_available()),\n        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,\n        "elapsed_seconds": elapsed,\n    }\n    (output / "mlp_morgan_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")\n    joblib.dump({"scaffolds": scaffolds}, output / "mlp_morgan_scaffolds.joblib")\n    print(f"Done. Outputs written to {output}")\n\n\nif __name__ == "__main__":\n    main()\n'


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
