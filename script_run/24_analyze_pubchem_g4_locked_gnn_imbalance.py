#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Run imbalance-aware PubChem locked-GNN analysis."""

STEP_ID = '24'
STEP_TITLE = 'Run imbalance-aware PubChem locked-GNN analysis'
EMBEDDED_SOURCE_NAME = '25_analyze_pubchem_g4_locked_gnn_imbalance.py'
SEED_MODE = 'single'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Imbalance-aware analysis for the locked-GNN PubChem G4 benchmark.\n\nThis script consumes existing locked GNN predictions only. It does not retrain,\nselect models, or tune thresholds.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nfrom datetime import datetime\nfrom pathlib import Path\nfrom typing import Any\n\nimport matplotlib\n\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\nfrom sklearn.metrics import (\n    average_precision_score,\n    balanced_accuracy_score,\n    brier_score_loss,\n    confusion_matrix,\n    matthews_corrcoef,\n    recall_score,\n    roc_auc_score,\n)\n\n\nDEFAULT_PREDICTIONS = (\n    "data/pubchem_g4_expanded_external_evidence_set/locked_gnn_validation/"\n    "pubchem_g4_expanded_strict_locked_gnn_predictions.csv"\n)\nDEFAULT_OUTPUT = (\n    "data/pubchem_g4_expanded_external_evidence_set/locked_gnn_validation/"\n    "imbalance_sensitivity"\n)\nPOSITIVE_LABELS = {"assay_confirmed_active", "g4_active_like_annotation"}\nNEGATIVE_LABELS = {\n    "background_putative_decoy",\n    "assay_confirmed_inactive",\n    "assay_derived_inactive_like_high_concentration",\n}\nUNRESOLVED_LABEL = "unresolved_assay_evidence"\nTHRESHOLD = 0.5\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS, help="Existing locked-GNN predictions CSV.")\n    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Directory for imbalance-aware outputs.")\n    parser.add_argument("--iterations", type=int, default=1000, help="Balanced negative resampling iterations.")\n    parser.add_argument("--seed", type=int, default=20260603, help="Random seed for resampling.")\n    return parser.parse_args()\n\n\ndef root_from_script() -> Path:\n    return Path(__file__).resolve().parents[1]\n\n\ndef resolve(path: str | Path, root: Path) -> Path:\n    path = Path(path)\n    return path if path.is_absolute() else root / path\n\n\ndef assign_merged_benchmark_label(df: pd.DataFrame) -> pd.Series:\n    labels = pd.Series(np.nan, index=df.index, dtype="float64")\n    labels[df["conservative_label"].isin(POSITIVE_LABELS)] = 1.0\n    labels[df["conservative_label"].isin(NEGATIVE_LABELS)] = 0.0\n    return labels\n\n\ndef metric_row(subset: str, y_true: np.ndarray, score: np.ndarray) -> dict[str, Any]:\n    pred = (score >= THRESHOLD).astype(int)\n    row: dict[str, Any] = {\n        "subset": subset,\n        "n": int(len(y_true)),\n        "positives": int(y_true.sum()),\n        "negatives": int(len(y_true) - y_true.sum()),\n        "positive_rate": float(y_true.mean()) if len(y_true) else np.nan,\n        "threshold": THRESHOLD,\n    }\n    if len(y_true) == 0 or len(np.unique(y_true)) < 2:\n        return row\n    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()\n    row.update(\n        {\n            "pr_auc": float(average_precision_score(y_true, score)),\n            "roc_auc": float(roc_auc_score(y_true, score)),\n            "mcc": float(matthews_corrcoef(y_true, pred)),\n            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),\n            "sensitivity": float(recall_score(y_true, pred, zero_division=0)),\n            "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,\n            "brier": float(brier_score_loss(y_true, score)),\n            "tn": int(tn),\n            "fp": int(fp),\n            "fn": int(fn),\n            "tp": int(tp),\n        }\n    )\n    return row\n\n\ndef main_metrics(labeled: pd.DataFrame) -> pd.DataFrame:\n    rows = []\n    rows.append(\n        metric_row(\n            "merged_pubchem_evidence_supported_enrichment_background_benchmark",\n            labeled["merged_evaluation_label"].to_numpy(dtype=int),\n            labeled["gnn_probability"].to_numpy(dtype=float),\n        )\n    )\n    for tier, group in labeled.groupby("evidence_tier", dropna=False):\n        rows.append(\n            metric_row(\n                f"evidence_tier::{tier}",\n                group["merged_evaluation_label"].to_numpy(dtype=int),\n                group["gnn_probability"].to_numpy(dtype=float),\n            )\n        )\n    return pd.DataFrame(rows)\n\n\ndef confusion_table(labeled: pd.DataFrame) -> pd.DataFrame:\n    y = labeled["merged_evaluation_label"].to_numpy(dtype=int)\n    score = labeled["gnn_probability"].to_numpy(dtype=float)\n    pred = (score >= THRESHOLD).astype(int)\n    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()\n    return pd.DataFrame(\n        [\n            {"actual": 0, "predicted": 0, "count": int(tn)},\n            {"actual": 0, "predicted": 1, "count": int(fp)},\n            {"actual": 1, "predicted": 0, "count": int(fn)},\n            {"actual": 1, "predicted": 1, "count": int(tp)},\n        ]\n    )\n\n\ndef topk_metrics(labeled: pd.DataFrame, fractions: tuple[float, ...] = (0.01, 0.05, 0.10)) -> pd.DataFrame:\n    ranked = labeled.sort_values("gnn_probability", ascending=False).reset_index(drop=True)\n    y = ranked["merged_evaluation_label"].to_numpy(dtype=int)\n    total_pos = int(y.sum())\n    base_rate = float(y.mean())\n    rows = []\n    for frac in fractions:\n        k = max(1, int(np.ceil(frac * len(ranked))))\n        hit_pos = int(y[:k].sum())\n        precision = hit_pos / k\n        rows.append(\n            {\n                "top_fraction": frac,\n                "top_percent": int(round(frac * 100)),\n                "k": k,\n                "positives_recovered": hit_pos,\n                "total_positives": total_pos,\n                "recall_at_k": hit_pos / total_pos if total_pos else np.nan,\n                "precision_at_k": precision,\n                "base_positive_rate": base_rate,\n                "enrichment_over_base_rate": precision / base_rate if base_rate else np.nan,\n                "score_cutoff_min_in_top_k": float(ranked.loc[k - 1, "gnn_probability"]),\n            }\n        )\n    return pd.DataFrame(rows)\n\n\ndef balanced_resampling(labeled: pd.DataFrame, iterations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:\n    positives = labeled[labeled["merged_evaluation_label"].eq(1)].copy()\n    negatives = labeled[labeled["merged_evaluation_label"].eq(0)].copy()\n    n_pos = len(positives)\n    if n_pos == 0 or len(negatives) < n_pos:\n        raise SystemExit("Balanced resampling requires at least one positive and at least as many negatives.")\n    rng = np.random.default_rng(seed)\n    rows = []\n    neg_index = negatives.index.to_numpy()\n    for iteration in range(1, iterations + 1):\n        sampled_neg_index = rng.choice(neg_index, size=n_pos, replace=False)\n        sample = pd.concat([positives, negatives.loc[sampled_neg_index]], axis=0)\n        y = sample["merged_evaluation_label"].to_numpy(dtype=int)\n        score = sample["gnn_probability"].to_numpy(dtype=float)\n        row = metric_row("balanced_positive_negative_resample", y, score)\n        row["iteration"] = iteration\n        rows.append(row)\n    iter_df = pd.DataFrame(rows)\n    metric_cols = ["roc_auc", "pr_auc", "mcc", "balanced_accuracy", "sensitivity", "specificity"]\n    summary_rows = []\n    for metric in metric_cols:\n        values = iter_df[metric].dropna()\n        summary_rows.append(\n            {\n                "metric": metric,\n                "iterations": int(len(values)),\n                "mean": float(values.mean()),\n                "sd": float(values.std(ddof=1)),\n                "q2_5": float(values.quantile(0.025)),\n                "median": float(values.median()),\n                "q97_5": float(values.quantile(0.975)),\n            }\n        )\n    return iter_df, pd.DataFrame(summary_rows)\n\n\ndef score_summary(df: pd.DataFrame) -> pd.DataFrame:\n    rows = []\n    group_cols = ["evidence_tier", "conservative_label"]\n    for keys, group in df.groupby(group_cols, dropna=False):\n        labels = group["merged_evaluation_label"]\n        scores = group["gnn_probability"].astype(float)\n        rows.append(\n            {\n                "evidence_tier": keys[0],\n                "conservative_label": keys[1],\n                "n": int(len(group)),\n                "merged_evaluation_positives": int(labels.eq(1).sum()),\n                "merged_evaluation_negatives": int(labels.eq(0).sum()),\n                "unresolved_or_excluded": int(labels.isna().sum()),\n                "predicted_positive_n_at_0_5": int((scores >= THRESHOLD).sum()),\n                "predicted_positive_rate_at_0_5": float((scores >= THRESHOLD).mean()),\n                "mean_gnn_probability": float(scores.mean()),\n                "median_gnn_probability": float(scores.median()),\n                "q25_gnn_probability": float(scores.quantile(0.25)),\n                "q75_gnn_probability": float(scores.quantile(0.75)),\n                "min_gnn_probability": float(scores.min()),\n                "max_gnn_probability": float(scores.max()),\n                "median_nearest_train_similarity": float(group["nearest_train_similarity"].median())\n                if "nearest_train_similarity" in group\n                else np.nan,\n            }\n        )\n    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)\n\n\ndef plot_main_curves(labeled: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:\n    from sklearn.metrics import precision_recall_curve, roc_curve\n\n    y = labeled["merged_evaluation_label"].to_numpy(dtype=int)\n    score = labeled["gnn_probability"].to_numpy(dtype=float)\n    fpr, tpr, _ = roc_curve(y, score)\n    precision, recall, _ = precision_recall_curve(y, score)\n    main = metrics.iloc[0]\n    sns.set_theme(style="whitegrid", context="talk")\n    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=220)\n    axes[0].plot(fpr, tpr, linewidth=2, label=f"ROC-AUC={main[\'roc_auc\']:.3f}")\n    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)\n    axes[0].set_xlabel("False positive rate")\n    axes[0].set_ylabel("True positive rate")\n    axes[0].set_title("Merged benchmark ROC")\n    axes[0].legend()\n    axes[1].plot(recall, precision, linewidth=2, label=f"PR-AUC={main[\'pr_auc\']:.3f}")\n    axes[1].axhline(y.mean(), color="black", linestyle="--", linewidth=1, label=f"Base rate={y.mean():.3f}")\n    axes[1].set_xlabel("Recall")\n    axes[1].set_ylabel("Precision")\n    axes[1].set_title("Merged benchmark precision-recall")\n    axes[1].legend()\n    plt.tight_layout()\n    plt.savefig(output / "merged_pubchem_locked_gnn_roc_pr_curves.png")\n    plt.close()\n\n\ndef plot_topk(topk: pd.DataFrame, output: Path) -> None:\n    sns.set_theme(style="whitegrid", context="talk")\n    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)\n    sns.barplot(data=topk, x="top_percent", y="enrichment_over_base_rate", color="#4C78A8", ax=ax)\n    ax.set_xlabel("Top ranked compounds (%)")\n    ax.set_ylabel("Precision enrichment over base rate")\n    ax.set_title("Top-k enrichment in merged PubChem benchmark")\n    for i, row in topk.reset_index(drop=True).iterrows():\n        ax.text(i, row["enrichment_over_base_rate"], f"{row[\'positives_recovered\']}/{row[\'k\']}", ha="center", va="bottom")\n    plt.tight_layout()\n    plt.savefig(output / "merged_pubchem_locked_gnn_topk_enrichment.png")\n    plt.close()\n\n\ndef plot_resampling(iter_df: pd.DataFrame, output: Path) -> None:\n    metric_cols = ["roc_auc", "pr_auc", "mcc", "balanced_accuracy", "sensitivity", "specificity"]\n    long_df = iter_df.melt(value_vars=metric_cols, var_name="metric", value_name="value")\n    sns.set_theme(style="whitegrid", context="talk")\n    fig, ax = plt.subplots(figsize=(12, 6), dpi=220)\n    sns.boxplot(data=long_df, x="metric", y="value", color="#72B7B2", ax=ax)\n    ax.set_xlabel("")\n    ax.set_ylabel("Metric value across balanced resamples")\n    ax.set_title("Balanced negative resampling sensitivity")\n    ax.tick_params(axis="x", rotation=25)\n    plt.tight_layout()\n    plt.savefig(output / "balanced_negative_resampling_metric_distributions.png")\n    plt.close()\n\n\ndef plot_score_distribution(df: pd.DataFrame, output: Path) -> None:\n    sns.set_theme(style="whitegrid", context="talk")\n    order = df["conservative_label"].value_counts().index.tolist()\n    fig, ax = plt.subplots(figsize=(13, 7), dpi=220)\n    sns.boxplot(data=df, x="conservative_label", y="gnn_probability", order=order, ax=ax)\n    ax.axhline(THRESHOLD, color="black", linestyle="--", linewidth=1)\n    ax.set_xlabel("")\n    ax.set_ylabel("Locked GNN positive probability")\n    ax.set_title("Locked GNN scores by retained evidence label")\n    ax.tick_params(axis="x", rotation=25)\n    plt.tight_layout()\n    plt.savefig(output / "merged_pubchem_locked_gnn_score_distribution_by_label.png")\n    plt.close()\n\n\ndef to_markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:\n    formatted = df.copy()\n    for col in formatted.columns:\n        if pd.api.types.is_float_dtype(formatted[col]):\n            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))\n    return formatted.to_markdown(index=False)\n\n\ndef write_report(\n    output: Path,\n    predictions_path: Path,\n    df: pd.DataFrame,\n    labeled: pd.DataFrame,\n    metrics: pd.DataFrame,\n    topk: pd.DataFrame,\n    resampling_summary: pd.DataFrame,\n    summaries: pd.DataFrame,\n    iterations: int,\n    seed: int,\n) -> None:\n    main = metrics.iloc[0]\n    unresolved_n = int(df["merged_evaluation_label"].isna().sum())\n    lines = [\n        "# Locked GNN PubChem Imbalance Sensitivity Analysis",\n        "",\n        f"Generated: {datetime.now().isoformat(timespec=\'seconds\')}",\n        "",\n        "## Scope",\n        "",\n        "- Input is the existing locked-GNN PubChem prediction table; no model retraining, threshold tuning, or model selection was performed.",\n        "- Merged benchmark positives are `assay_confirmed_active` and `g4_active_like_annotation`.",\n        "- Merged benchmark negatives are `background_putative_decoy`, `assay_confirmed_inactive`, and `assay_derived_inactive_like_high_concentration`.",\n        "- `unresolved_assay_evidence` rows remain scored for descriptive summaries but are excluded from binary metrics.",\n        "",\n        "## Inputs And Counts",\n        "",\n        f"- Predictions: `{predictions_path}`",\n        f"- Total scored compounds: {len(df)}",\n        f"- Binary benchmark compounds: {len(labeled)}",\n        f"- Positives: {int(labeled[\'merged_evaluation_label\'].sum())}",\n        f"- Negatives/background: {int(len(labeled) - labeled[\'merged_evaluation_label\'].sum())}",\n        f"- Unresolved assay-evidence rows excluded from binary metrics: {unresolved_n}",\n        f"- Balanced resampling: {iterations} iterations, {int(labeled[\'merged_evaluation_label\'].sum())} positives plus equal sampled negatives per iteration, seed {seed}.",\n        "",\n        "## Main Merged Benchmark Metrics",\n        "",\n        to_markdown_table(metrics),\n        "",\n        "## Top-k Enrichment",\n        "",\n        to_markdown_table(topk),\n        "",\n        "## Balanced Negative Resampling Summary",\n        "",\n        to_markdown_table(resampling_summary),\n        "",\n        "## Stratified Descriptive Breakdown",\n        "",\n        to_markdown_table(summaries),\n        "",\n        "## Manuscript-Safe Interpretation",\n        "",\n        (\n            f"The merged evidence-supported PubChem enrichment/background benchmark contains "\n            f"{int(labeled[\'merged_evaluation_label\'].sum())} positives and "\n            f"{int(len(labeled) - labeled[\'merged_evaluation_label\'].sum())} negatives/background compounds. "\n            f"The locked GNN achieved PR-AUC {main[\'pr_auc\']:.3f}, ROC-AUC {main[\'roc_auc\']:.3f}, "\n            f"MCC {main[\'mcc\']:.3f}, balanced accuracy {main[\'balanced_accuracy\']:.3f}, "\n            f"sensitivity {main[\'sensitivity\']:.3f}, and specificity {main[\'specificity\']:.3f} at the fixed 0.5 threshold."\n        ),\n        "",\n        (\n            "These values should be described as evidence-supported ranking/enrichment performance on a merged PubChem "\n            "benchmark, not as fully experimental external validation. The largest negative class is a property-shaped "\n            "PubChem background/putative decoy set rather than uniformly assay-confirmed inactive chemistry."\n        ),\n        "",\n        (\n            "Balanced negative resampling checks whether the conclusion is dominated by the large background class. "\n            "Report the mean and interval estimates as sensitivity analyses alongside the full merged benchmark, "\n            "and retain unresolved assay-evidence compounds only as descriptive scored compounds."\n        ),\n    ]\n    (output / "locked_gnn_pubchem_imbalance_sensitivity_report.md").write_text(\n        "\\n".join(lines) + "\\n", encoding="utf-8"\n    )\n\n\ndef main() -> int:\n    args = parse_args()\n    root = root_from_script()\n    predictions_path = resolve(args.predictions, root)\n    output = resolve(args.output_dir, root)\n    output.mkdir(parents=True, exist_ok=True)\n\n    if not predictions_path.exists():\n        raise SystemExit(f"Predictions file not found: {predictions_path}")\n\n    df = pd.read_csv(predictions_path)\n    required = {"conservative_label", "evidence_tier", "gnn_probability"}\n    missing = sorted(required - set(df.columns))\n    if missing:\n        raise SystemExit(f"Predictions file missing required column(s): {\', \'.join(missing)}")\n\n    df["merged_evaluation_label"] = assign_merged_benchmark_label(df)\n    labeled = df.dropna(subset=["merged_evaluation_label"]).copy()\n    labeled["merged_evaluation_label"] = labeled["merged_evaluation_label"].astype(int)\n\n    if labeled["merged_evaluation_label"].nunique() < 2:\n        raise SystemExit("Merged binary benchmark needs both positive and negative labels.")\n\n    metrics = main_metrics(labeled)\n    conf = confusion_table(labeled)\n    topk = topk_metrics(labeled)\n    iter_df, resampling_summary = balanced_resampling(labeled, args.iterations, args.seed)\n    summaries = score_summary(df)\n\n    metrics.to_csv(output / "merged_pubchem_locked_gnn_main_metrics.csv", index=False)\n    conf.to_csv(output / "merged_pubchem_locked_gnn_confusion_matrix.csv", index=False)\n    topk.to_csv(output / "merged_pubchem_locked_gnn_topk_enrichment.csv", index=False)\n    iter_df.to_csv(output / "balanced_negative_resampling_iterations.csv", index=False)\n    resampling_summary.to_csv(output / "balanced_negative_resampling_summary.csv", index=False)\n    summaries.to_csv(output / "stratified_score_summary_by_evidence_label.csv", index=False)\n    df.to_csv(output / "pubchem_locked_gnn_predictions_with_merged_labels.csv", index=False)\n\n    plot_main_curves(labeled, metrics, output)\n    plot_topk(topk, output)\n    plot_resampling(iter_df, output)\n    plot_score_distribution(df, output)\n    write_report(output, predictions_path, df, labeled, metrics, topk, resampling_summary, summaries, args.iterations, args.seed)\n\n    print(f"Wrote imbalance-aware analysis outputs to {output}")\n    print(\n        "Main merged metrics: "\n        f"n={int(metrics.iloc[0][\'n\'])}, positives={int(metrics.iloc[0][\'positives\'])}, "\n        f"negatives={int(metrics.iloc[0][\'negatives\'])}, PR-AUC={metrics.iloc[0][\'pr_auc\']:.4f}, "\n        f"ROC-AUC={metrics.iloc[0][\'roc_auc\']:.4f}, MCC={metrics.iloc[0][\'mcc\']:.4f}"\n    )\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'


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
