#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Build final model-comparison tables/figures."""

STEP_ID = '12'
STEP_TITLE = 'Build final model-comparison tables/figures'
EMBEDDED_SOURCE_NAME = '13_build_final_model_comparison_table.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Build final model comparison tables for the G4 ligand classifier."""\n\nfrom __future__ import annotations\n\nfrom pathlib import Path\n\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\n\nPROJECT_ROOT = Path(__file__).resolve().parents[1]\nMODELING_DIR = PROJECT_ROOT / "data" / "modeling"\nOUTPUT_DIR = MODELING_DIR / "final_model_comparison"\n\nMETRICS = [\n    "roc_auc",\n    "pr_auc",\n    "mcc",\n    "balanced_accuracy",\n    "brier_score",\n]\n\nHIGHER_IS_BETTER = {\n    "roc_auc": True,\n    "pr_auc": True,\n    "mcc": True,\n    "balanced_accuracy": True,\n    "brier_score": False,\n}\n\nMODEL_SOURCES = [\n    {\n        "model_key": "rf",\n        "model_name": "Morgan RF",\n        "family": "Classical ML",\n        "representation": "Morgan fingerprint",\n        "primary_metrics": MODELING_DIR / "baseline_validation" / "baseline_validation_metrics.csv",\n        "repeated_metrics": MODELING_DIR / "baseline_validation" / "baseline_repeated_split_metrics.csv",\n        "repeated_summary": MODELING_DIR / "baseline_validation" / "baseline_repeated_split_summary.csv",\n    },\n    {\n        "model_key": "mlp",\n        "model_name": "Morgan MLP",\n        "family": "Deep learning",\n        "representation": "Morgan fingerprint",\n        "primary_metrics": MODELING_DIR / "deep_learning_mlp" / "mlp_morgan_metrics.csv",\n        "repeated_metrics": MODELING_DIR / "deep_learning_mlp" / "mlp_morgan_metrics.csv",\n        "repeated_summary": MODELING_DIR / "deep_learning_mlp" / "mlp_morgan_repeated_split_summary.csv",\n    },\n    {\n        "model_key": "smiles",\n        "model_name": "SMILES CNN-BiGRU",\n        "family": "Deep learning",\n        "representation": "SMILES sequence",\n        "primary_metrics": MODELING_DIR / "deep_learning_smiles" / "smiles_sequence_metrics.csv",\n        "repeated_metrics": MODELING_DIR / "deep_learning_smiles" / "smiles_sequence_metrics.csv",\n        "repeated_summary": MODELING_DIR / "deep_learning_smiles" / "smiles_sequence_repeated_split_summary.csv",\n    },\n    {\n        "model_key": "chemberta",\n        "model_name": "ChemBERTa-10M",\n        "family": "Pretrained transformer",\n        "representation": "SMILES language model",\n        "primary_metrics": MODELING_DIR / "deep_learning_chemberta" / "chemberta_metrics.csv",\n        "repeated_metrics": MODELING_DIR / "deep_learning_chemberta" / "chemberta_metrics.csv",\n        "repeated_summary": MODELING_DIR / "deep_learning_chemberta" / "chemberta_repeated_split_summary.csv",\n    },\n    {\n        "model_key": "gnn",\n        "model_name": "Molecular Graph GNN",\n        "family": "Graph deep learning",\n        "representation": "Molecular graph",\n        "primary_metrics": MODELING_DIR / "deep_learning_gnn" / "gnn_molecular_graph_metrics.csv",\n        "repeated_metrics": MODELING_DIR / "deep_learning_gnn" / "gnn_molecular_graph_metrics.csv",\n        "repeated_summary": MODELING_DIR / "deep_learning_gnn" / "gnn_molecular_graph_repeated_split_summary.csv",\n    },\n]\n\n\ndef add_model_metadata(df: pd.DataFrame, source: dict) -> pd.DataFrame:\n    out = df.copy()\n    for key in ["model_key", "model_name", "family", "representation"]:\n        out[key] = source[key]\n    return out\n\n\ndef load_primary_comparison() -> pd.DataFrame:\n    rows = []\n    for source in MODEL_SOURCES:\n        df = pd.read_csv(source["primary_metrics"])\n        primary = df[df["seed"].eq(42)].copy()\n        if primary.empty:\n            raise ValueError(f"No seed 42 rows found in {source[\'primary_metrics\']}")\n        primary = add_model_metadata(primary, source)\n        rows.append(primary)\n    keep = [\n        "model_key",\n        "model_name",\n        "family",\n        "representation",\n        "split",\n        "seed",\n        "train_n",\n        "test_n",\n        *METRICS,\n        "accuracy",\n        "sensitivity_recall",\n        "specificity",\n        "precision",\n        "f1",\n        "tn",\n        "fp",\n        "fn",\n        "tp",\n    ]\n    combined = pd.concat(rows, ignore_index=True)\n    return combined[[c for c in keep if c in combined.columns]].sort_values(["split", "model_key"])\n\n\ndef repeated_seed_counts(source: dict) -> pd.DataFrame:\n    df = pd.read_csv(source["repeated_metrics"])\n    if "seed" in df.columns:\n        df = df[df["seed"].ne(42)]\n    counts = df.groupby("split")["seed"].nunique().rename("repeated_n").reset_index()\n    counts["model_key"] = source["model_key"]\n    return counts\n\n\ndef load_repeated_summary() -> pd.DataFrame:\n    rows = []\n    counts = []\n    for source in MODEL_SOURCES:\n        df = pd.read_csv(source["repeated_summary"])\n        df = add_model_metadata(df, source)\n        rows.append(df)\n        counts.append(repeated_seed_counts(source))\n    combined = pd.concat(rows, ignore_index=True)\n    count_df = pd.concat(counts, ignore_index=True)\n    combined = combined.merge(count_df, on=["model_key", "split"], how="left")\n    order = [\n        "model_key",\n        "model_name",\n        "family",\n        "representation",\n        "split",\n        "repeated_n",\n        "roc_auc_mean",\n        "roc_auc_std",\n        "pr_auc_mean",\n        "pr_auc_std",\n        "mcc_mean",\n        "mcc_std",\n        "balanced_accuracy_mean",\n        "balanced_accuracy_std",\n        "brier_score_mean",\n        "brier_score_std",\n    ]\n    return combined[[c for c in order if c in combined.columns]].sort_values(["split", "model_key"])\n\n\ndef build_metric_rankings(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:\n    ranking_frames = []\n    for split, split_df in primary.groupby("split"):\n        rank_df = split_df[["model_key", "model_name", "split", *METRICS]].copy()\n        for metric in METRICS:\n            rank_df[f"{metric}_rank"] = rank_df[metric].rank(\n                ascending=not HIGHER_IS_BETTER[metric], method="min"\n            )\n        rank_cols = [f"{metric}_rank" for metric in METRICS]\n        rank_df["mean_rank"] = rank_df[rank_cols].mean(axis=1)\n        ranking_frames.append(rank_df)\n    rankings = pd.concat(ranking_frames, ignore_index=True).sort_values(["split", "mean_rank"])\n\n    long_rows = []\n    for _, row in primary.iterrows():\n        for metric in METRICS:\n            long_rows.append(\n                {\n                    "split": row["split"],\n                    "metric": metric,\n                    "model_key": row["model_key"],\n                    "model_name": row["model_name"],\n                    "value": row[metric],\n                    "rank": rankings.loc[\n                        (rankings["split"].eq(row["split"]))\n                        & (rankings["model_key"].eq(row["model_key"])),\n                        f"{metric}_rank",\n                    ].iloc[0],\n                }\n            )\n    return rankings, pd.DataFrame(long_rows).sort_values(["split", "metric", "rank"])\n\n\ndef fmt_num(value: float) -> str:\n    return f"{value:.4f}"\n\n\ndef fmt_mean_sd(row: pd.Series, metric: str) -> str:\n    return f"{row[f\'{metric}_mean\']:.4f} +/- {row[f\'{metric}_std\']:.4f}"\n\n\ndef markdown_table(df: pd.DataFrame, columns: list[str], headers: list[str]) -> str:\n    lines = []\n    lines.append("| " + " | ".join(headers) + " |")\n    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")\n    for _, row in df.iterrows():\n        values = [str(row[col]) for col in columns]\n        lines.append("| " + " | ".join(values) + " |")\n    return "\\n".join(lines)\n\n\ndef compact_primary_table(primary: pd.DataFrame, split: str) -> pd.DataFrame:\n    df = primary[primary["split"].eq(split)].copy()\n    for metric in METRICS:\n        df[metric] = df[metric].map(fmt_num)\n    return df[\n        [\n            "model_name",\n            "roc_auc",\n            "pr_auc",\n            "mcc",\n            "balanced_accuracy",\n            "brier_score",\n        ]\n    ].sort_values("model_name")\n\n\ndef compact_repeated_table(repeated: pd.DataFrame, split: str) -> pd.DataFrame:\n    df = repeated[repeated["split"].eq(split)].copy()\n    out = pd.DataFrame(\n        {\n            "model_name": df["model_name"],\n            "repeated_n": df["repeated_n"].astype(int),\n            "roc_auc": df.apply(lambda row: fmt_mean_sd(row, "roc_auc"), axis=1),\n            "pr_auc": df.apply(lambda row: fmt_mean_sd(row, "pr_auc"), axis=1),\n            "mcc": df.apply(lambda row: fmt_mean_sd(row, "mcc"), axis=1),\n            "balanced_accuracy": df.apply(lambda row: fmt_mean_sd(row, "balanced_accuracy"), axis=1),\n            "brier_score": df.apply(lambda row: fmt_mean_sd(row, "brier_score"), axis=1),\n        }\n    )\n    return out.sort_values("model_name")\n\n\ndef best_model_sentence(primary: pd.DataFrame, split: str, metric: str) -> str:\n    df = primary[primary["split"].eq(split)]\n    ascending = not HIGHER_IS_BETTER[metric]\n    best = df.sort_values(metric, ascending=ascending).iloc[0]\n    direction = "highest" if HIGHER_IS_BETTER[metric] else "lowest"\n    return f"{metric}: {best[\'model_name\']} ({direction}, {best[metric]:.4f})"\n\n\ndef write_summary(primary: pd.DataFrame, repeated: pd.DataFrame, rankings: pd.DataFrame) -> None:\n    lines = [\n        "# Final Model Comparison: G4 Ligand Binary Classifier",\n        "",\n        "This report aggregates the final available RF, MLP, SMILES, ChemBERTa, and GNN results across random and scaffold splits.",\n        "",\n        "## Primary seed 42 comparison",\n        "",\n        "### Random split",\n        markdown_table(\n            compact_primary_table(primary, "random"),\n            ["model_name", "roc_auc", "pr_auc", "mcc", "balanced_accuracy", "brier_score"],\n            ["Model", "ROC-AUC", "PR-AUC", "MCC", "Balanced Acc.", "Brier"],\n        ),\n        "",\n        "### Scaffold split",\n        markdown_table(\n            compact_primary_table(primary, "scaffold"),\n            ["model_name", "roc_auc", "pr_auc", "mcc", "balanced_accuracy", "brier_score"],\n            ["Model", "ROC-AUC", "PR-AUC", "MCC", "Balanced Acc.", "Brier"],\n        ),\n        "",\n        "## Repeated-seed robustness",\n        "",\n        "### Random split",\n        markdown_table(\n            compact_repeated_table(repeated, "random"),\n            ["model_name", "repeated_n", "roc_auc", "pr_auc", "mcc", "balanced_accuracy", "brier_score"],\n            ["Model", "N", "ROC-AUC", "PR-AUC", "MCC", "Balanced Acc.", "Brier"],\n        ),\n        "",\n        "### Scaffold split",\n        markdown_table(\n            compact_repeated_table(repeated, "scaffold"),\n            ["model_name", "repeated_n", "roc_auc", "pr_auc", "mcc", "balanced_accuracy", "brier_score"],\n            ["Model", "N", "ROC-AUC", "PR-AUC", "MCC", "Balanced Acc.", "Brier"],\n        ),\n        "",\n        "## Metric winners from primary seed 42",\n        "",\n    ]\n    for split in ["random", "scaffold"]:\n        lines.append(f"### {split.capitalize()} split")\n        for metric in METRICS:\n            lines.append(f"- {best_model_sentence(primary, split, metric)}")\n        lines.append("")\n\n    scaffold_rank = rankings[rankings["split"].eq("scaffold")].sort_values("mean_rank")\n    random_rank = rankings[rankings["split"].eq("random")].sort_values("mean_rank")\n    lines.extend(\n        [\n            "## Interpretation",\n            "",\n            f"- Best average primary-metric rank on scaffold split: {scaffold_rank.iloc[0][\'model_name\']} (mean rank {scaffold_rank.iloc[0][\'mean_rank\']:.2f}).",\n            f"- Best average primary-metric rank on random split: {random_rank.iloc[0][\'model_name\']} (mean rank {random_rank.iloc[0][\'mean_rank\']:.2f}).",\n            "- Scaffold split should be prioritized for drug-discovery interpretation because it tests generalization to chemically distinct scaffolds.",\n            "- Repeated-seed robustness now uses the same seed set for all comparator families.",\n            "- Use ROC-AUC/PR-AUC for ranking quality, MCC/balanced accuracy for thresholded classification quality, and Brier score for calibration quality.",\n            "",\n            "## Output files",\n            "",\n            "- `final_model_primary_seed42_comparison.csv`",\n            "- `final_model_repeated_seed_summary.csv`",\n            "- `final_model_primary_rankings.csv`",\n            "- `final_model_metric_winners.csv`",\n            "- `final_model_primary_metric_heatmap.png`",\n        ]\n    )\n    (OUTPUT_DIR / "final_model_comparison_summary.md").write_text("\\n".join(lines) + "\\n")\n\n\ndef plot_heatmap(primary: pd.DataFrame) -> None:\n    plot_df = primary.copy()\n    label_map = {\n        "roc_auc": "ROC-AUC",\n        "pr_auc": "PR-AUC",\n        "mcc": "MCC",\n        "balanced_accuracy": "Balanced acc.",\n        "brier_score": "Brier",\n    }\n    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)\n    for ax, split in zip(axes, ["random", "scaffold"]):\n        sub = plot_df[plot_df["split"].eq(split)].set_index("model_name")[METRICS]\n        sub = sub.rename(columns=label_map)\n        im = ax.imshow(sub.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)\n        ax.set_title(f"{split.capitalize()} split")\n        ax.set_xticks(range(len(sub.columns)))\n        ax.set_xticklabels(sub.columns, rotation=35, ha="right")\n        ax.set_yticks(range(len(sub.index)))\n        ax.set_yticklabels(sub.index)\n        for i in range(sub.shape[0]):\n            for j in range(sub.shape[1]):\n                value = sub.iloc[i, j]\n                color = "white" if value < 0.55 else "black"\n                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8, color=color)\n    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Metric value")\n    fig.suptitle("Primary seed 42 model comparison")\n    fig.savefig(OUTPUT_DIR / "final_model_primary_metric_heatmap.png", dpi=200, bbox_inches="tight")\n    plt.close(fig)\n\n\ndef main() -> None:\n    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n    primary = load_primary_comparison()\n    repeated = load_repeated_summary()\n    rankings, metric_winners = build_metric_rankings(primary)\n\n    primary.to_csv(OUTPUT_DIR / "final_model_primary_seed42_comparison.csv", index=False)\n    repeated.to_csv(OUTPUT_DIR / "final_model_repeated_seed_summary.csv", index=False)\n    rankings.to_csv(OUTPUT_DIR / "final_model_primary_rankings.csv", index=False)\n    metric_winners.to_csv(OUTPUT_DIR / "final_model_metric_winners.csv", index=False)\n\n    write_summary(primary, repeated, rankings)\n    plot_heatmap(primary)\n\n    print(f"Wrote final comparison outputs to {OUTPUT_DIR}")\n    print(primary[["model_name", "split", "roc_auc", "pr_auc", "mcc", "brier_score"]])\n\n\nif __name__ == "__main__":\n    main()\n'


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
