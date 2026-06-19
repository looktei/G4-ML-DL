#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Run locked GNN on expanded PubChem G4 evidence set."""

STEP_ID = '23'
STEP_TITLE = 'Run locked GNN on expanded PubChem G4 evidence set'
EMBEDDED_SOURCE_NAME = '24_validate_pubchem_g4_expanded_locked_gnn.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Run the locked molecular graph GNN on the expanded PubChem G4 evidence set."""\n\nfrom __future__ import annotations\n\nimport argparse\nimport importlib.util\nimport sys\nfrom datetime import datetime\nfrom pathlib import Path\nfrom typing import Any\n\nimport matplotlib\n\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\nfrom sklearn.metrics import (\n    average_precision_score,\n    balanced_accuracy_score,\n    brier_score_loss,\n    confusion_matrix,\n    matthews_corrcoef,\n    precision_recall_curve,\n    recall_score,\n    roc_auc_score,\n    roc_curve,\n)\n\n\nDEFAULT_INPUT = "data/pubchem_g4_expanded_external_evidence_set/pubchem_g4_expanded_strict_no_training_overlap.csv"\nDEFAULT_OUTPUT = "data/pubchem_g4_expanded_external_evidence_set/locked_gnn_validation"\nDEFAULT_CHECKPOINT = "data/modeling/deep_learning_gnn/gnn_molecular_graph_scaffold_seed42.pt"\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--input", default=DEFAULT_INPUT, help="Strict no-training-overlap PubChem evidence CSV.")\n    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Output directory.")\n    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Locked Molecular Graph GNN checkpoint.")\n    parser.add_argument("--batch-size", type=int, default=256)\n    parser.add_argument("--cpu", action="store_true")\n    return parser.parse_args()\n\n\ndef root_from_script() -> Path:\n    return Path(__file__).resolve().parents[1]\n\n\ndef resolve(path: str | Path, root: Path) -> Path:\n    path = Path(path)\n    return path if path.is_absolute() else root / path\n\n\ndef load_gnn_helpers(root: Path) -> Any:\n    helper_path = root / "scripts" / "validate_bindingdb_kd_gnn_external.py"\n    spec = importlib.util.spec_from_file_location("bindingdb_kd_validator", helper_path)\n    if spec is None or spec.loader is None:\n        raise SystemExit(f"Cannot load locked-GNN helper module: {helper_path}")\n    module = importlib.util.module_from_spec(spec)\n    sys.modules[spec.name] = module\n    spec.loader.exec_module(module)\n    return module\n\n\ndef metric_row(subset: str, df: pd.DataFrame) -> dict[str, Any]:\n    row: dict[str, Any] = {\n        "subset": subset,\n        "n": int(len(df)),\n        "positives": int(df["evaluation_label"].sum()) if len(df) else 0,\n        "negatives": int(len(df) - df["evaluation_label"].sum()) if len(df) else 0,\n        "threshold": 0.5,\n    }\n    if df.empty:\n        return row\n    y = df["evaluation_label"].to_numpy(dtype=int)\n    p = df["gnn_probability"].to_numpy(dtype=float)\n    pred = (p >= 0.5).astype(int)\n    if len(np.unique(y)) == 2:\n        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()\n        row.update(\n            {\n                "roc_auc": float(roc_auc_score(y, p)),\n                "pr_auc": float(average_precision_score(y, p)),\n                "brier_score": float(brier_score_loss(y, p)),\n                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),\n                "sensitivity_recall": float(recall_score(y, pred, zero_division=0)),\n                "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,\n                "mcc": float(matthews_corrcoef(y, pred)),\n                "tn": int(tn),\n                "fp": int(fp),\n                "fn": int(fn),\n                "tp": int(tp),\n            }\n        )\n    return row\n\n\ndef build_metrics(predicted: pd.DataFrame) -> pd.DataFrame:\n    labeled = predicted.dropna(subset=["evaluation_label"]).copy()\n    labeled["evaluation_label"] = labeled["evaluation_label"].astype(int)\n    rows = [\n        metric_row("strict_no_training_overlap_all_labeled", labeled),\n        metric_row(\n            "tier1_pubchem_bioassay_labeled_only",\n            labeled[labeled["evidence_tier"].eq("tier1_assay_confirmed_or_derived")],\n        ),\n        metric_row(\n            "assay_confirmed_active_vs_background_decoy",\n            labeled[\n                labeled["conservative_label"].isin(\n                    ["assay_confirmed_active", "background_putative_decoy"]\n                )\n            ],\n        ),\n        metric_row(\n            "active_like_tiers_vs_background_decoy",\n            labeled[\n                labeled["conservative_label"].isin(\n                    ["assay_confirmed_active", "g4_active_like_annotation", "background_putative_decoy"]\n                )\n            ],\n        ),\n    ]\n    return pd.DataFrame(rows)\n\n\ndef score_summary(predicted: pd.DataFrame) -> pd.DataFrame:\n    group_cols = ["evidence_tier", "conservative_label"]\n    rows = []\n    for keys, group in predicted.groupby(group_cols, dropna=False):\n        probs = group["gnn_probability"].astype(float)\n        rows.append(\n            {\n                "evidence_tier": keys[0],\n                "conservative_label": keys[1],\n                "n": int(len(group)),\n                "predicted_positive_n": int((probs >= 0.5).sum()),\n                "predicted_positive_rate": float((probs >= 0.5).mean()),\n                "median_gnn_probability": float(probs.median()),\n                "mean_gnn_probability": float(probs.mean()),\n                "q25_gnn_probability": float(probs.quantile(0.25)),\n                "q75_gnn_probability": float(probs.quantile(0.75)),\n                "median_nearest_train_similarity": float(group["nearest_train_similarity"].median())\n                if "nearest_train_similarity" in group\n                else np.nan,\n            }\n        )\n    return pd.DataFrame(rows).sort_values(["evidence_tier", "conservative_label"]).reset_index(drop=True)\n\n\ndef plot_score_distribution(predicted: pd.DataFrame, output: Path) -> None:\n    sns.set_theme(style="whitegrid", context="talk")\n    order = predicted["conservative_label"].value_counts().index.tolist()\n    fig, ax = plt.subplots(figsize=(13, 7), dpi=220)\n    sns.boxplot(data=predicted, x="conservative_label", y="gnn_probability", order=order, ax=ax)\n    sns.stripplot(\n        data=predicted,\n        x="conservative_label",\n        y="gnn_probability",\n        order=order,\n        color="black",\n        alpha=0.35,\n        size=3,\n        ax=ax,\n    )\n    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)\n    ax.set_xlabel("")\n    ax.set_ylabel("Locked GNN positive probability")\n    ax.set_title("Locked GNN scores by PubChem evidence label")\n    ax.tick_params(axis="x", rotation=25)\n    plt.tight_layout()\n    plt.savefig(output / "pubchem_g4_expanded_locked_gnn_score_distribution.png")\n    plt.close()\n\n\ndef plot_similarity_vs_score(predicted: pd.DataFrame, output: Path) -> None:\n    if "nearest_train_similarity" not in predicted.columns:\n        return\n    sns.set_theme(style="whitegrid", context="talk")\n    fig, ax = plt.subplots(figsize=(9, 6), dpi=220)\n    sns.scatterplot(\n        data=predicted,\n        x="nearest_train_similarity",\n        y="gnn_probability",\n        hue="conservative_label",\n        alpha=0.75,\n        s=45,\n        ax=ax,\n    )\n    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)\n    ax.set_xlabel("Nearest training Morgan Tanimoto")\n    ax.set_ylabel("Locked GNN positive probability")\n    ax.set_title("PubChem strict no-overlap: similarity versus score")\n    plt.tight_layout()\n    plt.savefig(output / "pubchem_g4_expanded_similarity_vs_gnn_score.png")\n    plt.close()\n\n\ndef plot_roc_pr(predicted: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:\n    labeled = predicted.dropna(subset=["evaluation_label"]).copy()\n    if labeled.empty or labeled["evaluation_label"].nunique() < 2:\n        return\n    y = labeled["evaluation_label"].astype(int).to_numpy()\n    p = labeled["gnn_probability"].astype(float).to_numpy()\n    fpr, tpr, _ = roc_curve(y, p)\n    precision, recall, _ = precision_recall_curve(y, p)\n    row = metrics[metrics["subset"].eq("strict_no_training_overlap_all_labeled")].iloc[0]\n    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=220)\n    axes[0].plot(fpr, tpr, linewidth=2, label=f"ROC-AUC={row[\'roc_auc\']:.3f}")\n    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)\n    axes[0].set_xlabel("False positive rate")\n    axes[0].set_ylabel("True positive rate")\n    axes[0].set_title("Strict no-overlap ROC")\n    axes[0].legend()\n    axes[1].plot(recall, precision, linewidth=2, label=f"PR-AUC={row[\'pr_auc\']:.3f}")\n    axes[1].set_xlabel("Recall")\n    axes[1].set_ylabel("Precision")\n    axes[1].set_title("Strict no-overlap PR")\n    axes[1].legend()\n    plt.tight_layout()\n    plt.savefig(output / "pubchem_g4_expanded_locked_gnn_roc_pr_curves.png")\n    plt.close()\n\n\ndef write_report(output: Path, input_path: Path, checkpoint: Path, predicted: pd.DataFrame, metrics: pd.DataFrame, summaries: pd.DataFrame) -> None:\n    labeled_n = int(predicted["evaluation_label"].notna().sum())\n    unresolved_n = int(predicted["evaluation_label"].isna().sum())\n    lines = [\n        "# PubChem Expanded G4 Locked-GNN Validation",\n        "",\n        f"Generated: {datetime.now().isoformat(timespec=\'seconds\')}",\n        "",\n        "## Locked Scope",\n        "",\n        "- Input: strict no-training-overlap PubChem expanded evidence set.",\n        "- Model: locked Molecular Graph GNN checkpoint.",\n        "- No retraining, threshold tuning, or model selection was performed.",\n        "- Background rows are property-shaped putative decoys, not experimental true inactives.",\n        "",\n        "## Inputs",\n        "",\n        f"- Evidence table: `{input_path}`",\n        f"- Checkpoint: `{checkpoint}`",\n        "",\n        "## Counts",\n        "",\n        f"- Total strict no-training-overlap compounds: {len(predicted)}",\n        f"- Labeled for evaluation after benchmark labeling: {labeled_n}",\n        f"- Unresolved assay evidence scored but excluded from binary metrics: {unresolved_n}",\n        "",\n        "## Metrics",\n        "",\n        metrics.to_markdown(index=False, floatfmt=".4f"),\n        "",\n        "## Score Summary By Evidence Label",\n        "",\n        summaries.to_markdown(index=False, floatfmt=".4f"),\n        "",\n        "## Manuscript-Safe Interpretation",\n        "",\n        "- Treat the all-labeled metric as an enrichment/background benchmark, not a pure bioassay external validation, because most negatives are PubChem background/putative decoys.",\n        "- Treat Tier 1 BioAssay metrics as a small strict PubChem assay checkpoint; class imbalance and small inactive count limit reviewer-facing claims.",\n        "- Use unresolved assay-evidence scores descriptively only, not in binary ROC/PR metrics.",\n    ]\n    (output / "pubchem_g4_expanded_locked_gnn_validation_report.md").write_text(\n        "\\n".join(lines) + "\\n", encoding="utf-8"\n    )\n\n\ndef main() -> int:\n    args = parse_args()\n    root = root_from_script()\n    input_path = resolve(args.input, root)\n    output = resolve(args.output_dir, root)\n    checkpoint = resolve(args.checkpoint, root)\n    output.mkdir(parents=True, exist_ok=True)\n\n    if not input_path.exists():\n        raise SystemExit(f"Input table not found: {input_path}")\n    if not checkpoint.exists():\n        raise SystemExit(f"Locked checkpoint not found: {checkpoint}")\n\n    helpers = load_gnn_helpers(root)\n    evidence = pd.read_csv(input_path)\n    required = {"canonical_smiles", "binary_label_for_evaluation", "conservative_label", "evidence_tier"}\n    missing = sorted(required - set(evidence.columns))\n    if missing:\n        raise SystemExit(f"Input is missing required column(s): {\', \'.join(missing)}")\n\n    predicted, status = helpers.predict_with_locked_gnn(evidence, checkpoint, args.batch_size, args.cpu)\n    if not status.ok:\n        raise SystemExit(status.message)\n    predicted["evaluation_label"] = pd.to_numeric(predicted["binary_label_for_evaluation"], errors="coerce")\n    predicted.loc[predicted["conservative_label"].eq("background_putative_decoy"), "evaluation_label"] = 0\n    predicted.loc[predicted["conservative_label"].eq("g4_active_like_annotation"), "evaluation_label"] = 1\n    predicted.to_csv(output / "pubchem_g4_expanded_strict_locked_gnn_predictions.csv", index=False)\n\n    metrics = build_metrics(predicted)\n    summaries = score_summary(predicted)\n    metrics.to_csv(output / "pubchem_g4_expanded_locked_gnn_metrics.csv", index=False)\n    summaries.to_csv(output / "pubchem_g4_expanded_locked_gnn_score_summary_by_label.csv", index=False)\n\n    top = predicted.sort_values("gnn_probability", ascending=False).head(100)\n    top.to_csv(output / "pubchem_g4_expanded_top100_locked_gnn_hits.csv", index=False)\n\n    plot_score_distribution(predicted, output)\n    plot_similarity_vs_score(predicted, output)\n    plot_roc_pr(predicted, metrics, output)\n    write_report(output, input_path, checkpoint, predicted, metrics, summaries)\n    print(status.message)\n    print(f"Wrote outputs to {output}")\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'


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
