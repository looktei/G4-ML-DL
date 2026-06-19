#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Create research-summary workflow/overview figures."""

STEP_ID = '13'
STEP_TITLE = 'Create research-summary workflow/overview figures'
EMBEDDED_SOURCE_NAME = '14_make_research_summary_figures.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Create summary infographic and workflow pipeline figures for the Nipah G4 project."""\n\nfrom pathlib import Path\n\nimport matplotlib.pyplot as plt\nimport matplotlib.patches as patches\nimport pandas as pd\n\n\nBASE = Path(__file__).resolve().parents[1]\nCOMPARISON = BASE / "data/modeling/final_model_comparison"\nOUT = COMPARISON / "figures_research_summary"\nOUT.mkdir(parents=True, exist_ok=True)\n\nPRIMARY = pd.read_csv(COMPARISON / "final_model_primary_seed42_comparison.csv")\nREPEATED = pd.read_csv(COMPARISON / "final_model_repeated_seed_summary.csv")\n\nplt.rcParams.update(\n    {\n        "font.family": "DejaVu Sans",\n        "figure.facecolor": "#f7f9fb",\n        "axes.facecolor": "#f7f9fb",\n        "savefig.facecolor": "#f7f9fb",\n        "axes.edgecolor": "#334155",\n        "axes.labelcolor": "#1f2937",\n        "xtick.color": "#334155",\n        "ytick.color": "#334155",\n        "text.color": "#172033",\n    }\n)\n\n\ndef add_round_box(ax, x, y, w, h, fc, ec="#d6dee8", lw=1.2, r=0.035):\n    box = patches.FancyBboxPatch(\n        (x, y),\n        w,\n        h,\n        boxstyle=f"round,pad=0.012,rounding_size={r}",\n        linewidth=lw,\n        edgecolor=ec,\n        facecolor=fc,\n    )\n    ax.add_patch(box)\n    return box\n\n\ndef add_title(ax, title, subtitle):\n    ax.text(0.05, 0.95, title, fontsize=27, fontweight="bold", va="top", color="#111827")\n    ax.text(0.05, 0.905, subtitle, fontsize=13, va="top", color="#475569")\n    ax.plot([0.05, 0.95], [0.875, 0.875], color="#ccd7e3", lw=1.2)\n\n\ndef add_metric_card(ax, x, y, title, value, note, color):\n    add_round_box(ax, x, y, 0.205, 0.12, "#ffffff", ec="#dbe4ee", lw=1.1, r=0.025)\n    ax.text(x + 0.018, y + 0.085, title, fontsize=10.5, fontweight="bold", color="#334155")\n    ax.text(x + 0.018, y + 0.045, value, fontsize=24, fontweight="bold", color=color)\n    ax.text(x + 0.018, y + 0.018, note, fontsize=8.6, color="#64748b")\n\n\ndef draw_infographic():\n    fig, ax = plt.subplots(figsize=(17, 10), dpi=200)\n    ax.set_xlim(0, 1)\n    ax.set_ylim(0, 1)\n    ax.axis("off")\n\n    add_title(\n        ax,\n        "Nipah G4 Ligand Discovery: Project Overview",\n        "Curated ligand data, scaffold-aware machine learning, and simulation-ready candidate prioritization",\n    )\n\n    gnn_scaffold = PRIMARY[(PRIMARY["model_name"] == "Molecular Graph GNN") & (PRIMARY["split"] == "scaffold")].iloc[0]\n    gnn_repeat = REPEATED[(REPEATED["model_name"] == "Molecular Graph GNN") & (REPEATED["split"] == "scaffold")].iloc[0]\n    mlp_random = PRIMARY[(PRIMARY["model_name"] == "Morgan MLP") & (PRIMARY["split"] == "random")].iloc[0]\n    rf_scaffold = PRIMARY[(PRIMARY["model_name"] == "Morgan RF") & (PRIMARY["split"] == "scaffold")].iloc[0]\n\n    add_metric_card(ax, 0.05, 0.72, "Lead model", "Graph GNN", "best scaffold mean rank", "#2563eb")\n    add_metric_card(ax, 0.285, 0.72, "Scaffold ROC-AUC", f"{gnn_scaffold.roc_auc:.4f}", "primary seed 42", "#059669")\n    add_metric_card(ax, 0.52, 0.72, "Scaffold MCC", f"{gnn_scaffold.mcc:.4f}", "balanced decision quality", "#7c3aed")\n    add_metric_card(ax, 0.755, 0.72, "Scaffold Brier", f"{gnn_scaffold.brier_score:.4f}", "lower is better", "#dc2626")\n\n    add_round_box(ax, 0.05, 0.51, 0.28, 0.155, "#ffffff")\n    ax.text(0.075, 0.63, "1. Curated training task", fontsize=14, fontweight="bold")\n    ax.text(0.075, 0.595, "G4LDB positives + ChEMBL decoys", fontsize=11)\n    ax.text(0.075, 0.565, "Similarity filtering and assumed-negative framing", fontsize=10, color="#475569")\n    ax.text(0.075, 0.535, "Goal: binary G4 ligand classifier", fontsize=10, color="#475569")\n\n    add_round_box(ax, 0.36, 0.51, 0.28, 0.155, "#ffffff")\n    ax.text(0.385, 0.63, "2. Multi-model benchmark", fontsize=14, fontweight="bold")\n    models = "Morgan RF | Morgan MLP | SMILES CNN-BiGRU\\nChemBERTa-10M | Molecular Graph GNN"\n    ax.text(0.385, 0.585, models, fontsize=10.5, linespacing=1.4)\n    ax.text(0.385, 0.535, "Random and scaffold splits, seed 42 + repeated seeds", fontsize=10, color="#475569")\n\n    add_round_box(ax, 0.67, 0.51, 0.28, 0.155, "#ffffff")\n    ax.text(0.695, 0.63, "3. Validation signal", fontsize=14, fontweight="bold")\n    ax.text(0.695, 0.595, "Scaffold split is the key test", fontsize=11)\n    ax.text(0.695, 0.565, "Closest proxy for unseen database chemistry", fontsize=10, color="#475569")\n    ax.text(0.695, 0.535, f"GNN repeated ROC-AUC {gnn_repeat.roc_auc_mean:.4f} +/- {gnn_repeat.roc_auc_std:.4f}", fontsize=10, color="#475569")\n\n    # Comparison bars\n    left, bottom, width, height = 0.17, 0.215, 0.39, 0.245\n    add_round_box(ax, left - 0.005, bottom - 0.02, width + 0.02, height + 0.065, "#ffffff")\n    ax.text(left + 0.01, bottom + height + 0.027, "Primary seed 42: scaffold performance", fontsize=14, fontweight="bold")\n    bar_ax = fig.add_axes([left, bottom, width, height])\n    scaffold = PRIMARY[PRIMARY["split"] == "scaffold"].sort_values("roc_auc")\n    colors = ["#9aa8b7" if m != "Molecular Graph GNN" else "#2563eb" for m in scaffold["model_name"]]\n    bar_ax.barh(scaffold["model_name"], scaffold["roc_auc"], color=colors, height=0.48)\n    bar_ax.set_xlim(0.93, 0.98)\n    bar_ax.set_xlabel("ROC-AUC")\n    bar_ax.grid(axis="x", color="#e5ebf1", lw=0.8)\n    for spine in ["top", "right", "left"]:\n        bar_ax.spines[spine].set_visible(False)\n    bar_ax.tick_params(axis="y", labelsize=9)\n    bar_ax.tick_params(axis="x", labelsize=8)\n    for y, v in enumerate(scaffold["roc_auc"]):\n        bar_ax.text(v + 0.0005, y, f"{v:.4f}", va="center", fontsize=8.5, color="#334155")\n\n    add_round_box(ax, 0.63, 0.165, 0.32, 0.34, "#ffffff")\n    ax.text(0.655, 0.46, "Interpretive conclusion", fontsize=15, fontweight="bold")\n    bullets = [\n        ("Primary for external screening", "Molecular Graph GNN"),\n        ("Fast consensus baseline", f"Morgan RF, scaffold ROC-AUC {rf_scaffold.roc_auc:.4f}"),\n        ("Strong random-split classifier", f"Morgan MLP, MCC {mlp_random.mcc:.4f}"),\n        ("Q1 manuscript angle", "ML + docking + MD + prioritization"),\n    ]\n    y = 0.415\n    for head, text in bullets:\n        ax.text(0.66, y, head, fontsize=10.8, fontweight="bold", color="#1f2937")\n        ax.text(0.66, y - 0.028, text, fontsize=10.5, color="#475569")\n        y -= 0.075\n\n    path = OUT / "nipah_g4_project_overview_infographic.png"\n    fig.savefig(path, bbox_inches="tight")\n    plt.close(fig)\n    return path\n\n\ndef draw_workflow():\n    fig, ax = plt.subplots(figsize=(18, 10), dpi=200)\n    ax.set_xlim(0, 1)\n    ax.set_ylim(0, 1)\n    ax.axis("off")\n    add_title(\n        ax,\n        "Workflow Pipeline for External Candidate Discovery",\n        "From curated G4 ligand models to simulation-supported candidate prioritization",\n    )\n\n    steps = [\n        ("01", "Data curation", "G4LDB positives\\nChEMBL decoy set\\nSMILES validation"),\n        ("02", "Representations", "Morgan fingerprints\\nSMILES sequences\\nMolecular graphs"),\n        ("03", "Model training", "RF, MLP, SMILES\\nChemBERTa, GNN\\nrandom + scaffold"),\n        ("04", "Model selection", "Lead: Graph GNN\\nConsensus: Morgan RF\\nOptional: Morgan MLP"),\n        ("05", "External screening", "New compound database\\ncanonicalize SMILES\\nbatch prediction"),\n        ("06", "AD + consensus", "Applicability domain\\nagreement check\\nranked candidate tiers"),\n        ("07", "Docking", "G4 target structure\\nbinding pose\\ninteraction rationale"),\n        ("08", "MD simulation", "RMSD/RMSF/Rg/SASA\\nH-bond/contact occupancy\\nMM/PBSA if suitable"),\n        ("09", "Final shortlist", "Top candidates\\nmechanistic support\\nmanuscript figures"),\n    ]\n\n    xs = [0.11, 0.305, 0.50, 0.695, 0.89, 0.20, 0.40, 0.60, 0.80]\n    ys = [0.69, 0.69, 0.69, 0.69, 0.69, 0.36, 0.36, 0.36, 0.36]\n    w, h = 0.15, 0.18\n    palette = ["#dbeafe", "#dcfce7", "#fef3c7", "#ede9fe", "#fee2e2", "#e0f2fe", "#fae8ff", "#ecfccb", "#f1f5f9"]\n    edge = ["#60a5fa", "#34d399", "#f59e0b", "#a78bfa", "#fb7185", "#38bdf8", "#d946ef", "#84cc16", "#94a3b8"]\n\n    for i, ((num, title, body), x, y) in enumerate(zip(steps, xs, ys)):\n        add_round_box(ax, x - w / 2, y - h / 2, w, h, palette[i], ec=edge[i], lw=1.5, r=0.025)\n        ax.text(x - w / 2 + 0.014, y + h / 2 - 0.038, num, fontsize=11, fontweight="bold", color="#0f172a")\n        ax.text(x - w / 2 + 0.014, y + h / 2 - 0.072, title, fontsize=12.5, fontweight="bold", color="#111827")\n        ax.text(x - w / 2 + 0.014, y + h / 2 - 0.113, body, fontsize=8.8, linespacing=1.27, color="#334155", va="top")\n\n    def arrow(x1, y1, x2, y2):\n        ax.annotate(\n            "",\n            xy=(x2, y2),\n            xytext=(x1, y1),\n            arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#64748b", shrinkA=6, shrinkB=6),\n        )\n\n    for a, b in zip(range(4), range(1, 5)):\n        arrow(xs[a] + w / 2, ys[a], xs[b] - w / 2, ys[b])\n    arrow(xs[4], ys[4] - h / 2, xs[5], ys[5] + h / 2)\n    for a, b in zip(range(5, 8), range(6, 9)):\n        arrow(xs[a] + w / 2, ys[a], xs[b] - w / 2, ys[b])\n\n    add_round_box(ax, 0.055, 0.105, 0.285, 0.105, "#ffffff")\n    ax.text(0.08, 0.175, "Decision rule", fontsize=12.5, fontweight="bold")\n    ax.text(0.08, 0.145, "Tier 1: high GNN + RF support + in-domain", fontsize=8.8, color="#475569")\n    ax.text(0.08, 0.122, "Tier 2: high GNN but uncertain AD/consensus", fontsize=8.8, color="#475569")\n\n    add_round_box(ax, 0.36, 0.105, 0.285, 0.105, "#ffffff")\n    ax.text(0.385, 0.175, "Publication support", fontsize=12.5, fontweight="bold")\n    ax.text(0.385, 0.145, "Scaffold-aware ML + docking + MD stability", fontsize=8.8, color="#475569")\n    ax.text(0.385, 0.122, "Limitations: assumed negatives, computational validation", fontsize=8.8, color="#475569")\n\n    add_round_box(ax, 0.665, 0.105, 0.285, 0.105, "#ffffff")\n    ax.text(0.69, 0.175, "Core deliverables", fontsize=12.5, fontweight="bold")\n    ax.text(0.69, 0.145, "Candidate table, model evidence, docking/MD figures", fontsize=8.8, color="#475569")\n    ax.text(0.69, 0.122, "Integrated computational discovery manuscript", fontsize=8.8, color="#475569")\n\n    path = OUT / "nipah_g4_candidate_discovery_workflow.png"\n    fig.savefig(path, bbox_inches="tight")\n    plt.close(fig)\n    return path\n\n\ndef main():\n    for path in [draw_infographic(), draw_workflow()]:\n        print(path)\n\n\nif __name__ == "__main__":\n    main()\n'


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
