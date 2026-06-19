#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Create Q1-style key-result figure."""

STEP_ID = '14'
STEP_TITLE = 'Create Q1-style key-result figure'
EMBEDDED_SOURCE_NAME = '15_make_q1_key_result_figure.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Create a publication-style key result figure for the Nipah G4 project."""\n\nfrom pathlib import Path\n\nimport matplotlib.pyplot as plt\nimport matplotlib.patches as patches\nimport pandas as pd\n\n\nBASE = Path(__file__).resolve().parents[1]\nCOMPARISON = BASE / "data/modeling/final_model_comparison"\nOUT = COMPARISON / "figures_research_summary"\nOUT.mkdir(parents=True, exist_ok=True)\n\nPRIMARY = pd.read_csv(COMPARISON / "final_model_primary_seed42_comparison.csv")\nREPEATED = pd.read_csv(COMPARISON / "final_model_repeated_seed_summary.csv")\nRANKINGS = pd.read_csv(COMPARISON / "final_model_primary_rankings.csv")\n\nMODEL_ORDER = [\n    "ChemBERTa-10M",\n    "Morgan MLP",\n    "SMILES CNN-BiGRU",\n    "Morgan RF",\n    "Molecular Graph GNN",\n]\n\nCOLORS = {\n    "Molecular Graph GNN": "#0f766e",\n    "Morgan RF": "#2563eb",\n    "SMILES CNN-BiGRU": "#7c3aed",\n    "Morgan MLP": "#ea580c",\n    "ChemBERTa-10M": "#64748b",\n}\n\n\nplt.rcParams.update(\n    {\n        "font.family": "DejaVu Sans",\n        "figure.facecolor": "white",\n        "savefig.facecolor": "white",\n        "axes.facecolor": "white",\n        "axes.edgecolor": "#111827",\n        "axes.labelcolor": "#111827",\n        "xtick.color": "#111827",\n        "ytick.color": "#111827",\n        "text.color": "#111827",\n        "axes.titleweight": "bold",\n        "axes.titlesize": 12,\n        "axes.labelsize": 10,\n    }\n)\n\n\ndef panel_label(ax, label):\n    ax.text(\n        -0.12,\n        1.08,\n        label,\n        transform=ax.transAxes,\n        fontsize=15,\n        fontweight="bold",\n        va="top",\n        ha="left",\n    )\n\n\ndef add_box(ax, xy, width, height, facecolor, edgecolor="#d1d5db", lw=1.2):\n    box = patches.FancyBboxPatch(\n        xy,\n        width,\n        height,\n        boxstyle="round,pad=0.012,rounding_size=0.018",\n        linewidth=lw,\n        edgecolor=edgecolor,\n        facecolor=facecolor,\n        transform=ax.transAxes,\n        clip_on=False,\n    )\n    ax.add_patch(box)\n    return box\n\n\ndef metric_bar_panel(ax):\n    scaffold = PRIMARY[PRIMARY["split"] == "scaffold"].copy()\n    scaffold["model_name"] = pd.Categorical(scaffold["model_name"], MODEL_ORDER, ordered=True)\n    scaffold = scaffold.sort_values("model_name")\n\n    metrics = [\n        ("roc_auc", "ROC-AUC"),\n        ("pr_auc", "PR-AUC"),\n        ("mcc", "MCC"),\n        ("balanced_accuracy", "Balanced acc."),\n    ]\n    x = range(len(metrics))\n    width = 0.15\n    offsets = [-2 * width, -width, 0, width, 2 * width]\n\n    for idx, model in enumerate(MODEL_ORDER):\n        row = scaffold[scaffold["model_name"] == model].iloc[0]\n        values = [row[key] for key, _ in metrics]\n        ax.bar(\n            [i + offsets[idx] for i in x],\n            values,\n            width=width,\n            color=COLORS[model],\n            label=model,\n            edgecolor="white",\n            linewidth=0.8,\n            alpha=0.96,\n        )\n\n    ax.set_title("Scaffold split: GNN is the top model across ranking/classification metrics", loc="left")\n    ax.set_xticks(list(x))\n    ax.set_xticklabels([label for _, label in metrics], rotation=0)\n    ax.set_ylim(0.62, 1.02)\n    ax.set_ylabel("Metric value")\n    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)\n    ax.spines[["top", "right"]].set_visible(False)\n    ax.legend(\n        ncol=3,\n        frameon=False,\n        fontsize=8.5,\n        loc="lower center",\n        bbox_to_anchor=(0.5, -0.30),\n        handlelength=1.1,\n        columnspacing=1.1,\n    )\n\n    gnn = scaffold[scaffold["model_name"] == "Molecular Graph GNN"].iloc[0]\n    ax.annotate(\n        "GNN leads all shown\\nscaffold metrics",\n        xy=(3, gnn["balanced_accuracy"]),\n        xytext=(3.25, 0.99),\n        arrowprops=dict(arrowstyle="-|>", color="#0f766e", lw=1.2),\n        fontsize=9,\n        color="#0f766e",\n        ha="left",\n        va="top",\n    )\n\n\ndef repeated_seed_panel(ax):\n    scaffold = REPEATED[REPEATED["split"] == "scaffold"].copy()\n    scaffold["model_name"] = pd.Categorical(scaffold["model_name"], MODEL_ORDER, ordered=True)\n    scaffold = scaffold.sort_values("roc_auc_mean")\n\n    y = range(len(scaffold))\n    ax.errorbar(\n        scaffold["roc_auc_mean"],\n        y,\n        xerr=scaffold["roc_auc_std"],\n        fmt="o",\n        markersize=7,\n        capsize=4,\n        color="#111827",\n        ecolor="#475569",\n        elinewidth=1.4,\n    )\n    for yi, (_, row) in zip(y, scaffold.iterrows()):\n        ax.scatter(\n            row["roc_auc_mean"],\n            yi,\n            s=90 if row["model_name"] == "Molecular Graph GNN" else 58,\n            color=COLORS[str(row["model_name"])],\n            zorder=3,\n        )\n        ax.text(\n            row["roc_auc_mean"] + 0.0018,\n            yi,\n            f"{row[\'roc_auc_mean\']:.4f} +/- {row[\'roc_auc_std\']:.4f}",\n            va="center",\n            fontsize=8.5,\n            color="#374151",\n        )\n\n    ax.set_yticks(list(y))\n    ax.set_yticklabels(scaffold["model_name"])\n    ax.set_xlim(0.935, 0.984)\n    ax.set_xlabel("Repeated-seed scaffold ROC-AUC")\n    ax.set_title("Robustness over repeated scaffold splits", loc="left")\n    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)\n    ax.spines[["top", "right", "left"]].set_visible(False)\n    ax.tick_params(axis="y", length=0)\n\n\ndef decision_panel(ax):\n    ax.axis("off")\n    gnn = PRIMARY[\n        (PRIMARY["split"] == "scaffold")\n        & (PRIMARY["model_name"] == "Molecular Graph GNN")\n    ].iloc[0]\n    gnn_rep = REPEATED[\n        (REPEATED["split"] == "scaffold")\n        & (REPEATED["model_name"] == "Molecular Graph GNN")\n    ].iloc[0]\n    rf = PRIMARY[\n        (PRIMARY["split"] == "scaffold")\n        & (PRIMARY["model_name"] == "Morgan RF")\n    ].iloc[0]\n    rank = RANKINGS[\n        (RANKINGS["split"] == "scaffold")\n        & (RANKINGS["model_name"] == "Molecular Graph GNN")\n    ].iloc[0]\n\n    add_box(ax, (0.02, 0.05), 0.96, 0.90, "#f8fafc", "#cbd5e1")\n    ax.text(0.06, 0.83, "Key result for Q1 framing", fontsize=15, fontweight="bold", transform=ax.transAxes)\n    ax.text(\n        0.06,\n        0.71,\n        "A molecular graph GNN provides the strongest scaffold-level generalization,\\n"\n        "supporting its use as the primary model for external G4-ligand screening.",\n        fontsize=10.3,\n        linespacing=1.18,\n        transform=ax.transAxes,\n    )\n\n    facts = [\n        ("Scaffold ROC-AUC", f"{gnn.roc_auc:.4f}"),\n        ("Scaffold PR-AUC", f"{gnn.pr_auc:.4f}"),\n        ("Scaffold MCC", f"{gnn.mcc:.4f}"),\n        ("Mean primary rank", f"{rank.mean_rank:.2f}"),\n    ]\n    card_positions = [(0.06, 0.38), (0.52, 0.38), (0.06, 0.20), (0.52, 0.20)]\n    for (label, value), (x, y) in zip(facts, card_positions):\n        ax.text(x, y, value, fontsize=18, fontweight="bold", color="#0f766e", transform=ax.transAxes)\n        ax.text(x, y - 0.08, label, fontsize=8.8, color="#475569", transform=ax.transAxes)\n\n    ax.text(0.06, 0.07, f"Repeated scaffold ROC-AUC: {gnn_rep.roc_auc_mean:.4f} +/- {gnn_rep.roc_auc_std:.4f}", fontsize=8.8, color="#334155", transform=ax.transAxes)\n\n\ndef workflow_panel(ax):\n    ax.axis("off")\n    ax.set_title("Manuscript-ready interpretation", loc="left", pad=12)\n\n    nodes = [\n        ("Curated G4\\ndataset", "#dbeafe", "#2563eb"),\n        ("Scaffold\\nmodel\\ncomparison", "#dcfce7", "#16a34a"),\n        ("Lead model:\\nGraph GNN", "#ccfbf1", "#0f766e"),\n        ("External\\nscreening\\n+ RF consensus", "#fef3c7", "#d97706"),\n        ("Docking / MD\\nvalidation", "#ede9fe", "#7c3aed"),\n    ]\n    x0, y0, w, h, gap = 0.02, 0.38, 0.158, 0.31, 0.04\n    for idx, (text, face, edge) in enumerate(nodes):\n        x = x0 + idx * (w + gap)\n        add_box(ax, (x, y0), w, h, face, edge, lw=1.5)\n        ax.text(x + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=8.2, fontweight="bold", transform=ax.transAxes, linespacing=1.08)\n        if idx < len(nodes) - 1:\n            ax.annotate(\n                "",\n                xy=(x + w + gap - 0.012, y0 + h / 2),\n                xytext=(x + w + 0.012, y0 + h / 2),\n                xycoords=ax.transAxes,\n                textcoords=ax.transAxes,\n                arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.4),\n            )\n\n    ax.text(\n        0.02,\n        0.12,\n        "Recommended claim: robust scaffold-aware ML prioritization, followed by molecular simulation support.\\n"\n        "Avoid claiming true external validation until independent/prospective candidates are tested.",\n        fontsize=9.0,\n        color="#334155",\n        transform=ax.transAxes,\n        wrap=True,\n    )\n\n\ndef main():\n    fig = plt.figure(figsize=(15.5, 10.5), dpi=300)\n    grid = fig.add_gridspec(\n        3,\n        2,\n        height_ratios=[0.18, 1.0, 0.82],\n        width_ratios=[1.28, 1.0],\n        hspace=0.78,\n        wspace=0.36,\n    )\n\n    title_ax = fig.add_subplot(grid[0, :])\n    title_ax.axis("off")\n    title_ax.text(\n        0.0,\n        0.86,\n        "Structure-aware graph learning identifies the lead model for G4-ligand discovery",\n        fontsize=18,\n        fontweight="bold",\n        ha="left",\n        va="top",\n    )\n    title_ax.text(\n        0.0,\n        0.22,\n        "Key result: molecular graph GNN ranks first across all scaffold-split primary metrics and remains robust over repeated seeds.",\n        fontsize=10.8,\n        color="#475569",\n        ha="left",\n        va="top",\n    )\n\n    ax_a = fig.add_subplot(grid[1, 0])\n    ax_b = fig.add_subplot(grid[1, 1])\n    ax_c = fig.add_subplot(grid[2, 0])\n    ax_d = fig.add_subplot(grid[2, 1])\n\n    panel_label(ax_a, "A")\n    panel_label(ax_b, "B")\n    panel_label(ax_c, "C")\n    panel_label(ax_d, "D")\n\n    metric_bar_panel(ax_a)\n    repeated_seed_panel(ax_b)\n    workflow_panel(ax_c)\n    decision_panel(ax_d)\n\n    output = OUT / "nipah_g4_q1_key_result_figure.png"\n    pdf_output = OUT / "nipah_g4_q1_key_result_figure.pdf"\n    fig.savefig(output, bbox_inches="tight")\n    fig.savefig(pdf_output, bbox_inches="tight")\n    plt.close(fig)\n    print(output)\n    print(pdf_output)\n\n\nif __name__ == "__main__":\n    main()\n'


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
