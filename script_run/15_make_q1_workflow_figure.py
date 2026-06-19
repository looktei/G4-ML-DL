#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Create Q1-style workflow figure."""

STEP_ID = '15'
STEP_TITLE = 'Create Q1-style workflow figure'
EMBEDDED_SOURCE_NAME = '16_make_q1_workflow_figure.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Create a Q1-style workflow figure for the Nipah G4 ligand ML project."""\n\nfrom pathlib import Path\nimport textwrap\n\nimport matplotlib.pyplot as plt\nimport matplotlib.patches as patches\n\n\nBASE = Path(__file__).resolve().parents[1]\nOUT = BASE / "data/modeling/final_model_comparison/figures_research_summary"\nOUT.mkdir(parents=True, exist_ok=True)\n\nPNG = OUT / "nipah_g4_q1_workflow_figure.png"\nPDF = OUT / "nipah_g4_q1_workflow_figure.pdf"\n\n\nCOLORS = {\n    "data": "#5b8f22",\n    "data_light": "#dcefc8",\n    "scaffold": "#6c5a91",\n    "scaffold_light": "#e5ddf1",\n    "model": "#d9251d",\n    "model_light": "#fde2df",\n    "analysis": "#f1d43a",\n    "analysis_light": "#fff5b8",\n    "future": "#2f79b7",\n    "future_light": "#dcecf8",\n    "ink": "#1f2937",\n    "line": "#374151",\n    "muted": "#64748b",\n}\n\n\ndef wrap(text, width=18):\n    return "\\n".join(textwrap.wrap(text, width=width))\n\n\ndef add_box(ax, x, y, w, h, text, fc, ec=None, fs=8.5, weight="bold", lw=1.4):\n    if ec is None:\n        ec = COLORS["line"]\n    box = patches.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc, edgecolor=ec, linewidth=lw)\n    ax.add_patch(box)\n    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight=weight, color=COLORS["ink"], linespacing=1.12)\n    return box\n\n\ndef add_cylinder(ax, x, y, w, h, text, fc, ec=None, fs=8.5):\n    if ec is None:\n        ec = COLORS["line"]\n    rect = patches.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc, edgecolor=ec, linewidth=1.4)\n    top = patches.Ellipse((x, y + h / 2), w, h * 0.28, facecolor=fc, edgecolor=ec, linewidth=1.4)\n    bottom = patches.Ellipse((x, y - h / 2), w, h * 0.28, facecolor=fc, edgecolor=ec, linewidth=1.4)\n    ax.add_patch(rect)\n    ax.add_patch(bottom)\n    ax.add_patch(top)\n    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=COLORS["ink"], linespacing=1.1)\n\n\ndef add_arrow(ax, start, end, color=None, lw=1.35, dashed=False):\n    if color is None:\n        color = COLORS["line"]\n    ax.annotate(\n        "",\n        xy=end,\n        xytext=start,\n        arrowprops=dict(\n            arrowstyle="-|>",\n            lw=lw,\n            color=color,\n            shrinkA=4,\n            shrinkB=4,\n            linestyle="--" if dashed else "-",\n            mutation_scale=10,\n        ),\n    )\n\n\ndef add_label(ax, x, y, text, fs=7.8, color=None):\n    if color is None:\n        color = COLORS["muted"]\n    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color, linespacing=1.1)\n\n\ndef main():\n    plt.rcParams.update(\n        {\n            "font.family": "DejaVu Sans",\n            "figure.facecolor": "white",\n            "savefig.facecolor": "white",\n            "pdf.fonttype": 42,\n            "ps.fonttype": 42,\n        }\n    )\n\n    fig, ax = plt.subplots(figsize=(8.1, 10.6), dpi=300)\n    ax.set_xlim(0, 1)\n    ax.set_ylim(0, 1)\n    ax.axis("off")\n\n    ax.text(\n        0.5,\n        0.972,\n        "Integrated workflow for G4 ligand model development and candidate prioritization",\n        ha="center",\n        va="top",\n        fontsize=11.7,\n        fontweight="bold",\n        color="#111827",\n    )\n    ax.text(\n        0.5,\n        0.944,\n        "Curated G4LDB/ChEMBL dataset, scaffold-aware ML validation, and simulation-ready external screening",\n        ha="center",\n        va="top",\n        fontsize=8.3,\n        color=COLORS["muted"],\n    )\n\n    # Central data stream.\n    add_cylinder(ax, 0.5, 0.895, 0.16, 0.065, "9,346\\ncompounds", COLORS["data_light"], COLORS["data"], fs=8.4)\n    add_box(ax, 0.5, 0.815, 0.18, 0.055, "Data cleansing\\nSMILES validation", COLORS["data_light"], COLORS["data"], fs=7.8)\n    add_cylinder(\n        ax,\n        0.5,\n        0.73,\n        0.19,\n        0.07,\n        "4,673 positives\\n4,673 decoys",\n        COLORS["data_light"],\n        COLORS["data"],\n        fs=7.8,\n    )\n    add_arrow(ax, (0.5, 0.862), (0.5, 0.845), COLORS["data"])\n    add_arrow(ax, (0.5, 0.787), (0.5, 0.764), COLORS["data"])\n\n    # Branch connectors from curated dataset.\n    ax.plot([0.5, 0.19], [0.695, 0.695], color=COLORS["line"], lw=1.15)\n    ax.plot([0.5, 0.5], [0.695, 0.695], color=COLORS["line"], lw=1.15)\n    ax.plot([0.5, 0.81], [0.695, 0.695], color=COLORS["line"], lw=1.15)\n    add_arrow(ax, (0.19, 0.695), (0.19, 0.655), COLORS["line"])\n    add_arrow(ax, (0.5, 0.695), (0.5, 0.655), COLORS["line"])\n    add_arrow(ax, (0.81, 0.695), (0.81, 0.655), COLORS["line"])\n\n    # Left branch: scaffold/diversity.\n    add_box(ax, 0.19, 0.625, 0.18, 0.055, "Murcko\\nscaffold analysis", COLORS["scaffold_light"], COLORS["scaffold"], fs=7.7)\n    add_box(ax, 0.19, 0.548, 0.17, 0.05, "Scaffold\\ngrouping", COLORS["scaffold_light"], COLORS["scaffold"], fs=7.7)\n    add_arrow(ax, (0.19, 0.596), (0.19, 0.576), COLORS["scaffold"])\n\n    scaffold_x = [0.055, 0.145, 0.235, 0.325]\n    scaffold_txt = ["Scaffold 1", "Scaffold 2", "...", "Scaffold N"]\n    for x, txt in zip(scaffold_x, scaffold_txt):\n        add_cylinder(ax, x, 0.462, 0.077, 0.043, txt, COLORS["scaffold_light"], COLORS["scaffold"], fs=6.2)\n        add_arrow(ax, (0.19, 0.523), (x, 0.49), COLORS["scaffold"], lw=1.0)\n\n    add_box(ax, 0.19, 0.375, 0.27, 0.048, "Scaffold-aware split\\nand chemical diversity check", COLORS["scaffold_light"], COLORS["scaffold"], fs=7.0)\n    for x in scaffold_x:\n        add_arrow(ax, (x, 0.438), (0.19, 0.402), COLORS["scaffold"], lw=0.95)\n\n    # Middle branch: ML modeling.\n    add_box(ax, 0.5, 0.625, 0.18, 0.055, "Molecular\\nrepresentations", COLORS["model_light"], COLORS["model"], fs=7.7)\n    add_box(\n        ax,\n        0.5,\n        0.548,\n        0.20,\n        0.052,\n        "Morgan fingerprint\\nSMILES sequence\\nMolecular graph",\n        COLORS["model_light"],\n        COLORS["model"],\n        fs=6.5,\n    )\n    add_box(ax, 0.5, 0.455, 0.19, 0.06, "RF / MLP / SMILES\\nChemBERTa / GNN", COLORS["model_light"], COLORS["model"], fs=7.0)\n    add_box(ax, 0.5, 0.36, 0.18, 0.055, "Internal model\\nvalidation", COLORS["model_light"], COLORS["model"], fs=7.5)\n    add_arrow(ax, (0.5, 0.596), (0.5, 0.576), COLORS["model"])\n    add_arrow(ax, (0.5, 0.521), (0.5, 0.491), COLORS["model"])\n    add_arrow(ax, (0.5, 0.423), (0.5, 0.391), COLORS["model"])\n\n    # Right branch: exploratory analysis.\n    add_box(ax, 0.81, 0.625, 0.18, 0.055, "Exploratory data\\nanalysis", COLORS["analysis_light"], COLORS["analysis"], fs=7.7)\n    add_box(ax, 0.81, 0.545, 0.18, 0.055, "Chemical space\\nvisualization", COLORS["analysis_light"], COLORS["analysis"], fs=7.7)\n    add_box(ax, 0.81, 0.455, 0.18, 0.055, "PCA / descriptor\\nprofiling", COLORS["analysis_light"], COLORS["analysis"], fs=7.7)\n    add_box(ax, 0.81, 0.36, 0.18, 0.055, "Applicability-domain\\nsupport", COLORS["analysis_light"], COLORS["analysis"], fs=7.2)\n    add_arrow(ax, (0.81, 0.596), (0.81, 0.576), COLORS["analysis"])\n    add_arrow(ax, (0.81, 0.516), (0.81, 0.486), COLORS["analysis"])\n    add_arrow(ax, (0.81, 0.426), (0.81, 0.391), COLORS["analysis"])\n\n    # Cross-branch support into validation.\n    add_arrow(ax, (0.325, 0.375), (0.41, 0.36), COLORS["line"], lw=1.0)\n    add_arrow(ax, (0.72, 0.36), (0.59, 0.36), COLORS["line"], lw=1.0)\n\n    # Decision and future external validation stream.\n    add_box(\n        ax,\n        0.5,\n        0.265,\n        0.47,\n        0.065,\n        "Lead model decision: Molecular Graph GNN\\nScaffold split: ROC-AUC 0.9780, PR-AUC 0.9785, MCC 0.8462",\n        "#eef2ff",\n        "#4f46e5",\n        fs=6.9,\n    )\n    add_arrow(ax, (0.5, 0.331), (0.5, 0.301), "#4f46e5")\n\n    add_box(\n        ax,\n        0.5,\n        0.175,\n        0.50,\n        0.062,\n        "External screening pipeline\\nGNN primary prediction + Morgan RF consensus\\ncandidate tiering",\n        COLORS["future_light"],\n        COLORS["future"],\n        fs=6.45,\n    )\n    add_arrow(ax, (0.5, 0.232), (0.5, 0.208), COLORS["future"], dashed=True)\n\n    add_box(\n        ax,\n        0.5,\n        0.088,\n        0.50,\n        0.06,\n        "Downstream validation\\nmolecular docking -> molecular dynamics\\nfinal candidate shortlist",\n        "#e8f5e9",\n        "#1f8a45",\n        fs=6.45,\n    )\n    add_arrow(ax, (0.5, 0.143), (0.5, 0.118), "#1f8a45", dashed=True)\n\n    # Small side notes.\n    add_label(ax, 0.19, 0.315, "Generalization to\\nunseen scaffolds", fs=6.7)\n    add_label(ax, 0.81, 0.315, "Model reliability and\\nchemical-domain context", fs=6.7)\n    add_label(ax, 0.5, 0.018, "Dashed arrows indicate planned external/prospective validation steps before deployment claims.", fs=6.6)\n\n    fig.savefig(PNG, bbox_inches="tight", pad_inches=0.08)\n    fig.savefig(PDF, bbox_inches="tight", pad_inches=0.08)\n    plt.close(fig)\n    print(PNG)\n    print(PDF)\n\n\nif __name__ == "__main__":\n    main()\n'


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
