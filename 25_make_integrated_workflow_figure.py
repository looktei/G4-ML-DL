#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Create integrated workflow figure."""

STEP_ID = '25'
STEP_TITLE = 'Create integrated workflow figure'
EMBEDDED_SOURCE_NAME = '26_make_integrated_workflow_figure.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Integrated workflow figure for G4 ligand model development & candidate\nprioritization. Style: clean vertical funnel, color-per-lane, terse boxes,\nicons per stage, gray italic side notes (low text density).\nPure matplotlib; no model training.\n"""\nimport os\nimport matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nfrom matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse\n\nROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))\nOUT = os.path.join(ROOT, "data/modeling/final_model_comparison/figures_research_summary",\n                   "nipah_g4_integrated_workflow.png")\nos.makedirs(os.path.dirname(OUT), exist_ok=True)\n\n# pastel fill / saturated border per lane\nLANE = {\n    "data":   ("#e7f0fb", "#2457a6"),\n    "feat":   ("#e3f4f2", "#1c7c78"),\n    "model":  ("#efeaf8", "#5a3e9a"),\n    "select": ("#e6f5ec", "#28784d"),\n    "bdb":    ("#fbf1df", "#9a6517"),\n    "pcm":    ("#e4f2f7", "#1f7a99"),\n    "robust": ("#fbe6e6", "#a63a3a"),\n    "out":    ("#e6f5ec", "#0f5132"),\n}\nINK = "#16243a"\nGRAY = "#6b7785"\n\nfig, ax = plt.subplots(figsize=(13.5, 15), dpi=150)\nax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")\n\ndef rbox(cx, cy, w, h, lane, icon, title, sub=None, tfs=12, sfs=10):\n    fill, edge = LANE[lane]\n    x, y = cx - w/2, cy - h/2\n    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=2",\n                 linewidth=2, edgecolor=edge, facecolor=fill, zorder=3))\n    label = f"{icon}  {title}" if icon else title\n    if sub:\n        ax.text(cx, cy + h*0.16, label, ha="center", va="center",\n                color=INK, fontsize=tfs, fontweight="bold", zorder=4)\n        ax.text(cx, cy - h*0.24, sub, ha="center", va="center",\n                color=edge, fontsize=sfs, zorder=4)\n    else:\n        ax.text(cx, cy, label, ha="center", va="center",\n                color=INK, fontsize=tfs, fontweight="bold", zorder=4)\n    return cx, cy, w, h, edge\n\ndef cyl(cx, cy, w, h, lane, icon, title, sub):\n    fill, edge = LANE[lane]\n    x, y = cx - w/2, cy - h/2\n    ax.add_patch(Ellipse((cx, y + h), w, h*0.28, facecolor=fill, edgecolor=edge, lw=2, zorder=3))\n    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, lw=2, zorder=2))\n    ax.plot([x, x], [y, y+h], color=edge, lw=2, zorder=3)\n    ax.plot([x+w, x+w], [y, y+h], color=edge, lw=2, zorder=3)\n    ax.add_patch(Ellipse((cx, y), w, h*0.28, facecolor=fill, edgecolor=edge, lw=2, zorder=4))\n    ax.text(cx, cy + h*0.12, f"{icon}  {title}", ha="center", va="center",\n            color=INK, fontsize=12, fontweight="bold", zorder=5)\n    ax.text(cx, cy - h*0.22, sub, ha="center", va="center", color=edge, fontsize=10, zorder=5)\n    return cx, cy, w, h, edge\n\ndef down(top, bot, color=GRAY, lw=2.4):\n    p1 = (top[0], top[1] - top[3]/2)\n    p2 = (bot[0], bot[1] + bot[3]/2)\n    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=20,\n                 lw=lw, color=color, zorder=1))\n\ndef split(top, bot, color):\n    p1 = (top[0], top[1] - top[3]/2)\n    p2 = (bot[0], bot[1] + bot[3]/2)\n    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=18,\n                 lw=2.2, color=color, zorder=1,\n                 connectionstyle="arc3,rad=0.0"))\n\ndef note(x, y, txt, color=GRAY):\n    ax.text(x, y, txt, ha="left", va="center", color=color, fontsize=9, style="italic", zorder=4)\n\n# Title\nax.text(50, 98, "Integrated Workflow — G4 Ligand Model Development & Candidate Prioritization",\n        ha="center", va="center", fontsize=15, fontweight="bold", color=INK)\nax.text(50, 95.2, "Nipah virus G-quadruplex project · 31 May – 3 June 2026 · locked model = scaffold-split GNN (seed 42)",\n        ha="center", va="center", fontsize=9.5, color=GRAY)\n\n# Spine\nd = cyl(50, 87, 30, 9, "data", "[1]", "DATASET", "9,346  ·  4,673 : 4,673")\nnote(67, 87, "G4LDB + ChEMBL decoys")\n\nf = rbox(50, 75, 34, 8, "feat", "[2]", "Features & Splits", "Morgan · descriptors · scaffold", sfs=9.5)\nnote(68.5, 75, "scaffold = generalization")\n\nm = rbox(50, 63, 34, 8, "model", "[3]", "Model Development", "5 families · 20-seed robustness", sfs=9.5)\nnote(68.5, 63, "RF·MLP·CNN·ChemBERTa·GNN")\n\ns = rbox(50, 51, 38, 8.5, "select", "[4]", "LOCKED Model — GNN", "ROC-AUC 0.978 · MCC 0.846", sfs=9.5)\nnote(70, 51, "scaffold-split winner")\n\ndown(d, f); down(f, m); down(m, s)\n\n# Fan out to 3 lanes\nb = rbox(18, 36, 26, 9, "bdb", "[5]", "BindingDB", "known 3/3 · Kd ROC 0.62", sfs=9)\np = rbox(50, 36, 26, 9, "pcm", "[6]", "PubChem", "strict 48 · expanded 442", sfs=9)\nr = rbox(82, 36, 26, 9, "robust", "[7]", "Robustness", "PR 0.832 · enrich 8.7×", sfs=9)\nsplit(s, b, LANE["bdb"][1]); split(s, p, LANE["pcm"][1]); split(s, r, LANE["robust"][1])\n\nnote(5.5, 30.5, "positive-control + domain probe")\nnote(40, 30.5, "evidence-supported set")\nnote(72, 30.5, "1000× balanced resampling")\n\n# Reconverge to outcome\no = rbox(50, 19, 56, 9.5, "out", "[8]", "Candidate Prioritization",\n         "robust enrichment benchmark (not full exp. validation)", sfs=9.5)\nfor lane_box, col in [(b, LANE["bdb"][1]), (p, LANE["pcm"][1]), (r, LANE["robust"][1])]:\n    p1 = (lane_box[0], lane_box[1] - lane_box[3]/2)\n    p2 = (o[0] + (lane_box[0]-50)*0.32, o[1] + o[3]/2)\n    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=18,\n                 lw=2.2, color=col, zorder=1))\n\n# footnote\nax.text(50, 9.5, "Single locked checkpoint for all external tests · no retraining / no threshold tuning · exact-overlap removal + similarity control",\n        ha="center", va="center", fontsize=8.6, color=GRAY, style="italic")\n\nplt.tight_layout()\nplt.savefig(OUT, bbox_inches="tight", facecolor="white")\nprint("WROTE", OUT, os.path.getsize(OUT), "bytes")\n'


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
