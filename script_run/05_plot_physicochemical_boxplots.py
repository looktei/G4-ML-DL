#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Generate physicochemical boxplots/statistics."""

STEP_ID = '05'
STEP_TITLE = 'Generate physicochemical boxplots/statistics'
EMBEDDED_SOURCE_NAME = '06_plot_physicochemical_boxplots.py'
SEED_MODE = 'single'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Box plot comparison of physicochemical properties by binary ligand group."""\n\nfrom __future__ import annotations\n\nimport argparse\nfrom pathlib import Path\n\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\nfrom scipy import stats\n\n\nPROPERTIES = [\n    ("rdkit_mw", "Molecular weight", "Da"),\n    ("rdkit_alogp", "AlogP", ""),\n    ("rdkit_hbd", "H-bond donors", "count"),\n    ("rdkit_hba", "H-bond acceptors", "count"),\n    ("rdkit_tpsa", "TPSA", "A^2"),\n]\nGROUP_LABELS = {\n    1: "Group 1\\nG4LDB positive",\n    0: "Group 2\\nChEMBL assumed-negative",\n}\nPALETTE = {\n    "Group 1\\nG4LDB positive": "#36C4E5",\n    "Group 2\\nChEMBL assumed-negative": "#F26A4B",\n}\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument(\n        "--dataset",\n        default="data/modeling/g4ldb_chembl_binary_dataset.csv",\n        help="Combined G4LDB/ChEMBL binary dataset CSV.",\n    )\n    parser.add_argument(\n        "--png-output",\n        default="data/modeling/g4ldb_chembl_physicochemical_boxplots.png",\n        help="Output PNG path.",\n    )\n    parser.add_argument(\n        "--csv-output",\n        default="data/modeling/g4ldb_chembl_physicochemical_stats.csv",\n        help="Statistical summary CSV output path.",\n    )\n    parser.add_argument(\n        "--summary-output",\n        default="data/modeling/g4ldb_chembl_physicochemical_boxplots_summary.md",\n        help="Markdown summary output path.",\n    )\n    parser.add_argument(\n        "--jitter-per-group",\n        type=int,\n        default=750,\n        help="Maximum molecules per group shown as jitter points in each panel.",\n    )\n    parser.add_argument("--seed", type=int, default=7, help="Random seed for jitter subsampling.")\n    return parser.parse_args()\n\n\ndef bh_fdr(p_values: list[float]) -> list[float]:\n    p = np.asarray(p_values, dtype=float)\n    n = len(p)\n    order = np.argsort(p)\n    ranked = p[order]\n    adjusted = ranked * n / np.arange(1, n + 1)\n    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]\n    adjusted = np.clip(adjusted, 0, 1)\n    q = np.empty(n, dtype=float)\n    q[order] = adjusted\n    return q.tolist()\n\n\ndef format_p(value: float) -> str:\n    if value < 1e-300:\n        return "<1e-300"\n    if value < 0.001:\n        return f"{value:.1e}"\n    if value < 0.01:\n        return f"{value:.4f}"\n    return f"{value:.3f}"\n\n\ndef stars(q_value: float) -> str:\n    if q_value < 0.0001:\n        return "****"\n    if q_value < 0.001:\n        return "***"\n    if q_value < 0.01:\n        return "**"\n    if q_value < 0.05:\n        return "*"\n    return "ns"\n\n\ndef cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:\n    # Rank-based formulation avoids the memory cost of all pairwise comparisons.\n    values = np.concatenate([x, y])\n    ranks = stats.rankdata(values, method="average")\n    rx = ranks[: len(x)].sum()\n    nx = len(x)\n    ny = len(y)\n    u = rx - nx * (nx + 1) / 2\n    return float((2 * u) / (nx * ny) - 1)\n\n\ndef load_dataset(path: Path) -> pd.DataFrame:\n    df = pd.read_csv(path)\n    required = {"label", *[column for column, _, _ in PROPERTIES]}\n    missing = sorted(required - set(df.columns))\n    if missing:\n        raise SystemExit(f"Missing required column(s): {\', \'.join(missing)}")\n\n    df = df.copy()\n    df["group"] = df["label"].map(GROUP_LABELS)\n    df = df[df["group"].notna()].copy()\n    for column, _, _ in PROPERTIES:\n        df[column] = pd.to_numeric(df[column], errors="coerce")\n    return df\n\n\ndef compute_stats(df: pd.DataFrame) -> pd.DataFrame:\n    rows = []\n    for column, name, unit in PROPERTIES:\n        group1 = df.loc[df["label"] == 1, column].dropna().to_numpy(float)\n        group2 = df.loc[df["label"] == 0, column].dropna().to_numpy(float)\n        test = stats.mannwhitneyu(group1, group2, alternative="two-sided")\n        rows.append(\n            {\n                "property": column,\n                "property_label": name,\n                "unit": unit,\n                "group1_n": int(len(group1)),\n                "group2_n": int(len(group2)),\n                "group1_median": float(np.median(group1)),\n                "group2_median": float(np.median(group2)),\n                "group1_iqr": float(np.percentile(group1, 75) - np.percentile(group1, 25)),\n                "group2_iqr": float(np.percentile(group2, 75) - np.percentile(group2, 25)),\n                "group1_mean": float(np.mean(group1)),\n                "group2_mean": float(np.mean(group2)),\n                "median_difference_group1_minus_group2": float(np.median(group1) - np.median(group2)),\n                "mannwhitney_u": float(test.statistic),\n                "p_value": float(test.pvalue),\n                "cliffs_delta_group1_vs_group2": cliffs_delta(group1, group2),\n            }\n        )\n    stats_df = pd.DataFrame(rows)\n    stats_df["q_value_bh_fdr"] = bh_fdr(stats_df["p_value"].tolist())\n    stats_df["significance"] = stats_df["q_value_bh_fdr"].map(stars)\n    return stats_df\n\n\ndef sample_for_points(df: pd.DataFrame, column: str, max_per_group: int, seed: int) -> pd.DataFrame:\n    parts = []\n    for label in [1, 0]:\n        subset = df.loc[df["label"] == label, ["group", column]].dropna()\n        if len(subset) > max_per_group:\n            subset = subset.sample(max_per_group, random_state=seed + label)\n        parts.append(subset)\n    return pd.concat(parts, ignore_index=True)\n\n\ndef add_significance_bar(ax: plt.Axes, q_value: float) -> None:\n    ymin, ymax = ax.get_ylim()\n    span = ymax - ymin\n    y = ymax - span * 0.11\n    h = span * 0.035\n    ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], color="black", linewidth=1.1)\n    ax.text(\n        0.5,\n        y + h * 1.15,\n        f"q={format_p(q_value)} {stars(q_value)}",\n        ha="center",\n        va="bottom",\n        fontsize=12,\n    )\n    ax.set_ylim(ymin, ymax + h * 1.8)\n\n\ndef display_limits(values: pd.Series) -> tuple[float, float]:\n    clean = values.dropna()\n    lower = float(clean.min())\n    upper = float(clean.max())\n    q005 = float(clean.quantile(0.005))\n    q995 = float(clean.quantile(0.995))\n    if lower >= 0:\n        ymin = 0.0\n    else:\n        ymin = q005 if abs(lower) > abs(q005) * 2 else lower\n    ymax = q995 if upper > q995 * 1.8 else upper\n    padding = (ymax - ymin) * 0.08 if ymax > ymin else 1.0\n    return ymin - padding * 0.2, ymax + padding\n\n\ndef write_plot(df: pd.DataFrame, stats_df: pd.DataFrame, output: Path, max_per_group: int, seed: int) -> None:\n    sns.set_theme(style="white", context="talk")\n    plt.rcParams.update(\n        {\n            "font.family": "DejaVu Sans",\n            "axes.linewidth": 1.1,\n            "xtick.major.width": 1.0,\n            "ytick.major.width": 1.0,\n            "axes.titlesize": 16,\n            "axes.labelsize": 12,\n        }\n    )\n\n    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.8), dpi=300)\n    axes = axes.flatten()\n    order = [GROUP_LABELS[1], GROUP_LABELS[0]]\n\n    for index, (column, name, unit) in enumerate(PROPERTIES):\n        ax = axes[index]\n        plot_df = df[["group", column]].dropna()\n        point_df = sample_for_points(df, column, max_per_group, seed)\n        sns.boxplot(\n            data=plot_df,\n            x="group",\n            y=column,\n            hue="group",\n            order=order,\n            hue_order=order,\n            palette=PALETTE,\n            width=0.58,\n            linewidth=1.3,\n            showfliers=False,\n            ax=ax,\n        )\n        sns.stripplot(\n            data=point_df,\n            x="group",\n            y=column,\n            hue="group",\n            order=order,\n            hue_order=order,\n            palette=PALETTE,\n            size=2.5,\n            alpha=0.28,\n            jitter=0.25,\n            linewidth=0.25,\n            edgecolor="black",\n            ax=ax,\n        )\n        legend = ax.get_legend()\n        if legend is not None:\n            legend.remove()\n        ax.set_ylim(*display_limits(plot_df[column]))\n        q_value = float(stats_df.loc[stats_df["property"] == column, "q_value_bh_fdr"].iloc[0])\n        add_significance_bar(ax, q_value)\n        ax.set_title(name, pad=18)\n        ax.set_xlabel("")\n        ax.set_ylabel(f"{name} ({unit})" if unit else name)\n        ax.tick_params(axis="x", labelsize=10)\n        ax.spines["top"].set_visible(False)\n        ax.spines["right"].set_visible(False)\n\n    axes[-1].axis("off")\n    handles = [\n        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETTE[label], markersize=13)\n        for label in order\n    ]\n    axes[-1].legend(handles, order, frameon=False, loc="center", fontsize=13)\n    axes[-1].text(\n        0.0,\n        0.18,\n        "Statistics: two-sided Mann-Whitney U test;\\nq-values are Benjamini-Hochberg FDR adjusted.\\nBoxes use all molecules; jitter shows a reproducible subset.",\n        transform=axes[-1].transAxes,\n        fontsize=11,\n        va="center",\n    )\n    axes[-1].text(\n        0.0,\n        0.03,\n        "Y axes are clipped at the 99.5th percentile only when extreme values would flatten the boxes.",\n        transform=axes[-1].transAxes,\n        fontsize=10,\n        va="bottom",\n    )\n\n    fig.suptitle(\n        "Physicochemical property comparison between Group 1 and Group 2 molecules",\n        fontsize=20,\n        y=0.995,\n    )\n    fig.tight_layout(rect=[0, 0, 1, 0.97], w_pad=2.3, h_pad=2.4)\n    output.parent.mkdir(parents=True, exist_ok=True)\n    fig.savefig(output, bbox_inches="tight")\n    plt.close(fig)\n\n\ndef write_summary(df: pd.DataFrame, stats_df: pd.DataFrame, output: Path, png_output: Path, csv_output: Path) -> None:\n    lines = [\n        "# Physicochemical Property Box Plot Summary",\n        "",\n        f"- Input rows: {len(df):,}",\n        f"- Group 1: {(df[\'label\'] == 1).sum():,} G4LDB positive molecules",\n        f"- Group 2: {(df[\'label\'] == 0).sum():,} ChEMBL assumed-negative molecules",\n        "- Statistical test: two-sided Mann-Whitney U test for each property",\n        "- Multiple testing correction: Benjamini-Hochberg FDR across the five properties",\n        "- Plot note: y axes are clipped at the 99.5th percentile only when extreme values would flatten the boxes; statistics use all molecules.",\n        f"- Figure: `{png_output}`",\n        f"- Statistics CSV: `{csv_output}`",\n        "",\n        "## Results",\n        "",\n        "| Property | Group 1 median | Group 2 median | Difference | p-value | q-value | Sig. | Cliff\'s delta |",\n        "|---|---:|---:|---:|---:|---:|:---:|---:|",\n    ]\n    for row in stats_df.itertuples(index=False):\n        lines.append(\n            "| {label} | {g1:.3f} | {g2:.3f} | {diff:.3f} | {p} | {q} | {sig} | {delta:.3f} |".format(\n                label=row.property_label,\n                g1=row.group1_median,\n                g2=row.group2_median,\n                diff=row.median_difference_group1_minus_group2,\n                p=format_p(row.p_value),\n                q=format_p(row.q_value_bh_fdr),\n                sig=row.significance,\n                delta=row.cliffs_delta_group1_vs_group2,\n            )\n        )\n    output.write_text("\\n".join(lines) + "\\n", encoding="utf-8")\n\n\ndef main() -> None:\n    args = parse_args()\n    dataset = Path(args.dataset)\n    png_output = Path(args.png_output)\n    csv_output = Path(args.csv_output)\n    summary_output = Path(args.summary_output)\n\n    df = load_dataset(dataset)\n    stats_df = compute_stats(df)\n    csv_output.parent.mkdir(parents=True, exist_ok=True)\n    stats_df.to_csv(csv_output, index=False)\n    write_plot(df, stats_df, png_output, args.jitter_per_group, args.seed)\n    write_summary(df, stats_df, summary_output, png_output, csv_output)\n    print(f"Wrote {png_output}")\n    print(f"Wrote {csv_output}")\n    print(f"Wrote {summary_output}")\n\n\nif __name__ == "__main__":\n    main()\n'


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
