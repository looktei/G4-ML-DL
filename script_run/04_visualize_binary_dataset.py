#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Visualize/validate binary G4LDB vs ChEMBL dataset."""

STEP_ID = '04'
STEP_TITLE = 'Visualize/validate binary G4LDB vs ChEMBL dataset'
EMBEDDED_SOURCE_NAME = '05_visualize_binary_dataset.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Visualize and validate the G4LDB/ChEMBL binary ligand dataset."""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport warnings\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nfrom typing import Any\n\n\nCORE_FIELDS = [\n    "source",\n    "compound_id",\n    "name",\n    "smiles",\n    "canonical_smiles",\n    "label",\n    "assay_note",\n    "rdkit_formula",\n    "rdkit_mw",\n    "rdkit_hbd",\n    "rdkit_hba",\n    "rdkit_alogp",\n    "rdkit_tpsa",\n]\nDESCRIPTORS = ["rdkit_mw", "rdkit_alogp", "rdkit_hbd", "rdkit_hba", "rdkit_tpsa"]\nLABEL_NAMES = {1: "G4LDB positive", 0: "ChEMBL assumed-negative"}\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument(\n        "--dataset",\n        default="data/modeling/g4ldb_chembl_binary_dataset.csv",\n        help="Combined binary dataset CSV.",\n    )\n    parser.add_argument(\n        "--decoy-report",\n        default="data/chembl/chembl_negative_decoy_report.json",\n        help="JSON report from negative decoy preparation.",\n    )\n    parser.add_argument(\n        "--removed-similar",\n        default="data/chembl/chembl_negative_decoys_removed_similar_to_g4ldb.csv",\n        help="CSV of ChEMBL decoys removed by positive similarity filter.",\n    )\n    parser.add_argument(\n        "--png-output",\n        default="data/modeling/g4ldb_chembl_binary_dataset_overview.png",\n        help="Dashboard PNG output path.",\n    )\n    parser.add_argument(\n        "--summary-output",\n        default="data/modeling/g4ldb_chembl_binary_dataset_visualization_summary.md",\n        help="Markdown summary output path.",\n    )\n    return parser.parse_args()\n\n\ndef import_stack() -> dict[str, Any]:\n    modules: dict[str, Any] = {}\n    missing_required = []\n    for name in ["pandas", "numpy"]:\n        try:\n            modules[name] = __import__(name)\n        except ImportError:\n            missing_required.append(name)\n    if missing_required:\n        raise SystemExit(\n            "Missing required package(s): "\n            + ", ".join(missing_required)\n            + ". Install them to read and summarize the dataset."\n        )\n\n    for name in ["matplotlib.pyplot", "seaborn"]:\n        try:\n            if name == "matplotlib.pyplot":\n                modules["plt"] = __import__("matplotlib.pyplot", fromlist=["pyplot"])\n            else:\n                modules[name] = __import__(name)\n        except ImportError:\n            modules[name] = None\n\n    try:\n        modules["PCA"] = __import__("sklearn.decomposition", fromlist=["PCA"]).PCA\n        modules["StandardScaler"] = __import__(\n            "sklearn.preprocessing", fromlist=["StandardScaler"]\n        ).StandardScaler\n    except ImportError:\n        modules["PCA"] = None\n        modules["StandardScaler"] = None\n    return modules\n\n\ndef read_json(path: Path) -> dict[str, Any]:\n    if not path.exists():\n        return {}\n    with path.open("r", encoding="utf-8") as handle:\n        return json.load(handle)\n\n\ndef format_number(value: Any, digits: int = 3) -> str:\n    if value is None:\n        return "n/a"\n    try:\n        number = float(value)\n    except (TypeError, ValueError):\n        return str(value)\n    if number.is_integer():\n        return f"{int(number):,}"\n    return f"{number:,.{digits}f}".rstrip("0").rstrip(".")\n\n\ndef load_inputs(args: argparse.Namespace, pd: Any) -> tuple[Any, dict[str, Any], Any | None]:\n    dataset_path = Path(args.dataset)\n    if not dataset_path.exists():\n        raise SystemExit(f"Dataset not found: {dataset_path}")\n\n    df = pd.read_csv(dataset_path)\n    for column in ["label", *DESCRIPTORS]:\n        if column in df.columns:\n            df[column] = pd.to_numeric(df[column], errors="coerce")\n    df["label_name"] = df["label"].map(LABEL_NAMES).fillna(df["label"].astype(str))\n\n    report = read_json(Path(args.decoy_report))\n    removed_path = Path(args.removed_similar)\n    removed = pd.read_csv(removed_path) if removed_path.exists() else None\n    return df, report, removed\n\n\ndef validate_dataset(df: Any, report: dict[str, Any], removed: Any | None) -> dict[str, Any]:\n    missing_core = {\n        column: int(df[column].isna().sum()) if column in df.columns else int(len(df))\n        for column in CORE_FIELDS\n    }\n    positive = df[df["label"] == 1]\n    negative = df[df["label"] == 0]\n    positive_smiles = set(positive["canonical_smiles"].dropna())\n    negative_smiles = set(negative["canonical_smiles"].dropna())\n\n    similarity = report.get("similarity_filter", {})\n    kept_summary = similarity.get("kept_similarity_summary", {})\n    removed_summary = similarity.get("removed_similarity_summary", {})\n    selected_negative_max_similarity = kept_summary.get("max")\n    removed_max_similarity = removed_summary.get("max")\n    if removed is not None and "max_positive_tanimoto" in removed.columns:\n        removed_max_similarity = float(removed["max_positive_tanimoto"].max())\n\n    return {\n        "rows": int(len(df)),\n        "label_counts": {int(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},\n        "unique_canonical_smiles": int(df["canonical_smiles"].nunique(dropna=True)),\n        "combined_duplicate_canonical_smiles": int(\n            df.duplicated(subset=["canonical_smiles"], keep=False).sum()\n        ),\n        "negative_duplicate_canonical_smiles": int(\n            negative.duplicated(subset=["canonical_smiles"], keep=False).sum()\n        ),\n        "positive_negative_overlap": int(len(positive_smiles & negative_smiles)),\n        "missing_core_fields": missing_core,\n        "missing_core_total": int(sum(missing_core.values())),\n        "selected_negative_max_similarity": selected_negative_max_similarity,\n        "removed_max_similarity": removed_max_similarity,\n        "similarity_threshold": similarity.get("threshold"),\n        "similarity_kept": similarity.get("kept_candidates"),\n        "similarity_removed": similarity.get("removed_candidates"),\n    }\n\n\ndef descriptor_stats(df: Any, np: Any) -> tuple[Any, Any]:\n    grouped = df.groupby("label")[DESCRIPTORS].agg(["mean", "median", "std"])\n    residual_rows = []\n    for descriptor in DESCRIPTORS:\n        positive_mean = float(grouped.loc[1, (descriptor, "mean")])\n        negative_mean = float(grouped.loc[0, (descriptor, "mean")])\n        positive_std = float(grouped.loc[1, (descriptor, "std")])\n        residual = negative_mean - positive_mean\n        residual_rows.append(\n            {\n                "descriptor": descriptor,\n                "positive_mean": positive_mean,\n                "negative_mean": negative_mean,\n                "mean_residual_negative_minus_positive": residual,\n                "standardized_residual": residual / positive_std\n                if positive_std and not np.isnan(positive_std)\n                else np.nan,\n            }\n        )\n    return grouped, residual_rows\n\n\ndef add_pca_or_scatter(df: Any, modules: dict[str, Any]) -> tuple[Any, str, str, float | None, float | None]:\n    np = modules["numpy"]\n    pd = modules["pandas"]\n    clean = df.dropna(subset=DESCRIPTORS + ["label"]).copy()\n    if modules["PCA"] is None or modules["StandardScaler"] is None:\n        return (\n            clean[["rdkit_mw", "rdkit_alogp", "label", "label_name"]],\n            "rdkit_mw",\n            "rdkit_alogp",\n            None,\n            None,\n        )\n\n    scaled = modules["StandardScaler"]().fit_transform(clean[DESCRIPTORS])\n    pca = modules["PCA"](n_components=2, random_state=0)\n    coords = pca.fit_transform(scaled)\n    pca_df = pd.DataFrame(\n        {\n            "PC1": coords[:, 0],\n            "PC2": coords[:, 1],\n            "label": clean["label"].to_numpy(),\n            "label_name": clean["label_name"].to_numpy(),\n        }\n    )\n    return (\n        pca_df,\n        "PC1",\n        "PC2",\n        float(pca.explained_variance_ratio_[0]),\n        float(pca.explained_variance_ratio_[1]),\n    )\n\n\ndef write_dashboard(\n    df: Any,\n    report: dict[str, Any],\n    residual_rows: list[dict[str, Any]],\n    validation: dict[str, Any],\n    args: argparse.Namespace,\n    modules: dict[str, Any],\n) -> None:\n    plt = modules["plt"]\n    if plt is None:\n        return\n    sns = modules["seaborn"]\n    np = modules["numpy"]\n    pd = modules["pandas"]\n\n    if sns is not None:\n        sns.set_theme(style="whitegrid", context="talk")\n\n    colors = {1: "#2f6f9f", 0: "#c46a3a"}\n    fig, axes = plt.subplots(3, 3, figsize=(18, 15), dpi=210)\n    fig.suptitle("G4LDB Positive vs ChEMBL Assumed-Negative Binary Dataset", fontsize=24, y=0.995)\n\n    counts = df["label"].value_counts().sort_index()\n    axes[0, 0].bar(\n        [LABEL_NAMES.get(int(label), str(label)) for label in counts.index],\n        counts.values,\n        color=[colors.get(int(label), "#777777") for label in counts.index],\n    )\n    axes[0, 0].set_title("Class Balance")\n    axes[0, 0].set_ylabel("Rows")\n    axes[0, 0].tick_params(axis="x", rotation=12)\n    for index, value in enumerate(counts.values):\n        axes[0, 0].text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=12)\n\n    similarity = report.get("similarity_filter", {})\n    kept = validation.get("similarity_kept") or 0\n    removed_count = validation.get("similarity_removed")\n    removed_count = 0 if removed_count is None else removed_count\n    axes[0, 1].bar(["Kept candidates", "Removed similar"], [kept, removed_count], color=["#567568", "#b44747"])\n    axes[0, 1].set_title("Similarity Filter")\n    axes[0, 1].set_ylabel("ChEMBL candidates")\n    threshold = validation.get("similarity_threshold")\n    axes[0, 1].text(\n        0.5,\n        0.92,\n        f"Tanimoto threshold: {format_number(threshold)}",\n        transform=axes[0, 1].transAxes,\n        ha="center",\n        fontsize=12,\n    )\n    for index, value in enumerate([kept, removed_count]):\n        axes[0, 1].text(index, value, f"{int(value):,}", ha="center", va="bottom", fontsize=12)\n\n    residual_df = pd.DataFrame(residual_rows)\n    axes[0, 2].axhline(0, color="#333333", linewidth=1)\n    axes[0, 2].bar(\n        residual_df["descriptor"],\n        residual_df["mean_residual_negative_minus_positive"],\n        color=["#567568" if value >= 0 else "#b44747" for value in residual_df["mean_residual_negative_minus_positive"]],\n    )\n    axes[0, 2].set_title("Descriptor Mean Residuals")\n    axes[0, 2].set_ylabel("Negative mean - positive mean")\n    axes[0, 2].tick_params(axis="x", rotation=25)\n\n    pca_df, x_col, y_col, pc1_var, pc2_var = add_pca_or_scatter(df, modules)\n    for label in [1, 0]:\n        subset = pca_df[pca_df["label"] == label]\n        axes[1, 0].scatter(\n            subset[x_col],\n            subset[y_col],\n            s=10,\n            alpha=0.42,\n            color=colors[label],\n            label=LABEL_NAMES[label],\n            edgecolors="none",\n        )\n    scatter_title = "Descriptor PCA"\n    if pc1_var is not None and pc2_var is not None:\n        scatter_title += f" ({(pc1_var + pc2_var) * 100:.1f}% variance)"\n        axes[1, 0].set_xlabel(f"PC1 ({pc1_var * 100:.1f}%)")\n        axes[1, 0].set_ylabel(f"PC2 ({pc2_var * 100:.1f}%)")\n    else:\n        scatter_title = "Descriptor Scatter Fallback"\n        axes[1, 0].set_xlabel("Molecular weight")\n        axes[1, 0].set_ylabel("LogP")\n    axes[1, 0].set_title(scatter_title)\n    axes[1, 0].legend(frameon=False, fontsize=11, loc="best")\n\n    descriptor_axes = [axes[1, 1], axes[1, 2], axes[2, 0], axes[2, 1], axes[2, 2]]\n    for axis, descriptor in zip(descriptor_axes, DESCRIPTORS):\n        clean = df.dropna(subset=[descriptor])\n        low, high = np.nanpercentile(clean[descriptor], [1, 99])\n        clipped = clean[(clean[descriptor] >= low) & (clean[descriptor] <= high)]\n        if sns is not None:\n            sns.histplot(\n                data=clipped,\n                x=descriptor,\n                hue="label_name",\n                hue_order=[LABEL_NAMES[1], LABEL_NAMES[0]],\n                bins=38,\n                stat="density",\n                common_norm=False,\n                element="step",\n                fill=False,\n                palette=[colors[1], colors[0]],\n                ax=axis,\n                legend=False,\n            )\n        else:\n            for label in [1, 0]:\n                subset = clipped[clipped["label"] == label]\n                axis.hist(\n                    subset[descriptor],\n                    bins=38,\n                    density=True,\n                    histtype="step",\n                    linewidth=1.7,\n                    color=colors[label],\n                    label=LABEL_NAMES[label],\n                )\n        axis.set_title(descriptor)\n        axis.set_ylabel("Density")\n        axis.set_xlabel("")\n        axis.text(\n            0.02,\n            0.95,\n            "1st-99th pct display",\n            transform=axis.transAxes,\n            va="top",\n            fontsize=10,\n            color="#555555",\n        )\n\n    fig.text(\n        0.01,\n        0.01,\n        "Validation: "\n        f"duplicates={validation[\'combined_duplicate_canonical_smiles\']:,}; "\n        f"positive-negative overlap={validation[\'positive_negative_overlap\']:,}; "\n        f"missing core values={validation[\'missing_core_total\']:,}; "\n        f"max selected negative similarity={format_number(validation[\'selected_negative_max_similarity\'])}",\n        fontsize=12,\n        color="#333333",\n    )\n    fig.tight_layout(rect=[0, 0.025, 1, 0.97])\n    output_path = Path(args.png_output)\n    output_path.parent.mkdir(parents=True, exist_ok=True)\n    fig.savefig(output_path, bbox_inches="tight")\n    with warnings.catch_warnings():\n        warnings.simplefilter("ignore")\n        plt.close(fig)\n\n\ndef write_summary(\n    df: Any,\n    grouped: Any,\n    residual_rows: list[dict[str, Any]],\n    validation: dict[str, Any],\n    args: argparse.Namespace,\n) -> None:\n    lines = [\n        "# G4LDB/ChEMBL Binary Dataset Visualization Summary",\n        "",\n        f"Generated: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",\n        "",\n        f"Source dataset: `{args.dataset}`",\n        "",\n        f"Dashboard: `{args.png_output}`",\n        "",\n        "## Validation",\n        "",\n        f"- Total rows: {validation[\'rows\']:,}",\n        f"- Label counts: 1={validation[\'label_counts\'].get(1, 0):,}, 0={validation[\'label_counts\'].get(0, 0):,}",\n        f"- Unique canonical SMILES: {validation[\'unique_canonical_smiles\']:,}",\n        f"- Duplicate canonical SMILES in combined dataset: {validation[\'combined_duplicate_canonical_smiles\']:,}",\n        f"- Duplicate canonical SMILES among negatives: {validation[\'negative_duplicate_canonical_smiles\']:,}",\n        f"- Positive-negative canonical SMILES overlap: {validation[\'positive_negative_overlap\']:,}",\n        f"- Missing core field values: {validation[\'missing_core_total\']:,}",\n        f"- Max selected negative similarity to any G4LDB positive: {format_number(validation[\'selected_negative_max_similarity\'])}",\n        f"- Similarity filter kept/removed candidates: {format_number(validation[\'similarity_kept\'])}/{format_number(validation[\'similarity_removed\'])}",\n        f"- Max removed candidate similarity: {format_number(validation[\'removed_max_similarity\'])}",\n        "",\n        "## Descriptor Means",\n        "",\n        "| Descriptor | G4LDB positive mean | ChEMBL assumed-negative mean | Negative - positive |",\n        "|---|---:|---:|---:|",\n    ]\n    for row in residual_rows:\n        lines.append(\n            "| {descriptor} | {pos} | {neg} | {resid} |".format(\n                descriptor=row["descriptor"],\n                pos=format_number(row["positive_mean"]),\n                neg=format_number(row["negative_mean"]),\n                resid=format_number(row["mean_residual_negative_minus_positive"]),\n            )\n        )\n\n    lines.extend(["", "## Descriptor Medians", "", "| Descriptor | G4LDB positive median | ChEMBL assumed-negative median |", "|---|---:|---:|"])\n    for descriptor in DESCRIPTORS:\n        lines.append(\n            "| {descriptor} | {pos} | {neg} |".format(\n                descriptor=descriptor,\n                pos=format_number(grouped.loc[1, (descriptor, "median")]),\n                neg=format_number(grouped.loc[0, (descriptor, "median")]),\n            )\n        )\n\n    missing_nonzero = {\n        column: count for column, count in validation["missing_core_fields"].items() if count\n    }\n    if missing_nonzero:\n        lines.extend(["", "## Missing Core Fields", ""])\n        for column, count in missing_nonzero.items():\n            lines.append(f"- {column}: {count:,}")\n\n    lines.extend(\n        [\n            "",\n            "## Notes",\n            "",\n            "- ChEMBL label 0 rows are assumed negatives after G4LDB similarity filtering, not experimentally confirmed non-binders.",\n            "- Histogram panels clip to the 1st-99th percentile only for display readability; tabulated values use the full dataset.",\n        ]\n    )\n\n    output_path = Path(args.summary_output)\n    output_path.parent.mkdir(parents=True, exist_ok=True)\n    output_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")\n\n\ndef main() -> int:\n    args = parse_args()\n    modules = import_stack()\n    pd = modules["pandas"]\n    np = modules["numpy"]\n\n    df, report, removed = load_inputs(args, pd)\n    validation = validate_dataset(df, report, removed)\n    grouped, residual_rows = descriptor_stats(df, np)\n    write_dashboard(df, report, residual_rows, validation, args, modules)\n    write_summary(df, grouped, residual_rows, validation, args)\n\n    print(f"Rows: {validation[\'rows\']:,}")\n    print(\n        "Label counts: "\n        + ", ".join(f"{label}={count:,}" for label, count in validation["label_counts"].items())\n    )\n    print(f"Positive-negative canonical overlap: {validation[\'positive_negative_overlap\']:,}")\n    print(f"Missing core field values: {validation[\'missing_core_total\']:,}")\n    print(f"PNG: {args.png_output}")\n    print(f"Summary: {args.summary_output}")\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'


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
