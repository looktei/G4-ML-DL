#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Preprocess G4LDB positives into ML-ready table."""

STEP_ID = '02'
STEP_TITLE = 'Preprocess G4LDB positives into ML-ready table'
EMBEDDED_SOURCE_NAME = '03_preprocess_g4ldb_ml_ready.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Preprocess the G4LDB ML-ready positive ligand table.\n\nThe raw ML-ready file intentionally preserves G4LDB metadata. This script\ncreates a cleaner model-input table by validating SMILES, adding canonical\nSMILES/RDKit descriptors, removing duplicate structures, and writing an audit\nreport for removed rows and missing metadata.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nfrom typing import Any\n\nimport pandas as pd\nfrom rdkit import Chem\nfrom rdkit import RDLogger\nfrom rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors\n\n\nRDLogger.DisableLog("rdApp.*")\n\nREQUIRED_COLUMNS = ["ligandId", "name", "smiles", "label", "n_activity_records"]\nOPTIONAL_COLUMNS = [\n    "pubchemId",\n    "formula",\n    "mw",\n    "hbd",\n    "hba",\n    "alogp",\n    "solubility",\n    "year",\n    "activity1_set",\n    "activity2_set",\n    "sequence_examples",\n    "doi_examples",\n]\nOUTPUT_COLUMNS = [\n    "ligandId",\n    "name",\n    "smiles",\n    "canonical_smiles",\n    "label",\n    "n_activity_records",\n    "year",\n    "pubchemId",\n    "formula",\n    "mw",\n    "hbd",\n    "hba",\n    "alogp",\n    "solubility",\n    "rdkit_formula",\n    "rdkit_mw",\n    "rdkit_hbd",\n    "rdkit_hba",\n    "rdkit_alogp",\n    "activity1_set",\n    "activity2_set",\n    "sequence_examples",\n    "doi_examples",\n]\nCORE_OUTPUT_COLUMNS = [\n    "ligandId",\n    "name",\n    "smiles",\n    "canonical_smiles",\n    "label",\n    "n_activity_records",\n    "year",\n    "rdkit_formula",\n    "rdkit_mw",\n    "rdkit_hbd",\n    "rdkit_hba",\n    "rdkit_alogp",\n    "activity1_set",\n    "activity2_set",\n    "doi_examples",\n]\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument(\n        "--input",\n        default="data/g4ldb/g4ldb_g4_ligand_ml_ready.csv",\n        help="Input ML-ready CSV from fetch_g4ldb.py.",\n    )\n    parser.add_argument(\n        "--output",\n        default="data/g4ldb/g4ldb_g4_ligand_ml_preprocessed.csv",\n        help="Cleaned core ML CSV output path with no missing values.",\n    )\n    parser.add_argument(\n        "--metadata-output",\n        default="data/g4ldb/g4ldb_g4_ligand_ml_preprocessed_with_metadata.csv",\n        help="Cleaned CSV output path that retains optional metadata columns.",\n    )\n    parser.add_argument(\n        "--removed-output",\n        default="data/g4ldb/g4ldb_g4_ligand_ml_removed_rows.csv",\n        help="Audit CSV for removed rows.",\n    )\n    parser.add_argument(\n        "--report",\n        default="data/g4ldb/g4ldb_g4_ligand_ml_preprocessing_report.json",\n        help="JSON preprocessing report path.",\n    )\n    return parser.parse_args()\n\n\ndef normalize_empty_strings(df: pd.DataFrame) -> pd.DataFrame:\n    clean = df.copy()\n    for column in clean.columns:\n        if clean[column].dtype == "object":\n            clean[column] = clean[column].map(\n                lambda value: value.strip() if isinstance(value, str) else value\n            )\n    return clean.replace(r"^\\s*$", pd.NA, regex=True)\n\n\ndef canonicalize_smiles(smiles: Any) -> tuple[str | None, dict[str, Any] | None]:\n    if pd.isna(smiles):\n        return None, None\n    mol = Chem.MolFromSmiles(str(smiles))\n    if mol is None:\n        return None, None\n    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)\n    descriptors = {\n        "rdkit_formula": rdMolDescriptors.CalcMolFormula(mol),\n        "rdkit_mw": round(float(Descriptors.MolWt(mol)), 4),\n        "rdkit_hbd": int(Lipinski.NumHDonors(mol)),\n        "rdkit_hba": int(Lipinski.NumHAcceptors(mol)),\n        "rdkit_alogp": round(float(Crippen.MolLogP(mol)), 4),\n    }\n    return canonical, descriptors\n\n\ndef add_smiles_features(df: pd.DataFrame) -> pd.DataFrame:\n    rows = []\n    for _, row in df.iterrows():\n        canonical, descriptors = canonicalize_smiles(row["smiles"])\n        enriched = row.to_dict()\n        enriched["canonical_smiles"] = canonical\n        if descriptors:\n            enriched.update(descriptors)\n        else:\n            for column in ["rdkit_formula", "rdkit_mw", "rdkit_hbd", "rdkit_hba", "rdkit_alogp"]:\n                enriched[column] = pd.NA\n        rows.append(enriched)\n    return pd.DataFrame(rows)\n\n\ndef coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:\n    clean = df.copy()\n    for column in [\n        "label",\n        "n_activity_records",\n        "year",\n        "mw",\n        "hbd",\n        "hba",\n        "alogp",\n        "solubility",\n        "rdkit_mw",\n        "rdkit_hbd",\n        "rdkit_hba",\n        "rdkit_alogp",\n    ]:\n        if column in clean.columns:\n            clean[column] = pd.to_numeric(clean[column], errors="coerce")\n    return clean\n\n\ndef main() -> int:\n    args = parse_args()\n    input_path = Path(args.input)\n    output_path = Path(args.output)\n    metadata_output_path = Path(args.metadata_output)\n    removed_path = Path(args.removed_output)\n    report_path = Path(args.report)\n\n    df_raw = pd.read_csv(input_path, dtype=str, keep_default_na=False)\n    df = normalize_empty_strings(df_raw)\n\n    missing_before = {column: int(df[column].isna().sum()) for column in df.columns}\n\n    missing_required_mask = df[REQUIRED_COLUMNS].isna().any(axis=1)\n    removed_parts = []\n    if missing_required_mask.any():\n        missing_required = df.loc[missing_required_mask].copy()\n        missing_required["removal_reason"] = "missing_required_field"\n        removed_parts.append(missing_required)\n    df = df.loc[~missing_required_mask].copy()\n\n    df = add_smiles_features(df)\n    invalid_smiles_mask = df["canonical_smiles"].isna()\n    if invalid_smiles_mask.any():\n        invalid_smiles = df.loc[invalid_smiles_mask].copy()\n        invalid_smiles["removal_reason"] = "invalid_smiles"\n        removed_parts.append(invalid_smiles)\n    df = df.loc[~invalid_smiles_mask].copy()\n    df = coerce_numeric(df)\n\n    df["n_activity_records"] = df["n_activity_records"].fillna(0).astype(int)\n    df["label"] = df["label"].fillna(1).astype(int)\n\n    df = df.sort_values(\n        by=["canonical_smiles", "n_activity_records", "ligandId"],\n        ascending=[True, False, True],\n        kind="mergesort",\n    )\n    duplicate_mask = df.duplicated(subset=["canonical_smiles"], keep="first")\n    if duplicate_mask.any():\n        duplicate_rows = df.loc[duplicate_mask].copy()\n        duplicate_rows["removal_reason"] = "duplicate_canonical_smiles"\n        removed_parts.append(duplicate_rows)\n    df_clean = df.loc[~duplicate_mask].copy()\n    df_clean = df_clean.sort_values("ligandId", kind="mergesort").reset_index(drop=True)\n\n    for column in OPTIONAL_COLUMNS + ["canonical_smiles"]:\n        if column not in df_clean.columns:\n            df_clean[column] = pd.NA\n\n    metadata_columns = [column for column in OUTPUT_COLUMNS if column in df_clean.columns]\n    df_metadata = df_clean[metadata_columns].copy()\n\n    core_columns = [column for column in CORE_OUTPUT_COLUMNS if column in df_clean.columns]\n    df_core = df_clean[core_columns].copy()\n    core_missing_mask = df_core.isna().any(axis=1)\n    if core_missing_mask.any():\n        core_missing = df_clean.loc[core_missing_mask].copy()\n        core_missing["removal_reason"] = "missing_core_ml_field"\n        removed_parts.append(core_missing)\n        df_core = df_core.loc[~core_missing_mask].copy()\n\n    output_path.parent.mkdir(parents=True, exist_ok=True)\n    df_core.to_csv(output_path, index=False)\n    df_metadata.to_csv(metadata_output_path, index=False)\n\n    if removed_parts:\n        removed = pd.concat(removed_parts, ignore_index=True, sort=False)\n    else:\n        removed = pd.DataFrame(columns=list(df_raw.columns) + ["removal_reason"])\n    removed.to_csv(removed_path, index=False)\n\n    report = {\n        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),\n        "input": str(input_path),\n        "output": str(output_path),\n        "metadata_output": str(metadata_output_path),\n        "removed_rows_output": str(removed_path),\n        "input_rows": int(len(df_raw)),\n        "output_rows": int(len(df_core)),\n        "metadata_output_rows": int(len(df_metadata)),\n        "removed_rows": int(len(removed)),\n        "removed_by_reason": {\n            key: int(value)\n            for key, value in removed["removal_reason"].value_counts(dropna=False).items()\n        },\n        "missing_before": missing_before,\n        "missing_after_core_output": {\n            column: int(df_core[column].isna().sum()) for column in df_core.columns\n        },\n        "missing_after_metadata_output": {\n            column: int(df_metadata[column].isna().sum()) for column in df_metadata.columns\n        },\n        "duplicate_rows_before": int(df_raw.duplicated().sum()),\n        "duplicate_ligandId_before": int(df_raw["ligandId"].duplicated().sum()),\n        "duplicate_smiles_before": int(df_raw["smiles"].duplicated().sum()),\n        "duplicate_canonical_smiles_after": int(df_core["canonical_smiles"].duplicated().sum()),\n        "required_columns": REQUIRED_COLUMNS,\n        "core_output_columns": CORE_OUTPUT_COLUMNS,\n        "deduplication_rule": (\n            "Canonicalize SMILES with RDKit, then keep the row with the highest "\n            "n_activity_records per canonical_smiles; ties are resolved by ligandId."\n        ),\n        "notes": [\n            "Rows missing optional metadata such as pubchemId or sequence_examples are retained.",\n            "RDKit descriptor columns are added so model inputs do not depend on missing G4LDB descriptor fields.",\n            "The core output excludes optional columns with missing values; the metadata output preserves them.",\n            "The original g4ldb_g4_ligand_ml_ready.csv is preserved unchanged.",\n        ],\n    }\n    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")\n\n    print(json.dumps(report, indent=2, ensure_ascii=False))\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'


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
