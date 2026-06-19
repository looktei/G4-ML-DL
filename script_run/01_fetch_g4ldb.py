#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Fetch G4LDB ligand metadata/activity records."""

STEP_ID = '01'
STEP_TITLE = 'Fetch G4LDB ligand metadata/activity records'
EMBEDDED_SOURCE_NAME = '02_fetch_g4ldb.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Fetch G4LDB 3.x ligand metadata and activity records.\n\nOutputs are written under data/g4ldb/ by default. The script uses only the\nPython standard library so it can run in lightweight research environments.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport csv\nimport json\nimport random\nimport sys\nimport time\nfrom collections import defaultdict\nfrom concurrent.futures import ThreadPoolExecutor, as_completed\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nfrom typing import Any\nfrom urllib.error import HTTPError, URLError\nfrom urllib.parse import urlencode\nfrom urllib.request import Request, urlopen\n\n\nBASE_URL = "https://www.g4ldb.com:8088"\nPAGE_SIZE = 200\nUSER_AGENT = "Nipah-G4-ligand-prioritization-data-pull/1.0"\n\nLIGAND_FIELDS = [\n    "ligandId",\n    "nameRef",\n    "nameStd",\n    "smiles",\n    "pubchemId",\n    "formula",\n    "mw",\n    "hbd",\n    "hba",\n    "alogp",\n    "solubility",\n    "synonyms",\n    "isNew",\n    "complex",\n    "year",\n]\n\nML_FIELDS = [\n    "ligandId",\n    "name",\n    "smiles",\n    "pubchemId",\n    "formula",\n    "mw",\n    "hbd",\n    "hba",\n    "alogp",\n    "solubility",\n    "year",\n    "n_activity_records",\n    "activity1_set",\n    "activity2_set",\n    "sequence_examples",\n    "doi_examples",\n    "label",\n]\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--base-url", default=BASE_URL)\n    parser.add_argument("--out-dir", default="data/g4ldb")\n    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)\n    parser.add_argument("--timeout", type=float, default=30.0)\n    parser.add_argument("--retries", type=int, default=3)\n    parser.add_argument("--sleep-min", type=float, default=0.05)\n    parser.add_argument("--sleep-max", type=float, default=0.15)\n    parser.add_argument(\n        "--workers",\n        type=int,\n        default=8,\n        help="Concurrent workers for ligand activity/detail requests.",\n    )\n    return parser.parse_args()\n\n\ndef request_json(\n    base_url: str,\n    path: str,\n    params: dict[str, Any],\n    timeout: float,\n    retries: int,\n) -> dict[str, Any]:\n    url = f"{base_url.rstrip(\'/\')}{path}?{urlencode(params)}"\n    last_error: Exception | None = None\n    for attempt in range(retries):\n        try:\n            req = Request(url, headers={"User-Agent": USER_AGENT})\n            with urlopen(req, timeout=timeout) as response:\n                payload = response.read().decode("utf-8")\n            return json.loads(payload)\n        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:\n            last_error = exc\n            if attempt + 1 >= retries:\n                break\n            time.sleep(0.5 * (2**attempt))\n    raise RuntimeError(f"failed to fetch {url}: {last_error}") from last_error\n\n\ndef atomic_write_text(path: Path, text: str) -> None:\n    tmp = path.with_suffix(path.suffix + ".tmp")\n    tmp.write_text(text, encoding="utf-8")\n    tmp.replace(path)\n\n\ndef write_json(path: Path, data: Any) -> None:\n    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\\n")\n\n\ndef write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:\n    tmp = path.with_suffix(path.suffix + ".tmp")\n    with tmp.open("w", encoding="utf-8") as handle:\n        for row in rows:\n            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\\n")\n    tmp.replace(path)\n\n\ndef write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:\n    tmp = path.with_suffix(path.suffix + ".tmp")\n    with tmp.open("w", encoding="utf-8", newline="") as handle:\n        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")\n        writer.writeheader()\n        for row in rows:\n            writer.writerow({field: csv_value(row.get(field)) for field in fields})\n    tmp.replace(path)\n\n\ndef csv_value(value: Any) -> Any:\n    if value is None:\n        return ""\n    if isinstance(value, (list, dict)):\n        return json.dumps(value, ensure_ascii=False, sort_keys=True)\n    return value\n\n\ndef ordered_union_fields(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:\n    seen = set(preferred)\n    fields = list(preferred)\n    for row in rows:\n        for key in row:\n            if key not in seen:\n                seen.add(key)\n                fields.append(key)\n    return fields\n\n\ndef first_nonempty(*values: Any) -> Any:\n    for value in values:\n        if value is not None and value != "":\n            return value\n    return ""\n\n\ndef unique_examples(values: list[Any], limit: int = 5) -> str:\n    seen: set[str] = set()\n    examples: list[str] = []\n    for value in values:\n        if value is None or value == "":\n            continue\n        text = str(value)\n        if text in seen:\n            continue\n        seen.add(text)\n        examples.append(text)\n        if len(examples) >= limit:\n            break\n    return "|".join(examples)\n\n\ndef fetch_ligands(args: argparse.Namespace) -> list[dict[str, Any]]:\n    ligands: list[dict[str, Any]] = []\n    offset = 0\n    while True:\n        payload = request_json(\n            args.base_url,\n            "/ligand/export",\n            {"n": offset, "s": args.page_size},\n            args.timeout,\n            args.retries,\n        )\n        rows = payload.get("data") or []\n        if not isinstance(rows, list):\n            raise RuntimeError(f"unexpected /ligand/export data at offset {offset}: {type(rows)}")\n        if not rows:\n            break\n        ligands.extend(rows)\n        print(f"metadata offset {offset}: {len(rows)} rows", flush=True)\n        offset += args.page_size\n    return ligands\n\n\ndef fetch_activity_one(args: argparse.Namespace, ligand_id: str) -> list[dict[str, Any]]:\n    payload = request_json(\n        args.base_url,\n        "/ligand/get",\n        {"ligandId": ligand_id},\n        args.timeout,\n        args.retries,\n    )\n    rows = payload.get("data") or []\n    if not isinstance(rows, list):\n        raise RuntimeError(f"unexpected /ligand/get data for {ligand_id}: {type(rows)}")\n    time.sleep(random.uniform(args.sleep_min, args.sleep_max))\n    return rows\n\n\ndef fetch_activity(\n    args: argparse.Namespace, ligand_ids: list[str]\n) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:\n    activity_rows: list[dict[str, Any]] = []\n    errors: list[dict[str, Any]] = []\n    completed = 0\n    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:\n        futures = {\n            executor.submit(fetch_activity_one, args, ligand_id): ligand_id\n            for ligand_id in ligand_ids\n        }\n        for future in as_completed(futures):\n            ligand_id = futures[future]\n            completed += 1\n            try:\n                activity_rows.extend(future.result())\n            except Exception as exc:  # Keep successful records even if one ligand fails.\n                errors.append({"ligandId": ligand_id, "error": repr(exc)})\n            if completed == 1 or completed % 250 == 0 or completed == len(ligand_ids):\n                print(\n                    f"activity {completed}/{len(ligand_ids)} ligands: "\n                    f"{len(activity_rows)} rows collected, {len(errors)} errors",\n                    flush=True,\n                )\n    return activity_rows, errors\n\n\ndef build_ml_ready(\n    ligands: list[dict[str, Any]], activity_rows: list[dict[str, Any]]\n) -> list[dict[str, Any]]:\n    by_ligand: dict[str, list[dict[str, Any]]] = defaultdict(list)\n    for row in activity_rows:\n        ligand_id = row.get("ligandId")\n        if ligand_id:\n            by_ligand[str(ligand_id)].append(row)\n\n    ml_rows: list[dict[str, Any]] = []\n    for ligand in ligands:\n        ligand_id = str(ligand.get("ligandId") or "")\n        rows = by_ligand.get(ligand_id, [])\n        first = rows[0] if rows else {}\n        name = first_nonempty(\n            ligand.get("nameStd"),\n            ligand.get("nameRef"),\n            first.get("nameStd"),\n            first.get("nameRef"),\n            ligand.get("synonyms"),\n            ligand_id,\n        )\n        ml_rows.append(\n            {\n                "ligandId": ligand_id,\n                "name": name,\n                "smiles": first_nonempty(ligand.get("smiles"), first.get("smiles")),\n                "pubchemId": first_nonempty(ligand.get("pubchemId"), first.get("pubchemId")),\n                "formula": first_nonempty(ligand.get("formula"), first.get("formula")),\n                "mw": first_nonempty(ligand.get("mw"), first.get("mw")),\n                "hbd": first_nonempty(ligand.get("hbd"), first.get("hbd")),\n                "hba": first_nonempty(ligand.get("hba"), first.get("hba")),\n                "alogp": first_nonempty(ligand.get("alogp"), first.get("alogp")),\n                "solubility": first_nonempty(ligand.get("solubility"), first.get("solubility")),\n                "year": first_nonempty(ligand.get("year"), first.get("year")),\n                "n_activity_records": len(rows),\n                "activity1_set": "|".join(\n                    sorted({str(row.get("activity1")) for row in rows if row.get("activity1")})\n                ),\n                "activity2_set": "|".join(\n                    sorted({str(row.get("activity2")) for row in rows if row.get("activity2")})\n                ),\n                "sequence_examples": unique_examples([row.get("sequence") for row in rows]),\n                "doi_examples": unique_examples([row.get("doi") for row in rows]),\n                "label": 1,\n            }\n        )\n    return ml_rows\n\n\ndef write_readme(out_dir: Path, counts: dict[str, int], retrieved_at: str) -> None:\n    text = f"""# G4LDB 3.x Ligand Dataset\n\nRetrieved: {retrieved_at}\n\nSource: G4LDB backend at `https://www.g4ldb.com:8088/`.\n\n## Source endpoints\n\n- Ligand metadata: `GET /ligand/export?n=<offset>&s=200`, paginated from offset 0 until an empty page.\n- Ligand activity/details: `GET /ligand/get?ligandId=<id>`, one request per ligand.\n\n## Files\n\n- `g4ldb_ligands_raw.json`: raw ligand metadata list from `/ligand/export`.\n- `g4ldb_ligands.csv`: metadata table with ligand identifiers, names, SMILES, PubChem IDs, formulae, molecular descriptors, and year where available.\n- `g4ldb_activity_raw.jsonl`: raw activity/detail rows from `/ligand/get`, one JSON object per line.\n- `g4ldb_activity.csv`: activity/detail table including ligand, assay/activity, sequence, method, value, publication, DOI, and related fields.\n- `g4ldb_g4_ligand_ml_ready.csv`: positive ligand table for downstream ML use. Each row is a G4LDB ligand with `label=1` and aggregated activity, sequence, and DOI examples.\n\n## Row counts\n\n- Ligands: {counts["ligands"]}\n- Activity/detail records: {counts["activity"]}\n- ML-ready positive ligand rows: {counts["ml_ready"]}\n\n## ML-ready fields\n\n`ligandId`, `name`, `smiles`, `pubchemId`, `formula`, `mw`, `hbd`, `hba`, `alogp`, `solubility`, `year`, `n_activity_records`, `activity1_set`, `activity2_set`, `sequence_examples`, `doi_examples`, `label`.\n\n## Intended use\n\nThis dataset is prepared for Aim 2 of the Nipah virus G-quadruplex deep learning / ligand prioritization project. It provides known G-quadruplex ligands as positive examples and ligand descriptors for model training, candidate filtering, and prioritization. It does not include negative labels; negative or background molecules should be assembled separately with clear provenance.\n"""\n    atomic_write_text(out_dir / "README.md", text)\n\n\ndef main() -> int:\n    args = parse_args()\n    out_dir = Path(args.out_dir)\n    out_dir.mkdir(parents=True, exist_ok=True)\n\n    retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()\n\n    ligands = fetch_ligands(args)\n    ligand_ids = [str(row["ligandId"]) for row in ligands if row.get("ligandId")]\n    if len(ligand_ids) != len(set(ligand_ids)):\n        print("warning: duplicate ligandId values found in metadata", file=sys.stderr)\n\n    activity_rows, activity_errors = fetch_activity(args, ligand_ids)\n    ml_rows = build_ml_ready(ligands, activity_rows)\n\n    ligand_fields = ordered_union_fields(ligands, LIGAND_FIELDS)\n    activity_fields = ordered_union_fields(activity_rows, [])\n\n    write_json(out_dir / "g4ldb_ligands_raw.json", ligands)\n    write_csv(out_dir / "g4ldb_ligands.csv", ligands, ligand_fields)\n    write_jsonl(out_dir / "g4ldb_activity_raw.jsonl", activity_rows)\n    write_csv(out_dir / "g4ldb_activity.csv", activity_rows, activity_fields)\n    write_csv(out_dir / "g4ldb_g4_ligand_ml_ready.csv", ml_rows, ML_FIELDS)\n    write_json(out_dir / "g4ldb_activity_errors.json", activity_errors)\n\n    counts = {\n        "ligands": len(ligands),\n        "activity": len(activity_rows),\n        "ml_ready": len(ml_rows),\n    }\n    write_readme(out_dir, counts, retrieved_at)\n\n    print("\\nValidation summary")\n    print(f"ligands: {counts[\'ligands\']} rows")\n    print(f"activity/detail records: {counts[\'activity\']} rows")\n    print(f"activity/detail errors: {len(activity_errors)}")\n    print(f"ml-ready positives: {counts[\'ml_ready\']} rows")\n    print(f"ligand CSV first columns: {\', \'.join(ligand_fields[:8])}")\n    print(f"activity CSV first columns: {\', \'.join(activity_fields[:8])}")\n    print(f"ml-ready first columns: {\', \'.join(ML_FIELDS[:8])}")\n    if ml_rows:\n        sample = {field: ml_rows[0].get(field) for field in ML_FIELDS[:6]}\n        print("sample ml-ready row:", json.dumps(sample, ensure_ascii=False))\n\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'


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
