#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Collect candidate PubChem G4 BioAssay AIDs."""

STEP_ID = '20'
STEP_TITLE = 'Collect candidate PubChem G4 BioAssay AIDs'
EMBEDDED_SOURCE_NAME = '21_collect_pubchem_g4_bioassay_aids.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Collect and curate candidate PubChem BioAssay AIDs for G4 external validation.\n\nThis checkpoint stops at assay-level metadata and rough active/inactive counts.\nIt deliberately does not download compound-level assay rows or run model prediction.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport csv\nimport json\nimport re\nimport time\nfrom collections import defaultdict\nfrom datetime import datetime\nfrom pathlib import Path\nfrom typing import Any\nfrom urllib.parse import urlencode\nfrom urllib.request import Request, urlopen\n\n\nQUERY_TERMS = [\n    "G-quadruplex",\n    "G4 DNA",\n    "telomeric G-quadruplex",\n    "c-MYC G-quadruplex",\n    "BCL2 G-quadruplex",\n    "KRAS G-quadruplex",\n    "VEGF G-quadruplex",\n]\n\nOUTPUT_COLUMNS = [\n    "query_term",\n    "aid",\n    "title_name",\n    "source",\n    "source_id",\n    "method",\n    "assay_target",\n    "description_snippet",\n    "comment",\n    "pubchem_url",\n    "active_count",\n    "inactive_count",\n    "inconclusive_count",\n    "total_count",\n    "count_basis",\n    "curation_label",\n    "curation_evidence",\n    "matched_keywords",\n    "excluded_reason",\n    "deposit_date",\n    "modify_date",\n]\n\nEXCLUSION_PATTERNS = [\n    r"complex spheroids?",\n    r"cell line growth inhibition",\n    r"cell viability",\n    r"anticancer human tumor",\n    r"drug combination",\n    r"combination screen",\n]\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument(\n        "--output-dir",\n        default="data/pubchem_g4_bioassay_external_validation",\n        help="Directory for candidate/selected tables and checkpoint report.",\n    )\n    parser.add_argument(\n        "--retmax-per-query",\n        type=int,\n        default=75,\n        help="Maximum NCBI pcassay search hits to retrieve per query term.",\n    )\n    parser.add_argument(\n        "--batch-size",\n        type=int,\n        default=25,\n        help="AID batch size for NCBI ESummary and PubChem PUG-REST summaries.",\n    )\n    parser.add_argument("--sleep", type=float, default=0.35, help="Delay between official API calls.")\n    parser.add_argument("--timeout", type=int, default=45, help="HTTP timeout in seconds.")\n    return parser.parse_args()\n\n\ndef root_from_script() -> Path:\n    return Path(__file__).resolve().parents[1]\n\n\ndef resolve(path: str | Path, root: Path) -> Path:\n    path = Path(path)\n    return path if path.is_absolute() else root / path\n\n\ndef fetch_json(url: str, timeout: int, retries: int = 3) -> dict[str, Any]:\n    last_error: Exception | None = None\n    for attempt in range(retries):\n        try:\n            request = Request(url, headers={"User-Agent": "Nipah-G4-PubChem-AID-checkpoint/1.0"})\n            with urlopen(request, timeout=timeout) as response:\n                return json.loads(response.read().decode("utf-8"))\n        except Exception as exc:  # noqa: BLE001 - keep exact API blocker for report.\n            last_error = exc\n            time.sleep(1.5 * (attempt + 1))\n    raise RuntimeError(f"Failed to fetch {url}: {last_error}")\n\n\ndef eutils_url(endpoint: str, params: dict[str, Any]) -> str:\n    return f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{endpoint}.fcgi?{urlencode(params)}"\n\n\ndef chunks(items: list[int], size: int) -> list[list[int]]:\n    return [items[i : i + size] for i in range(0, len(items), size)]\n\n\ndef flat_text(value: Any) -> str:\n    if value is None:\n        return ""\n    if isinstance(value, str):\n        return value\n    if isinstance(value, list):\n        return " ".join(flat_text(item) for item in value)\n    if isinstance(value, dict):\n        return " ".join(flat_text(item) for item in value.values())\n    return str(value)\n\n\ndef compact(text: str, limit: int = 750) -> str:\n    text = re.sub(r"\\s+", " ", text or "").strip()\n    return text[: limit - 3] + "..." if len(text) > limit else text\n\n\ndef first_sentence_with_keyword(text: str, keywords: list[str], limit: int = 500) -> str:\n    cleaned = re.sub(r"\\s+", " ", text or "").strip()\n    if not cleaned:\n        return ""\n    parts = re.split(r"(?<=[.!?])\\s+", cleaned)\n    lowered_keywords = [keyword.lower() for keyword in keywords]\n    for part in parts:\n        low = part.lower()\n        if any(keyword in low for keyword in lowered_keywords):\n            return compact(part, limit)\n    return compact(cleaned, limit)\n\n\ndef count_from(summary: dict[str, Any], key: str) -> int | None:\n    value = summary.get(f"CIDCount{key}")\n    if value in (None, ""):\n        value = summary.get(f"SIDCount{key}")\n    if value in (None, ""):\n        return None\n    try:\n        return int(value)\n    except (TypeError, ValueError):\n        return None\n\n\ndef classify(summary: dict[str, Any]) -> tuple[str, str, str, str]:\n    title = summary.get("Name") or summary.get("assayname") or ""\n    description = flat_text(summary.get("Description") or summary.get("assaydescription"))\n    protocol = flat_text(summary.get("Protocol"))\n    comment = flat_text(summary.get("Comment"))\n    target = flat_text(summary.get("Target") or summary.get("proteintargetlist"))\n    evidence_text = " ".join([title, description, protocol, comment, target])\n    evidence_low = evidence_text.lower()\n    title_low = title.lower()\n    title_target_low = f"{title} {target}".lower()\n\n    matched = []\n    for keyword in ["g-quadruplex", "g4", "quadruplex", "telomeric", "c-myc", "bcl2", "bcl-2", "kras", "vegf"]:\n        if keyword in evidence_low:\n            matched.append(keyword)\n\n    has_g4_in_title_or_target = bool(re.search(r"(g-quadruplex|g4\\b|quadruplex|telomeric)", title_target_low))\n    direct_method_in_title = bool(\n        re.search(\n            r"(binding affinity|stabilization|melting temperature|fret|spr\\b|mst\\b|circular dichroism|cd spectroscopic|fluorescen|displacement)",\n            title_low,\n        )\n    )\n    direct_g4 = has_g4_in_title_or_target and direct_method_in_title\n    probable_g4 = bool(\n        re.search(r"(g-quadruplex|g4\\b|quadruplex-target|g-quadruplex-interactive|g4 stabilizer)", evidence_low)\n    )\n    related_biology_title = bool(\n        re.search(r"(telomerase|trap assay|transcription|promoter|antiproliferative|cytotoxicity|growth inhibition|inhibition)", title_low)\n    )\n    exclusions = [pattern for pattern in EXCLUSION_PATTERNS if re.search(pattern, evidence_low)]\n    hard_excluded = bool(exclusions) and not direct_g4\n\n    if direct_g4:\n        label = "confirmed_G4_assay"\n        reason = "The assay name/target itself describes direct G4/quadruplex binding, stabilization, melting, FRET, SPR, MST, CD, fluorescence, or displacement evidence."\n    elif probable_g4 and related_biology_title and not hard_excluded:\n        label = "probable_G4_assay"\n        reason = "G4/quadruplex evidence is present in the PubChem record, while the assay name is related biology rather than direct G4 binding."\n    else:\n        label = "ambiguous_or_excluded"\n        if matched:\n            reason = "G4 keyword hit is present, but the actual assay name/target does not provide enough direct or related G4 assay evidence for selection."\n        else:\n            reason = "No usable G4-specific evidence remained in title/description/protocol/comment/target text."\n    if label != "confirmed_G4_assay" and exclusions:\n        reason = f"{reason} Exclusion-context terms: {\', \'.join(exclusions[:4])}."\n\n    evidence = first_sentence_with_keyword(\n        " ".join([title, description, comment, protocol]),\n        ["g-quadruplex", "g4", "quadruplex", "telomeric", "c-myc", "bcl2", "bcl-2", "kras", "vegf"],\n    )\n    return label, evidence, "|".join(matched), reason\n\n\ndef search_aids(query_terms: list[str], retmax: int, sleep: float, timeout: int) -> tuple[dict[str, list[int]], dict[str, Any]]:\n    query_to_aids: dict[str, list[int]] = {}\n    raw: dict[str, Any] = {}\n    for term in query_terms:\n        url = eutils_url(\n            "esearch",\n            {\n                "db": "pcassay",\n                "term": f\'"{term}"\',\n                "retmode": "json",\n                "retmax": retmax,\n                "sort": "relevance",\n            },\n        )\n        payload = fetch_json(url, timeout=timeout)\n        raw[term] = payload\n        idlist = payload.get("esearchresult", {}).get("idlist", [])\n        query_to_aids[term] = [int(item) for item in idlist if str(item).isdigit()]\n        time.sleep(sleep)\n    return query_to_aids, raw\n\n\ndef fetch_esummaries(aids: list[int], batch_size: int, sleep: float, timeout: int) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:\n    summaries: dict[int, dict[str, Any]] = {}\n    raw_batches: list[dict[str, Any]] = []\n    for batch in chunks(aids, batch_size):\n        url = eutils_url("esummary", {"db": "pcassay", "id": ",".join(map(str, batch)), "retmode": "json"})\n        payload = fetch_json(url, timeout=timeout)\n        raw_batches.append(payload)\n        result = payload.get("result", {})\n        for uid in result.get("uids", []):\n            item = result.get(str(uid))\n            if item:\n                aid = int(item.get("aid") or item.get("uid") or uid)\n                summaries[aid] = item\n        time.sleep(sleep)\n    return summaries, raw_batches\n\n\ndef fetch_pug_summaries(aids: list[int], batch_size: int, sleep: float, timeout: int) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], list[str]]:\n    summaries: dict[int, dict[str, Any]] = {}\n    raw_batches: list[dict[str, Any]] = []\n    errors: list[str] = []\n    for batch in chunks(aids, batch_size):\n        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{\',\'.join(map(str, batch))}/summary/JSON"\n        try:\n            payload = fetch_json(url, timeout=timeout)\n            raw_batches.append(payload)\n            items = payload.get("AssaySummaries", {}).get("AssaySummary", [])\n            for item in items:\n                summaries[int(item["AID"])] = item\n        except Exception as exc:  # noqa: BLE001 - exact blocker is written to report.\n            errors.append(str(exc))\n        time.sleep(sleep)\n    return summaries, raw_batches, errors\n\n\ndef merge_summary(esummary: dict[str, Any], pug_summary: dict[str, Any] | None) -> dict[str, Any]:\n    if not pug_summary:\n        return esummary\n    merged = dict(esummary)\n    for key, value in pug_summary.items():\n        if value not in (None, "", []):\n            merged[key] = value\n    return merged\n\n\ndef row_for(query_terms: list[str], aid: int, summary: dict[str, Any]) -> dict[str, Any]:\n    label, evidence, matched_keywords, reason = classify(summary)\n    active = count_from(summary, "Active")\n    inactive = count_from(summary, "Inactive")\n    inconclusive = count_from(summary, "Inconclusive")\n    total = count_from(summary, "All")\n    if total is None:\n        total_value = summary.get("totalsidcount")\n        total = int(total_value) if str(total_value).isdigit() else None\n    title = summary.get("Name") or summary.get("assayname") or ""\n    description = flat_text(summary.get("Description") or summary.get("assaydescription"))\n    comment = flat_text(summary.get("Comment"))\n    target = flat_text(summary.get("Target") or summary.get("proteintargetlist"))\n    source = summary.get("SourceName") or summary.get("currentsourcename") or flat_text(summary.get("sourcenamelist"))\n    source_id = summary.get("SourceID") or summary.get("assaysourceid") or ""\n    return {\n        "query_term": "|".join(query_terms),\n        "aid": aid,\n        "title_name": compact(title, 300),\n        "source": compact(source, 200),\n        "source_id": source_id,\n        "method": summary.get("Method") or summary.get("activityoutcomemethod") or "",\n        "assay_target": compact(target, 300),\n        "description_snippet": first_sentence_with_keyword(description or title, ["g-quadruplex", "g4", "quadruplex", "telomeric", "c-myc", "bcl2", "bcl-2", "kras", "vegf"], 650),\n        "comment": compact(comment, 500),\n        "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}",\n        "active_count": active if active is not None else "",\n        "inactive_count": inactive if inactive is not None else "",\n        "inconclusive_count": inconclusive if inconclusive is not None else "",\n        "total_count": total if total is not None else "",\n        "count_basis": "CIDCount if available, else SIDCount; assay-level PubChem summary counts are rough and may not be mutually exclusive for all assay types.",\n        "curation_label": label,\n        "curation_evidence": evidence,\n        "matched_keywords": matched_keywords,\n        "excluded_reason": reason if label == "ambiguous_or_excluded" else "",\n        "deposit_date": summary.get("DepositDate") or summary.get("depositdate") or "",\n        "modify_date": summary.get("LastDataChange") or summary.get("modifydate") or "",\n    }\n\n\ndef write_csv(path: Path, rows: list[dict[str, Any]]) -> None:\n    with path.open("w", newline="", encoding="utf-8") as handle:\n        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)\n        writer.writeheader()\n        for row in rows:\n            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})\n\n\ndef numeric_sum(rows: list[dict[str, Any]], key: str) -> int:\n    total = 0\n    for row in rows:\n        value = row.get(key)\n        if value in ("", None):\n            continue\n        total += int(value)\n    return total\n\n\ndef report_markdown(rows: list[dict[str, Any]], selected: list[dict[str, Any]], query_to_aids: dict[str, list[int]], pug_errors: list[str]) -> str:\n    generated = datetime.now().replace(microsecond=0).isoformat()\n    labels = defaultdict(int)\n    for row in rows:\n        labels[row["curation_label"]] += 1\n    selected_active = numeric_sum(selected, "active_count")\n    selected_inactive = numeric_sum(selected, "inactive_count")\n    selected_inconclusive = numeric_sum(selected, "inconclusive_count")\n    selected_total = numeric_sum(selected, "total_count")\n    confirmed = [row for row in rows if row["curation_label"] == "confirmed_G4_assay"]\n    probable = [row for row in rows if row["curation_label"] == "probable_G4_assay"]\n    selected_with_counts = [row for row in selected if row.get("total_count") not in ("", None)]\n    feasible = "uncertain"\n    if selected_total >= 1200 and len(selected_with_counts) >= 3:\n        feasible = "likely feasible after compound-level download, deduplication, and novelty filtering"\n    elif selected_total >= 800:\n        feasible = "possibly feasible, but count inflation and overlap must be checked at compound level"\n    else:\n        feasible = "not yet demonstrated from selected assay-level counts"\n\n    per_query = "\\n".join(f"- {term}: {len(aids)} returned AIDs within the configured cap" for term, aids in query_to_aids.items())\n    top_selected = sorted(selected, key=lambda row: int(row["total_count"] or 0), reverse=True)[:15]\n    selected_lines = "\\n".join(\n        f"- AID {row[\'aid\']} ({row[\'curation_label\']}): total {row[\'total_count\'] or \'NA\'}, active {row[\'active_count\'] or \'NA\'}, inactive {row[\'inactive_count\'] or \'NA\'}; {row[\'title_name\']}"\n        for row in top_selected\n    )\n    if not selected_lines:\n        selected_lines = "- None selected at this checkpoint."\n    errors = "\\n".join(f"- {error}" for error in pug_errors[:10]) if pug_errors else "- None."\n\n    return f"""# PubChem G4 BioAssay AID Checkpoint\n\nGenerated: {generated}\n\n## Scope\n\nThis is the first checkpoint for `pubchem_g4_bioassay_external_validation`. It identifies and curates candidate PubChem BioAssay AIDs using official NCBI/PubChem endpoints only. It stops at assay-level metadata and rough active/inactive counts.\n\nNo GNN prediction, retraining, threshold tuning, or compound-level novelty filtering was performed. Do not claim independent external validation from this checkpoint alone.\n\n## Query Strategy\n\nOfficial routes used:\n\n- NCBI E-utilities `esearch.fcgi?db=pcassay` for phrase searches.\n- NCBI E-utilities `esummary.fcgi?db=pcassay` for candidate metadata.\n- PubChem PUG-REST `assay/aid/<AIDs>/summary/JSON` for fuller assay summaries and rough outcome counts.\n\nConfigured search cap: first {sum(len(v) for v in query_to_aids.values())} query-specific hits total before de-duplication.\n\n{per_query}\n\n## Curation Rules\n\n- `confirmed_G4_assay`: the assay name or target itself describes G4/quadruplex binding, stabilization, melting, FRET, SPR, MST, CD, fluorescence, or displacement.\n- `probable_G4_assay`: G4/quadruplex evidence is present in the PubChem record, while the assay name is related biology such as telomerase, transcription, antiproliferation, or inhibition rather than direct G4 binding.\n- `ambiguous_or_excluded`: keyword hit without enough G4 assay-name/target evidence, duplex/hairpin control assays, or mostly phenotypic/cell viability/drug-combination context.\n\nEvidence text and matched keywords are retained in `candidate_pubchem_g4_aids.csv`.\n\n## Counts\n\n- Unique candidate AIDs found: {len(rows)}\n- Selected AIDs: {len(selected)}\n- Confirmed G4 assay AIDs: {labels[\'confirmed_G4_assay\']}\n- Probable G4 assay AIDs: {labels[\'probable_G4_assay\']}\n- Ambiguous/excluded AIDs: {labels[\'ambiguous_or_excluded\']}\n\nRough selected assay-level outcome counts:\n\n- Total compounds/substances counted by PubChem summaries: {selected_total}\n- Active: {selected_active}\n- Inactive: {selected_inactive}\n- Inconclusive: {selected_inconclusive}\n\nBreakdown by curation class:\n\n- Confirmed only: total {numeric_sum(confirmed, "total_count")}, active {numeric_sum(confirmed, "active_count")}, inactive {numeric_sum(confirmed, "inactive_count")}, inconclusive {numeric_sum(confirmed, "inconclusive_count")}\n- Probable only: total {numeric_sum(probable, "total_count")}, active {numeric_sum(probable, "active_count")}, inactive {numeric_sum(probable, "inactive_count")}, inconclusive {numeric_sum(probable, "inconclusive_count")}\n\nCounts prefer PubChem `CIDCount*` fields when present and fall back to `SIDCount*`. These are assay-level rough counts and may not be mutually exclusive for every assay type.\n\n## Selected AIDs\n\n{selected_lines}\n\n## Feasibility For 800-1200 Compounds\n\nTarget feasibility: {feasible}.\n\nThe assay-level selected total is {selected_total}, but this is not a compound-ready benchmark yet. PubChem counts can include overlapping substances/compounds across related assays, combination-treatment rows, and assays whose activity outcome is not directly equivalent to G4 binding.\n\n## Recommended Next Step\n\nBefore full prediction, manually review the selected AIDs in PubChem, then download compound-level outcomes only for confirmed/probable assays. Standardize CIDs/SMILES, collapse replicates, define assay-specific active/inactive labels, remove training exact overlaps and high-similarity analogs, and report novelty-filtered counts. Only after that should the locked GNN be run.\n\n## API/Network Notes\n\nPUG-REST summary errors:\n\n{errors}\n\n## Generated Files\n\n- `candidate_pubchem_g4_aids.csv`\n- `selected_pubchem_g4_aids.csv`\n- `pubchem_g4_aid_checkpoint_report.md`\n- `raw_ncbi_esearch.json`\n- `raw_ncbi_esummary_batches.json`\n- `raw_pubchem_pug_summary_batches.json`\n"""\n\n\ndef main() -> None:\n    args = parse_args()\n    root = root_from_script()\n    output_dir = resolve(args.output_dir, root)\n    output_dir.mkdir(parents=True, exist_ok=True)\n\n    query_to_aids, raw_esearch = search_aids(QUERY_TERMS, args.retmax_per_query, args.sleep, args.timeout)\n    aid_to_queries: dict[int, list[str]] = defaultdict(list)\n    for query, aids in query_to_aids.items():\n        for aid in aids:\n            aid_to_queries[aid].append(query)\n    unique_aids = sorted(aid_to_queries)\n\n    esummaries, raw_esummary_batches = fetch_esummaries(unique_aids, args.batch_size, args.sleep, args.timeout)\n    pug_summaries, raw_pug_batches, pug_errors = fetch_pug_summaries(unique_aids, args.batch_size, args.sleep, args.timeout)\n\n    rows = []\n    for aid in unique_aids:\n        summary = merge_summary(esummaries.get(aid, {"aid": aid}), pug_summaries.get(aid))\n        rows.append(row_for(aid_to_queries[aid], aid, summary))\n\n    rows = sorted(rows, key=lambda row: (row["curation_label"] == "ambiguous_or_excluded", -int(row["total_count"] or 0), int(row["aid"])))\n    selected = [row for row in rows if row["curation_label"] in {"confirmed_G4_assay", "probable_G4_assay"}]\n\n    write_csv(output_dir / "candidate_pubchem_g4_aids.csv", rows)\n    write_csv(output_dir / "selected_pubchem_g4_aids.csv", selected)\n    (output_dir / "pubchem_g4_aid_checkpoint_report.md").write_text(\n        report_markdown(rows, selected, query_to_aids, pug_errors),\n        encoding="utf-8",\n    )\n    (output_dir / "raw_ncbi_esearch.json").write_text(json.dumps(raw_esearch, indent=2, sort_keys=True), encoding="utf-8")\n    (output_dir / "raw_ncbi_esummary_batches.json").write_text(json.dumps(raw_esummary_batches, indent=2), encoding="utf-8")\n    (output_dir / "raw_pubchem_pug_summary_batches.json").write_text(json.dumps(raw_pug_batches, indent=2), encoding="utf-8")\n\n    print(f"Wrote {len(rows)} candidate AIDs and {len(selected)} selected AIDs to {output_dir}")\n\n\nif __name__ == "__main__":\n    main()\n'


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
