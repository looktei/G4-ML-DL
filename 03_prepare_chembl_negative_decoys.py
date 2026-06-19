#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Prepare ChEMBL assumed-negative decoys and combined binary dataset."""

STEP_ID = '03'
STEP_TITLE = 'Prepare ChEMBL assumed-negative decoys and combined binary dataset'
EMBEDDED_SOURCE_NAME = '04_prepare_chembl_negative_decoys.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Prepare assumed-negative ChEMBL decoys matched to G4LDB positives.\n\nThese compounds are decoys/assumed negatives only. They are not experimentally\nconfirmed non-G4 binders. The script fetches ordinary small molecules from the\npublic ChEMBL API, removes structures that overlap the G4LDB positive set, and\nsamples a target-sized negative set matched on simple RDKit properties.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport math\nimport random\nimport time\nfrom collections import Counter, defaultdict\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nfrom typing import Any\n\nimport pandas as pd\nimport requests\nfrom rdkit import Chem, DataStructs\nfrom rdkit import RDLogger\nfrom rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors\nfrom rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator\n\n\nRDLogger.DisableLog("rdApp.*")\n\nCHEMBL_MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"\nALLOWED_ATOMIC_NUMBERS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}\nCORE_COLUMNS = [\n    "source",\n    "compound_id",\n    "name",\n    "smiles",\n    "canonical_smiles",\n    "label",\n    "assay_note",\n    "rdkit_formula",\n    "rdkit_mw",\n    "rdkit_hbd",\n    "rdkit_hba",\n    "rdkit_alogp",\n    "rdkit_tpsa",\n]\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument(\n        "--positive-input",\n        default="data/g4ldb/g4ldb_g4_ligand_ml_preprocessed.csv",\n        help="Preprocessed G4LDB positive CSV.",\n    )\n    parser.add_argument(\n        "--output",\n        default="data/chembl/chembl_negative_decoys.csv",\n        help="Core decoy CSV output.",\n    )\n    parser.add_argument(\n        "--metadata-output",\n        default="data/chembl/chembl_negative_decoys_with_metadata.csv",\n        help="Decoy CSV with ChEMBL metadata.",\n    )\n    parser.add_argument(\n        "--report",\n        default="data/chembl/chembl_negative_decoy_report.json",\n        help="JSON report output.",\n    )\n    parser.add_argument(\n        "--combined-output",\n        default="data/modeling/g4ldb_chembl_binary_dataset.csv",\n        help="Combined positive/decoy modeling CSV.",\n    )\n    parser.add_argument(\n        "--summary",\n        default="data/chembl/chembl_negative_decoy_summary.md",\n        help="Markdown summary output.",\n    )\n    parser.add_argument(\n        "--similarity-filtered-output",\n        default="data/chembl/chembl_negative_decoys_removed_similar_to_g4ldb.csv",\n        help="Audit CSV for ChEMBL candidates removed by positive-set similarity filtering.",\n    )\n    parser.add_argument(\n        "--similarity-threshold",\n        type=float,\n        default=0.70,\n        help="Remove ChEMBL candidates with max Morgan/Tanimoto similarity to any G4LDB positive >= this value.",\n    )\n    parser.add_argument("--fingerprint-radius", type=int, default=2, help="Morgan fingerprint radius.")\n    parser.add_argument("--fingerprint-size", type=int, default=2048, help="Morgan fingerprint bit size.")\n    parser.add_argument("--target-count", type=int, default=None, help="Override target decoy count.")\n    parser.add_argument("--limit", type=int, default=1000, help="ChEMBL page size.")\n    parser.add_argument("--max-pages", type=int, default=140, help="Maximum ChEMBL pages to fetch.")\n    parser.add_argument(\n        "--min-clean-candidates",\n        type=int,\n        default=None,\n        help="Stop fetching after at least this many clean candidates are available.",\n    )\n    parser.add_argument("--seed", type=int, default=20260601, help="Deterministic sampling seed.")\n    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")\n    parser.add_argument("--retries", type=int, default=4, help="HTTP retries per page.")\n    return parser.parse_args()\n\n\ndef utc_now() -> str:\n    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()\n\n\ndef canonicalize_and_describe(smiles: Any) -> tuple[str | None, dict[str, Any] | None, str | None]:\n    if pd.isna(smiles) or not str(smiles).strip():\n        return None, None, "missing_smiles"\n    raw = str(smiles).strip()\n    if "." in raw:\n        return None, None, "mixture_or_salt"\n    mol = Chem.MolFromSmiles(raw)\n    if mol is None:\n        return None, None, "invalid_smiles"\n    if not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):\n        return None, None, "non_organic"\n    if any(atom.GetAtomicNum() not in ALLOWED_ATOMIC_NUMBERS for atom in mol.GetAtoms()):\n        return None, None, "unusual_element"\n    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)\n    descriptors = {\n        "rdkit_formula": rdMolDescriptors.CalcMolFormula(mol),\n        "rdkit_mw": round(float(Descriptors.MolWt(mol)), 4),\n        "rdkit_hbd": int(Lipinski.NumHDonors(mol)),\n        "rdkit_hba": int(Lipinski.NumHAcceptors(mol)),\n        "rdkit_alogp": round(float(Crippen.MolLogP(mol)), 4),\n        "rdkit_tpsa": round(float(rdMolDescriptors.CalcTPSA(mol)), 4),\n        "heavy_atoms": int(mol.GetNumHeavyAtoms()),\n    }\n    return canonical, descriptors, None\n\n\ndef canonicalize_positive_smiles(smiles: Any) -> tuple[str | None, dict[str, Any] | None, str | None]:\n    if pd.isna(smiles) or not str(smiles).strip():\n        return None, None, "missing_smiles"\n    mol = Chem.MolFromSmiles(str(smiles).strip())\n    if mol is None:\n        return None, None, "invalid_smiles"\n    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)\n    descriptors = {\n        "rdkit_formula": rdMolDescriptors.CalcMolFormula(mol),\n        "rdkit_mw": round(float(Descriptors.MolWt(mol)), 4),\n        "rdkit_hbd": int(Lipinski.NumHDonors(mol)),\n        "rdkit_hba": int(Lipinski.NumHAcceptors(mol)),\n        "rdkit_alogp": round(float(Crippen.MolLogP(mol)), 4),\n        "rdkit_tpsa": round(float(rdMolDescriptors.CalcTPSA(mol)), 4),\n        "heavy_atoms": int(mol.GetNumHeavyAtoms()),\n    }\n    return canonical, descriptors, None\n\n\ndef load_positives(path: Path) -> tuple[pd.DataFrame, set[str], dict[str, Any]]:\n    positives = pd.read_csv(path)\n    rows = []\n    invalid = 0\n    for _, row in positives.iterrows():\n        canonical, descriptors, reason = canonicalize_positive_smiles(row.get("canonical_smiles") or row.get("smiles"))\n        if reason:\n            invalid += 1\n            continue\n        enriched = row.to_dict()\n        enriched["positive_canonical_from_rdkit"] = canonical\n        if descriptors:\n            enriched.update({f"positive_{key}": value for key, value in descriptors.items()})\n        rows.append(enriched)\n    clean = pd.DataFrame(rows)\n    positive_set = set(clean["positive_canonical_from_rdkit"])\n    report = {\n        "positive_input": str(path),\n        "positive_input_rows": int(len(positives)),\n        "positive_valid_rdkit_rows": int(len(clean)),\n        "positive_unique_canonical_smiles": int(len(positive_set)),\n        "positive_invalid_for_matching": int(invalid),\n    }\n    return clean, positive_set, report\n\n\ndef filter_candidates_by_positive_similarity(\n    candidates: pd.DataFrame,\n    positives: pd.DataFrame,\n    threshold: float,\n    radius: int,\n    fp_size: int,\n) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:\n    """Remove ChEMBL candidates that are too close to any G4LDB positive.\n\n    Exact canonical SMILES overlap is already removed during fetching. This\n    second pass catches scaffold/analog-level similarity using Morgan\n    fingerprints and Tanimoto similarity.\n    """\n    generator = GetMorganGenerator(radius=radius, fpSize=fp_size)\n    positive_items = []\n    for _, row in positives.iterrows():\n        mol = Chem.MolFromSmiles(str(row["positive_canonical_from_rdkit"]))\n        if mol is None:\n            continue\n        positive_items.append(\n            {\n                "compound_id": row.get("ligandId"),\n                "canonical_smiles": row["positive_canonical_from_rdkit"],\n                "fingerprint": generator.GetFingerprint(mol),\n            }\n        )\n    if not positive_items:\n        raise RuntimeError("No positive fingerprints available for similarity filtering.")\n\n    positive_fps = [item["fingerprint"] for item in positive_items]\n    kept_rows = []\n    removed_rows = []\n    invalid_for_similarity = 0\n\n    for _, row in candidates.iterrows():\n        mol = Chem.MolFromSmiles(str(row["canonical_smiles"]))\n        if mol is None:\n            invalid_for_similarity += 1\n            removed = row.to_dict()\n            removed.update(\n                {\n                    "removal_reason": "invalid_for_similarity",\n                    "max_positive_tanimoto": None,\n                    "nearest_positive_compound_id": None,\n                    "nearest_positive_canonical_smiles": None,\n                }\n            )\n            removed_rows.append(removed)\n            continue\n\n        fingerprint = generator.GetFingerprint(mol)\n        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, positive_fps)\n        max_idx, max_similarity = max(enumerate(similarities), key=lambda item: item[1])\n        if max_similarity >= threshold:\n            nearest = positive_items[max_idx]\n            removed = row.to_dict()\n            removed.update(\n                {\n                    "removal_reason": "similar_to_g4ldb_positive",\n                    "max_positive_tanimoto": round(float(max_similarity), 6),\n                    "nearest_positive_compound_id": nearest["compound_id"],\n                    "nearest_positive_canonical_smiles": nearest["canonical_smiles"],\n                }\n            )\n            removed_rows.append(removed)\n        else:\n            kept = row.to_dict()\n            kept["max_positive_tanimoto"] = round(float(max_similarity), 6)\n            kept_rows.append(kept)\n\n    kept = pd.DataFrame(kept_rows)\n    removed = pd.DataFrame(removed_rows)\n    report = {\n        "method": "Morgan fingerprint Tanimoto similarity against all G4LDB positives.",\n        "threshold": float(threshold),\n        "fingerprint_radius": int(radius),\n        "fingerprint_size": int(fp_size),\n        "positive_fingerprints": int(len(positive_fps)),\n        "input_candidates": int(len(candidates)),\n        "kept_candidates": int(len(kept)),\n        "removed_candidates": int(len(removed)),\n        "invalid_for_similarity": int(invalid_for_similarity),\n    }\n    if not removed.empty and "max_positive_tanimoto" in removed.columns:\n        values = pd.to_numeric(removed["max_positive_tanimoto"], errors="coerce").dropna()\n        if not values.empty:\n            report["removed_similarity_summary"] = {\n                "min": round(float(values.min()), 6),\n                "median": round(float(values.median()), 6),\n                "max": round(float(values.max()), 6),\n            }\n    if not kept.empty and "max_positive_tanimoto" in kept.columns:\n        values = pd.to_numeric(kept["max_positive_tanimoto"], errors="coerce").dropna()\n        if not values.empty:\n            report["kept_similarity_summary"] = {\n                "min": round(float(values.min()), 6),\n                "median": round(float(values.median()), 6),\n                "max": round(float(values.max()), 6),\n            }\n    return kept, removed, report\n\n\ndef request_page(session: requests.Session, params: dict[str, Any], timeout: float, retries: int) -> dict[str, Any]:\n    last_error: str | None = None\n    for attempt in range(retries):\n        try:\n            response = session.get(CHEMBL_MOLECULE_URL, params=params, timeout=timeout)\n            response.raise_for_status()\n            return response.json()\n        except (requests.RequestException, ValueError) as exc:\n            last_error = str(exc)\n            sleep_for = min(20, 2**attempt)\n            time.sleep(sleep_for)\n    raise RuntimeError(f"ChEMBL request failed after {retries} retries: {last_error}")\n\n\ndef chembl_property(record: dict[str, Any], key: str) -> Any:\n    props = record.get("molecule_properties") or {}\n    return props.get(key)\n\n\ndef clean_chembl_record(record: dict[str, Any], positive_smiles: set[str]) -> tuple[dict[str, Any] | None, str | None]:\n    structures = record.get("molecule_structures") or {}\n    canonical, descriptors, reason = canonicalize_and_describe(structures.get("canonical_smiles"))\n    if reason:\n        return None, reason\n    assert canonical is not None and descriptors is not None\n    if canonical in positive_smiles:\n        return None, "overlaps_positive"\n    mw = descriptors["rdkit_mw"]\n    if mw < 100 or mw > 1200:\n        return None, "outside_mw_window"\n    row = {\n        "source": "ChEMBL_assumed_negative_decoy",\n        "compound_id": record.get("molecule_chembl_id"),\n        "name": (record.get("pref_name") or record.get("molecule_chembl_id") or ""),\n        "smiles": structures.get("canonical_smiles"),\n        "canonical_smiles": canonical,\n        "label": 0,\n        "assay_note": "assumed-negative decoy; not experimentally confirmed non-G4 binder",\n        "chembl_id": record.get("molecule_chembl_id"),\n        "parent_chembl_id": (record.get("molecule_hierarchy") or {}).get("parent_chembl_id"),\n        "pref_name": record.get("pref_name"),\n        "molecule_type": record.get("molecule_type"),\n        "max_phase": record.get("max_phase"),\n        "therapeutic_flag": record.get("therapeutic_flag"),\n        "natural_product": record.get("natural_product"),\n        "black_box_warning": record.get("black_box_warning"),\n        "chembl_full_mwt": chembl_property(record, "full_mwt"),\n        "chembl_alogp": chembl_property(record, "alogp"),\n        "chembl_hbd": chembl_property(record, "hbd"),\n        "chembl_hba": chembl_property(record, "hba"),\n        "chembl_psa": chembl_property(record, "psa"),\n        "chembl_ro5_violations": chembl_property(record, "num_ro5_violations"),\n    }\n    row.update(descriptors)\n    return row, None\n\n\ndef fetch_candidates(\n    positive_smiles: set[str],\n    target_count: int,\n    limit: int,\n    max_pages: int,\n    min_clean_candidates: int | None,\n    timeout: float,\n    retries: int,\n) -> tuple[pd.DataFrame, dict[str, Any]]:\n    session = requests.Session()\n    session.headers.update({"User-Agent": "Nipah-virus-project-decoy-prep/1.0"})\n    min_clean = min_clean_candidates or max(target_count * 8, 25000)\n    rows: list[dict[str, Any]] = []\n    seen: set[str] = set()\n    removed = Counter()\n    pages = 0\n    raw_records = 0\n    next_url_seen = True\n    base_params = {\n        "limit": limit,\n        "offset": 0,\n        "molecule_type": "Small molecule",\n        "molecule_properties__full_mwt__gte": 100,\n        "molecule_properties__full_mwt__lte": 1200,\n    }\n    while pages < max_pages and next_url_seen and len(rows) < min_clean:\n        params = dict(base_params)\n        params["offset"] = pages * limit\n        payload = request_page(session, params, timeout=timeout, retries=retries)\n        molecules = payload.get("molecules") or []\n        raw_records += len(molecules)\n        for record in molecules:\n            row, reason = clean_chembl_record(record, positive_smiles)\n            if reason:\n                removed[reason] += 1\n                continue\n            assert row is not None\n            canonical = row["canonical_smiles"]\n            if canonical in seen:\n                removed["duplicate_canonical_smiles"] += 1\n                continue\n            seen.add(canonical)\n            rows.append(row)\n        pages += 1\n        page_meta = payload.get("page_meta") or {}\n        next_url_seen = bool(page_meta.get("next"))\n    report = {\n        "chembl_api_url": CHEMBL_MOLECULE_URL,\n        "fetched_pages": int(pages),\n        "raw_records_seen": int(raw_records),\n        "clean_unique_candidates": int(len(rows)),\n        "candidate_removals": {key: int(value) for key, value in sorted(removed.items())},\n        "fetch_stop_reason": (\n            "min_clean_candidates_reached"\n            if len(rows) >= min_clean\n            else "max_pages_or_api_exhausted"\n        ),\n        "min_clean_candidates": int(min_clean),\n    }\n    return pd.DataFrame(rows), report\n\n\ndef mw_bin(value: float) -> int:\n    return int(math.floor(float(value) / 50.0) * 50)\n\n\ndef logp_bin(value: float) -> int:\n    return int(math.floor(float(value)))\n\n\ndef hbd_bin(value: float) -> str:\n    return "5+" if float(value) >= 5 else str(int(value))\n\n\ndef hba_bin(value: float) -> str:\n    value = int(value)\n    if value <= 2:\n        return "0-2"\n    if value <= 5:\n        return "3-5"\n    if value <= 8:\n        return "6-8"\n    if value <= 12:\n        return "9-12"\n    return "13+"\n\n\ndef add_bins(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:\n    out = df.copy()\n    mw_col = f"{prefix}rdkit_mw" if prefix else "rdkit_mw"\n    logp_col = f"{prefix}rdkit_alogp" if prefix else "rdkit_alogp"\n    hbd_col = f"{prefix}rdkit_hbd" if prefix else "rdkit_hbd"\n    hba_col = f"{prefix}rdkit_hba" if prefix else "rdkit_hba"\n    out["mw_bin"] = out[mw_col].map(mw_bin)\n    out["logp_bin"] = out[logp_col].map(logp_bin)\n    out["hbd_bin"] = out[hbd_col].map(hbd_bin)\n    out["hba_bin"] = out[hba_col].map(hba_bin)\n    out["match_bin"] = list(zip(out["mw_bin"], out["logp_bin"], out["hbd_bin"], out["hba_bin"]))\n    out["mw_logp_bin"] = list(zip(out["mw_bin"], out["logp_bin"]))\n    return out\n\n\ndef sample_decoys(positives: pd.DataFrame, candidates: pd.DataFrame, target_count: int, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:\n    rng = random.Random(seed)\n    positives_binned = add_bins(positives, prefix="positive_")\n    candidates_binned = add_bins(candidates)\n    candidate_indices_by_full: dict[tuple[Any, ...], list[int]] = defaultdict(list)\n    candidate_indices_by_mw_logp: dict[tuple[Any, ...], list[int]] = defaultdict(list)\n    for idx, row in candidates_binned.iterrows():\n        candidate_indices_by_full[row["match_bin"]].append(idx)\n        candidate_indices_by_mw_logp[row["mw_logp_bin"]].append(idx)\n    for values in list(candidate_indices_by_full.values()) + list(candidate_indices_by_mw_logp.values()):\n        rng.shuffle(values)\n\n    selected: list[int] = []\n    selected_set: set[int] = set()\n    full_matches = 0\n    mw_logp_backfill = 0\n    positive_full_counts = Counter(positives_binned["match_bin"])\n    positive_mw_logp_counts = Counter(positives_binned["mw_logp_bin"])\n\n    for bin_key, needed in positive_full_counts.most_common():\n        pool = candidate_indices_by_full.get(bin_key, [])\n        take = min(needed, len([idx for idx in pool if idx not in selected_set]), target_count - len(selected))\n        for idx in pool:\n            if take <= 0:\n                break\n            if idx in selected_set:\n                continue\n            selected.append(idx)\n            selected_set.add(idx)\n            full_matches += 1\n            take -= 1\n        if len(selected) >= target_count:\n            break\n\n    if len(selected) < target_count:\n        selected_mw_logp_counts = Counter(candidates_binned.loc[selected, "mw_logp_bin"])\n        deficits = []\n        for bin_key, needed in positive_mw_logp_counts.items():\n            deficits.extend([bin_key] * max(0, needed - selected_mw_logp_counts.get(bin_key, 0)))\n        rng.shuffle(deficits)\n        for bin_key in deficits:\n            if len(selected) >= target_count:\n                break\n            pool = candidate_indices_by_mw_logp.get(bin_key, [])\n            for idx in pool:\n                if idx not in selected_set:\n                    selected.append(idx)\n                    selected_set.add(idx)\n                    mw_logp_backfill += 1\n                    break\n\n    nearest_backfill = 0\n    if len(selected) < target_count:\n        pos_mw_logp = list(positive_mw_logp_counts.elements())\n        rng.shuffle(pos_mw_logp)\n        remaining = candidates_binned.loc[~candidates_binned.index.isin(selected_set)].copy()\n        for bin_key in pos_mw_logp:\n            if len(selected) >= target_count or remaining.empty:\n                break\n            mw_target, logp_target = bin_key\n            remaining["distance"] = (\n                ((remaining["mw_bin"] - mw_target) / 50.0).abs()\n                + ((remaining["logp_bin"] - logp_target) / 1.0).abs()\n                + 0.25 * remaining["hbd_bin"].map(lambda value: 5 if value == "5+" else int(value))\n            )\n            best_idx = remaining.sort_values(["distance", "rdkit_mw", "canonical_smiles"]).index[0]\n            selected.append(best_idx)\n            selected_set.add(best_idx)\n            nearest_backfill += 1\n            remaining = remaining.drop(index=best_idx)\n\n    decoys = candidates_binned.loc[selected].copy().reset_index(drop=True)\n    decoys["selection_method"] = (\n        ["full_mw_logp_hbd_hba_bin"] * full_matches\n        + ["mw_logp_bin_backfill"] * mw_logp_backfill\n        + ["nearest_property_backfill"] * nearest_backfill\n    )[: len(decoys)]\n    report = {\n        "target_decoy_count": int(target_count),\n        "selected_decoy_count": int(len(decoys)),\n        "selection_counts": {\n            "full_mw_logp_hbd_hba_bin": int(full_matches),\n            "mw_logp_bin_backfill": int(mw_logp_backfill),\n            "nearest_property_backfill": int(nearest_backfill),\n        },\n    }\n    return decoys, report\n\n\ndef summarize_distribution(positive: pd.DataFrame, decoy: pd.DataFrame) -> dict[str, Any]:\n    summary: dict[str, Any] = {}\n    for column in ["rdkit_mw", "rdkit_alogp", "rdkit_hbd", "rdkit_hba", "rdkit_tpsa"]:\n        pos_col = f"positive_{column}" if f"positive_{column}" in positive.columns else column\n        if pos_col not in positive.columns or column not in decoy.columns:\n            continue\n        pos_values = pd.to_numeric(positive[pos_col], errors="coerce")\n        neg_values = pd.to_numeric(decoy[column], errors="coerce")\n        summary[column] = {\n            "positive_mean": round(float(pos_values.mean()), 4),\n            "decoy_mean": round(float(neg_values.mean()), 4),\n            "mean_difference_decoy_minus_positive": round(float(neg_values.mean() - pos_values.mean()), 4),\n            "positive_median": round(float(pos_values.median()), 4),\n            "decoy_median": round(float(neg_values.median()), 4),\n        }\n    return summary\n\n\ndef harmonize_positives(positives: pd.DataFrame) -> pd.DataFrame:\n    rows = []\n    for _, row in positives.iterrows():\n        rows.append(\n            {\n                "source": "G4LDB_positive",\n                "compound_id": row.get("ligandId"),\n                "name": row.get("name"),\n                "smiles": row.get("smiles"),\n                "canonical_smiles": row.get("positive_canonical_from_rdkit"),\n                "label": 1,\n                "assay_note": "G4LDB G4 ligand positive",\n                "rdkit_formula": row.get("positive_rdkit_formula") or row.get("rdkit_formula"),\n                "rdkit_mw": row.get("positive_rdkit_mw") or row.get("rdkit_mw"),\n                "rdkit_hbd": row.get("positive_rdkit_hbd") or row.get("rdkit_hbd"),\n                "rdkit_hba": row.get("positive_rdkit_hba") or row.get("rdkit_hba"),\n                "rdkit_alogp": row.get("positive_rdkit_alogp") or row.get("rdkit_alogp"),\n                "rdkit_tpsa": row.get("positive_rdkit_tpsa"),\n            }\n        )\n    return pd.DataFrame(rows)\n\n\ndef validate_outputs(positive_core: pd.DataFrame, decoys_core: pd.DataFrame, combined: pd.DataFrame) -> dict[str, Any]:\n    pos_set = set(positive_core["canonical_smiles"])\n    neg_set = set(decoys_core["canonical_smiles"])\n    core_missing = {\n        column: int(decoys_core[column].isna().sum())\n        for column in CORE_COLUMNS\n        if column in decoys_core.columns\n    }\n    return {\n        "negative_rows": int(len(decoys_core)),\n        "combined_rows": int(len(combined)),\n        "combined_label_counts": {\n            str(key): int(value) for key, value in combined["label"].value_counts().sort_index().items()\n        },\n        "negative_duplicate_canonical_smiles": int(decoys_core["canonical_smiles"].duplicated().sum()),\n        "combined_duplicate_canonical_smiles": int(combined["canonical_smiles"].duplicated().sum()),\n        "positive_negative_overlap": int(len(pos_set & neg_set)),\n        "negative_missing_core_fields": core_missing,\n    }\n\n\ndef write_summary(path: Path, report: dict[str, Any]) -> None:\n    lines = [\n        "# ChEMBL Assumed-Negative Decoys",\n        "",\n        "These rows are decoys/assumed negatives, not experimentally confirmed non-G4 binders.",\n        "",\n        f"- Generated at: {report[\'generated_at\']}",\n        f"- Positive target rows: {report[\'positive\'][\'positive_input_rows\']}",\n        f"- Selected decoy rows: {report[\'selection\'][\'selected_decoy_count\']}",\n        f"- Clean ChEMBL candidate pool: {report[\'fetch\'][\'clean_unique_candidates\']}",\n        f"- Removed as too similar to positives: {report[\'similarity_filter\'][\'removed_candidates\']}",\n        f"- Similarity threshold: Morgan/Tanimoto >= {report[\'similarity_filter\'][\'threshold\']}",\n        f"- Positive/decoy overlap after canonicalization: {report[\'validation\'][\'positive_negative_overlap\']}",\n        "",\n        "## Matching Residuals",\n        "",\n    ]\n    for key, values in report["distribution_summary"].items():\n        lines.append(\n            f"- {key}: positive mean {values[\'positive_mean\']}, decoy mean {values[\'decoy_mean\']}, "\n            f"difference {values[\'mean_difference_decoy_minus_positive\']}"\n        )\n    lines.extend(\n        [\n            "",\n            "## Files",\n            "",\n            f"- Core decoys: `{report[\'outputs\'][\'decoy_core\']}`",\n            f"- Decoys with metadata: `{report[\'outputs\'][\'decoy_metadata\']}`",\n            f"- Combined modeling dataset: `{report[\'outputs\'][\'combined_dataset\']}`",\n            f"- Similarity-filter audit: `{report[\'outputs\'][\'similarity_filtered_candidates\']}`",\n            f"- JSON report: `{report[\'outputs\'][\'report\']}`",\n        ]\n    )\n    path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")\n\n\ndef main() -> int:\n    args = parse_args()\n    positive_path = Path(args.positive_input)\n    output_path = Path(args.output)\n    metadata_output_path = Path(args.metadata_output)\n    report_path = Path(args.report)\n    combined_output_path = Path(args.combined_output)\n    summary_path = Path(args.summary)\n    similarity_filtered_output_path = Path(args.similarity_filtered_output)\n\n    positives, positive_smiles, positive_report = load_positives(positive_path)\n    target_count = int(args.target_count or positive_report["positive_input_rows"])\n\n    candidates, fetch_report = fetch_candidates(\n        positive_smiles=positive_smiles,\n        target_count=target_count,\n        limit=args.limit,\n        max_pages=args.max_pages,\n        min_clean_candidates=args.min_clean_candidates,\n        timeout=args.timeout,\n        retries=args.retries,\n    )\n    if candidates.empty:\n        raise RuntimeError("No clean ChEMBL candidates were fetched.")\n\n    candidates, similarity_removed, similarity_report = filter_candidates_by_positive_similarity(\n        candidates,\n        positives,\n        threshold=args.similarity_threshold,\n        radius=args.fingerprint_radius,\n        fp_size=args.fingerprint_size,\n    )\n    if candidates.empty:\n        raise RuntimeError("No ChEMBL candidates remained after positive-set similarity filtering.")\n\n    decoys, selection_report = sample_decoys(positives, candidates, target_count, args.seed)\n    if len(decoys) < target_count:\n        selection_report["warning"] = "Selected fewer decoys than requested because the clean candidate pool was insufficient."\n\n    decoys_core = decoys[CORE_COLUMNS].copy()\n    decoys_metadata = decoys.copy()\n    positive_core = harmonize_positives(positives)[CORE_COLUMNS].copy()\n    combined = pd.concat([positive_core, decoys_core], ignore_index=True, sort=False)\n\n    for path in [\n        output_path,\n        metadata_output_path,\n        similarity_filtered_output_path,\n        report_path,\n        combined_output_path,\n        summary_path,\n    ]:\n        path.parent.mkdir(parents=True, exist_ok=True)\n    decoys_core.to_csv(output_path, index=False)\n    decoys_metadata.to_csv(metadata_output_path, index=False)\n    similarity_removed.to_csv(similarity_filtered_output_path, index=False)\n    combined.to_csv(combined_output_path, index=False)\n\n    report = {\n        "generated_at": utc_now(),\n        "note": "ChEMBL decoys are assumed negatives only; absence of G4 activity evidence is not evidence of inactivity.",\n        "positive": positive_report,\n        "fetch": fetch_report,\n        "similarity_filter": similarity_report,\n        "selection": selection_report,\n        "distribution_summary": summarize_distribution(positives, decoys_core),\n        "validation": validate_outputs(positive_core, decoys_core, combined),\n        "matching_strategy": {\n            "primary": "Exact bins on MW width 50 Da, LogP width 1, HBD capped at 5+, and HBA grouped as 0-2/3-5/6-8/9-12/13+.",\n            "backfill": "Then MW+LogP bins, then nearest available MW/LogP bin when exact bins are underfilled.",\n            "positive_similarity_filter": (\n                "Before sampling, remove ChEMBL candidates with Morgan fingerprint Tanimoto "\n                f"similarity >= {args.similarity_threshold} to any G4LDB positive."\n            ),\n            "random_seed": int(args.seed),\n        },\n        "outputs": {\n            "decoy_core": str(output_path),\n            "decoy_metadata": str(metadata_output_path),\n            "similarity_filtered_candidates": str(similarity_filtered_output_path),\n            "report": str(report_path),\n            "combined_dataset": str(combined_output_path),\n            "summary": str(summary_path),\n        },\n    }\n    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")\n    write_summary(summary_path, report)\n    print(json.dumps(report, indent=2, ensure_ascii=False))\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'


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
