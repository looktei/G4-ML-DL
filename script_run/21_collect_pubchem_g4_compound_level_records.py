#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Collect PubChem compound-level G4 assay records."""

STEP_ID = '21'
STEP_TITLE = 'Collect PubChem compound-level G4 assay records'
EMBEDDED_SOURCE_NAME = '22_collect_pubchem_g4_compound_level_records.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Collect compound-level PubChem G4 BioAssay records for an external checkpoint.\n\nThis script builds tables only. It deliberately does not run the locked GNN,\nretrain models, or tune thresholds against model performance.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport io\nimport math\nimport re\nimport time\nfrom collections import Counter\nfrom datetime import datetime\nfrom pathlib import Path\nfrom typing import Any\nfrom urllib.error import HTTPError, URLError\nfrom urllib.request import Request, urlopen\n\nimport matplotlib\n\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nfrom rdkit import Chem, DataStructs, RDLogger\nfrom rdkit.Chem import rdFingerprintGenerator\nfrom rdkit.Chem.MolStandardize import rdMolStandardize\nfrom rdkit.Chem.Scaffolds import MurckoScaffold\n\n\nUSER_AGENT = "Nipah-G4-PubChem-compound-checkpoint/1.0 (external validation dataset build)"\nDEFAULT_OUTPUT_DIR = "data/pubchem_g4_bioassay_external_validation"\nDEFAULT_SELECTED = "data/pubchem_g4_bioassay_external_validation/selected_pubchem_g4_aids.csv"\nDEFAULT_TRAINING = "data/modeling/g4ldb_chembl_binary_dataset.csv"\nHIGH_CONCENTRATION_NM = 10_000.0\nTARGET_MIN = 800\nTARGET_MAX = 1200\nSIMILARITY_BINS = [-0.001, 0.4, 0.6, 0.8, 1.0]\nSIMILARITY_LABELS = ["lt_0_4", "0_4_to_0_6", "0_6_to_0_8", "ge_0_8"]\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--selected-aids", default=DEFAULT_SELECTED)\n    parser.add_argument("--training-dataset", default=DEFAULT_TRAINING)\n    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)\n    parser.add_argument("--sleep", type=float, default=0.35, help="Delay between PubChem calls.")\n    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")\n    parser.add_argument("--retries", type=int, default=3, help="Retries per AID.")\n    parser.add_argument("--fingerprint-radius", type=int, default=2)\n    parser.add_argument("--fingerprint-bits", type=int, default=2048)\n    return parser.parse_args()\n\n\ndef root_from_script() -> Path:\n    return Path(__file__).resolve().parents[1]\n\n\ndef resolve(path: str | Path, root: Path) -> Path:\n    path = Path(path)\n    return path if path.is_absolute() else root / path\n\n\ndef compact_join(values: pd.Series, limit: int = 30) -> str:\n    seen: list[str] = []\n    for value in values.dropna():\n        text = str(value).strip()\n        if text and text.lower() != "nan" and text not in seen:\n            seen.append(text)\n        if len(seen) >= limit:\n            break\n    return " | ".join(seen)\n\n\ndef first_nonempty(values: pd.Series) -> str:\n    for value in values.dropna():\n        text = str(value).strip()\n        if text and text.lower() != "nan":\n            return text\n    return ""\n\n\ndef normalize_text(value: Any) -> str:\n    if pd.isna(value):\n        return ""\n    return re.sub(r"\\s+", " ", str(value)).strip()\n\n\ndef fetch_assay_csv(aid: int, timeout: int, retries: int) -> tuple[str | None, str]:\n    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV"\n    last_error = ""\n    for attempt in range(1, retries + 1):\n        try:\n            request = Request(url, headers={"User-Agent": USER_AGENT})\n            with urlopen(request, timeout=timeout) as response:\n                return response.read().decode("utf-8", errors="replace"), ""\n        except (HTTPError, URLError, TimeoutError, OSError) as exc:\n            last_error = f"{type(exc).__name__}: {exc}"\n            time.sleep(1.5 * attempt)\n    return None, last_error\n\n\ndef read_pubchem_activity_csv(text: str) -> pd.DataFrame:\n    df = pd.read_csv(io.StringIO(text), dtype=str, low_memory=False)\n    if "PUBCHEM_RESULT_TAG" in df.columns:\n        tags = df["PUBCHEM_RESULT_TAG"].fillna("").astype(str)\n        df = df[tags.str.fullmatch(r"\\d+")].copy()\n    return df.reset_index(drop=True)\n\n\ndef find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:\n    exact = {column: column for column in df.columns}\n    lower = {column.lower(): column for column in df.columns}\n    for candidate in candidates:\n        if candidate in exact:\n            return exact[candidate]\n        if candidate.lower() in lower:\n            return lower[candidate.lower()]\n    return None\n\n\ndef to_float(value: Any) -> float:\n    if pd.isna(value):\n        return math.nan\n    text = str(value).strip().replace(",", "")\n    if not text:\n        return math.nan\n    match = re.search(r"[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?", text)\n    return float(match.group(0)) if match else math.nan\n\n\ndef standard_value_to_nm(value: Any, units: Any, pubchem_standard_um: Any = None) -> float:\n    numeric = to_float(value)\n    unit = normalize_text(units).lower()\n    if np.isfinite(numeric):\n        if unit in {"nm", "nanomolar", "nanomolars"}:\n            return numeric\n        if unit in {"um", "µm", "μm", "micromolar", "micromolars"}:\n            return numeric * 1000.0\n        if unit in {"mm", "millimolar", "millimolars"}:\n            return numeric * 1_000_000.0\n        if unit in {"m", "molar", "molars"}:\n            return numeric * 1_000_000_000.0\n        if unit in {"pm", "picomolar", "picomolars"}:\n            return numeric / 1000.0\n    pubchem_um = to_float(pubchem_standard_um)\n    if np.isfinite(pubchem_um):\n        return pubchem_um * 1000.0\n    return math.nan\n\n\ndef is_concentration_type(value: Any) -> bool:\n    text = normalize_text(value).lower()\n    if not text:\n        return False\n    return bool(\n        re.search(\n            r"\\b(ic50|ec50|ac50|gi50|cc50|lc50|kd|ki|kb|mic|minimum inhibitory|inhibitory concentration|binding affinity)\\b",\n            text,\n        )\n    )\n\n\ndef normalize_relation(value: Any) -> str:\n    text = normalize_text(value)\n    if not text:\n        return ""\n    match = re.match(r"^(<=|≥|>=|≤|<|>|=|~)", text)\n    if not match:\n        return text\n    relation = match.group(1)\n    return {">=": ">=", "≥": ">=", "<=": "<=", "≤": "<="}.get(relation, relation)\n\n\ndef label_activity_record(row: pd.Series) -> tuple[str, str, str]:\n    outcome = normalize_text(row.get("pubchem_activity_outcome", row.get("PUBCHEM_ACTIVITY_OUTCOME")))\n    if outcome == "Active":\n        return "active", "pubchem_activity_outcome_active", ""\n    if outcome == "Inactive":\n        return "inactive", "pubchem_activity_outcome_inactive", ""\n    if outcome in {"Inconclusive", "Probe", "Unspecified", "Untested"}:\n        relation = normalize_relation(row.get("standard_relation"))\n        standard_type = row.get("standard_type")\n        value_nm = to_float(row.get("standard_value_nM"))\n        if (\n            outcome == "Unspecified"\n            and is_concentration_type(standard_type)\n            and relation in {">", ">="}\n            and np.isfinite(value_nm)\n            and value_nm >= HIGH_CONCENTRATION_NM\n        ):\n            return (\n                "inactive_like_high_concentration",\n                "unspecified_high_concentration_greater_than_or_equal_10uM",\n                f"{standard_type} {relation} {value_nm:g} nM",\n            )\n        return "exclude_unclear", f"excluded_pubchem_outcome_{outcome or \'missing\'}", ""\n    return "exclude_unclear", f"excluded_pubchem_outcome_{outcome or \'missing\'}", ""\n\n\ndef standardize_mol(smiles: Any) -> tuple[Chem.Mol | None, str | None, str | None]:\n    if pd.isna(smiles):\n        return None, None, None\n    mol = Chem.MolFromSmiles(str(smiles))\n    if mol is None:\n        return None, None, None\n    try:\n        mol = rdMolStandardize.Cleanup(mol)\n        mol = rdMolStandardize.FragmentParent(mol)\n        mol = rdMolStandardize.Uncharger().uncharge(mol)\n    except Exception:\n        pass\n    try:\n        Chem.SanitizeMol(mol)\n    except Exception:\n        return None, None, None\n    canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)\n    inchikey = Chem.MolToInchiKey(mol)\n    return mol, canonical, inchikey\n\n\ndef collect_records(selected: pd.DataFrame, sleep: float, timeout: int, retries: int) -> tuple[pd.DataFrame, pd.DataFrame]:\n    rows: list[pd.DataFrame] = []\n    assay_status: list[dict[str, Any]] = []\n    selected = selected.copy()\n    selected["aid"] = selected["aid"].astype(int)\n    order = {"confirmed_G4_assay": 0, "probable_G4_assay": 1}\n    selected["_order"] = selected["curation_label"].map(order).fillna(9)\n    selected = selected.sort_values(["_order", "aid"]).reset_index(drop=True)\n\n    for index, assay in selected.iterrows():\n        aid = int(assay["aid"])\n        text, error = fetch_assay_csv(aid, timeout=timeout, retries=retries)\n        if text is None:\n            assay_status.append(\n                {\n                    "aid": aid,\n                    "curation_label": assay.get("curation_label", ""),\n                    "title_name": assay.get("title_name", ""),\n                    "download_status": "failed",\n                    "download_error": error,\n                    "activity_rows": 0,\n                }\n            )\n            time.sleep(sleep)\n            continue\n        try:\n            df = read_pubchem_activity_csv(text)\n        except Exception as exc:  # noqa: BLE001 - preserve exact parse failure in checkpoint.\n            assay_status.append(\n                {\n                    "aid": aid,\n                    "curation_label": assay.get("curation_label", ""),\n                    "title_name": assay.get("title_name", ""),\n                    "download_status": "parse_failed",\n                    "download_error": f"{type(exc).__name__}: {exc}",\n                    "activity_rows": 0,\n                }\n            )\n            time.sleep(sleep)\n            continue\n        df["aid"] = aid\n        df["assay_curation_label"] = assay.get("curation_label", "")\n        df["assay_title"] = assay.get("title_name", "")\n        df["assay_source"] = assay.get("source", "")\n        df["assay_source_id"] = assay.get("source_id", "")\n        df["assay_method"] = assay.get("method", "")\n        df["assay_pubchem_url"] = assay.get("pubchem_url", "")\n        rows.append(df)\n        assay_status.append(\n            {\n                "aid": aid,\n                "curation_label": assay.get("curation_label", ""),\n                "title_name": assay.get("title_name", ""),\n                "download_status": "ok",\n                "download_error": "",\n                "activity_rows": int(len(df)),\n            }\n        )\n        print(f"[{index + 1}/{len(selected)}] AID {aid}: {len(df)} activity rows")\n        time.sleep(sleep)\n\n    combined = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()\n    return combined, pd.DataFrame(assay_status)\n\n\ndef normalize_raw_records(raw: pd.DataFrame) -> pd.DataFrame:\n    if raw.empty:\n        return raw\n    cid_col = find_column(raw, ["PUBCHEM_CID", "CID"])\n    sid_col = find_column(raw, ["PUBCHEM_SID", "SID"])\n    smiles_col = find_column(raw, ["PUBCHEM_EXT_DATASOURCE_SMILES", "PUBCHEM_OPENEYE_CAN_SMILES", "SMILES"])\n    standard_type_col = find_column(raw, ["Standard Type", "STANDARD_TYPE"])\n    standard_relation_col = find_column(raw, ["Standard Relation", "STANDARD_RELATION"])\n    standard_value_col = find_column(raw, ["Standard Value", "STANDARD_VALUE"])\n    standard_units_col = find_column(raw, ["Standard Units", "STANDARD_UNITS"])\n    pubchem_standard_col = find_column(raw, ["PubChem Standard Value"])\n\n    out = pd.DataFrame(\n        {\n            "aid": raw.get("aid", ""),\n            "assay_curation_label": raw.get("assay_curation_label", ""),\n            "assay_title": raw.get("assay_title", ""),\n            "assay_source": raw.get("assay_source", ""),\n            "assay_source_id": raw.get("assay_source_id", ""),\n            "assay_method": raw.get("assay_method", ""),\n            "assay_pubchem_url": raw.get("assay_pubchem_url", ""),\n            "pubchem_result_tag": raw.get("PUBCHEM_RESULT_TAG", ""),\n            "pubchem_sid": raw[sid_col] if sid_col else "",\n            "pubchem_cid": raw[cid_col] if cid_col else "",\n            "external_datasource_smiles": raw[smiles_col] if smiles_col else "",\n            "pubchem_activity_outcome": raw.get("PUBCHEM_ACTIVITY_OUTCOME", ""),\n            "pubchem_activity_score": raw.get("PUBCHEM_ACTIVITY_SCORE", ""),\n            "pubchem_activity_url": raw.get("PUBCHEM_ACTIVITY_URL", ""),\n            "pubchem_assaydata_comment": raw.get("PUBCHEM_ASSAYDATA_COMMENT", ""),\n            "pubchem_standard_value_uM": raw[pubchem_standard_col] if pubchem_standard_col else "",\n            "standard_type": raw[standard_type_col] if standard_type_col else "",\n            "standard_relation": raw[standard_relation_col] if standard_relation_col else "",\n            "standard_value": raw[standard_value_col] if standard_value_col else "",\n            "standard_units": raw[standard_units_col] if standard_units_col else "",\n            "activity_comment": raw.get("Activity Comment", ""),\n        }\n    )\n    out["standard_relation"] = out["standard_relation"].apply(normalize_relation)\n    out["standard_value_nM"] = [\n        standard_value_to_nm(value, units, pubchem_um)\n        for value, units, pubchem_um in zip(\n            out["standard_value"], out["standard_units"], out["pubchem_standard_value_uM"]\n        )\n    ]\n    labels = out.apply(label_activity_record, axis=1)\n    out["record_label"] = [item[0] for item in labels]\n    out["label_source"] = [item[1] for item in labels]\n    out["label_detail"] = [item[2] for item in labels]\n    return out\n\n\ndef standardize_records(raw: pd.DataFrame) -> pd.DataFrame:\n    if raw.empty:\n        return raw\n    RDLogger.DisableLog("rdApp.*")\n    out = raw.copy()\n    standardized = out["external_datasource_smiles"].apply(standardize_mol)\n    out["canonical_smiles"] = [item[1] for item in standardized]\n    out["inchikey"] = [item[2] for item in standardized]\n    out["standardization_status"] = np.where(out["canonical_smiles"].notna(), "ok", "failed")\n    return out\n\n\ndef resolve_compound_label(group: pd.DataFrame) -> tuple[str | None, str, str]:\n    labels = set(group["record_label"].dropna().astype(str))\n    labeled_labels = labels - {"exclude_unclear"}\n    active_count = int((group["record_label"] == "active").sum())\n    inactive_count = int((group["record_label"] == "inactive").sum())\n    inactive_like_count = int((group["record_label"] == "inactive_like_high_concentration").sum())\n    if active_count:\n        if inactive_count or inactive_like_count:\n            return "active", "active_overrides_conservative_conflict", "active evidence present with inactive-like evidence"\n        return "active", "active_evidence", "active PubChem outcome present"\n    if inactive_count:\n        return "inactive", "inactive_evidence", "inactive PubChem outcome present"\n    if inactive_like_count:\n        return "inactive_like_high_concentration", "inactive_like_high_concentration_only", "only high-concentration greater-than unspecified evidence"\n    if labeled_labels:\n        return None, "unresolved_label_state", "unhandled non-clear label state"\n    return None, "excluded_no_clear_label", "only inconclusive/unclear records"\n\n\ndef deduplicate_compounds(standardized: pd.DataFrame) -> pd.DataFrame:\n    usable = standardized[standardized["standardization_status"].eq("ok")].copy()\n    usable = usable[usable["inchikey"].notna() | usable["canonical_smiles"].notna()].copy()\n    if usable.empty:\n        return pd.DataFrame()\n    usable["compound_key"] = usable["inchikey"].fillna(usable["canonical_smiles"])\n    rows: list[dict[str, Any]] = []\n    for key, group in usable.groupby("compound_key", sort=False):\n        compound_label, resolution, detail = resolve_compound_label(group)\n        active_aids = group.loc[group["record_label"].eq("active"), "aid"]\n        inactive_aids = group.loc[group["record_label"].eq("inactive"), "aid"]\n        inactive_like_aids = group.loc[group["record_label"].eq("inactive_like_high_concentration"), "aid"]\n        rows.append(\n            {\n                "compound_key": key,\n                "inchikey": first_nonempty(group["inchikey"]),\n                "canonical_smiles": first_nonempty(group["canonical_smiles"]),\n                "pubchem_cids": compact_join(group["pubchem_cid"]),\n                "pubchem_sids": compact_join(group["pubchem_sid"], limit=50),\n                "record_count": int(len(group)),\n                "aid_count": int(group["aid"].nunique()),\n                "aids": compact_join(group["aid"].astype(str), limit=80),\n                "confirmed_aids": compact_join(\n                    group.loc[group["assay_curation_label"].eq("confirmed_G4_assay"), "aid"].astype(str), limit=80\n                ),\n                "probable_aids": compact_join(\n                    group.loc[group["assay_curation_label"].eq("probable_G4_assay"), "aid"].astype(str), limit=80\n                ),\n                "assay_curation_labels": compact_join(group["assay_curation_label"]),\n                "activity_outcomes": compact_join(group["pubchem_activity_outcome"]),\n                "record_labels": compact_join(group["record_label"]),\n                "label_sources": compact_join(group["label_source"]),\n                "compound_label": compound_label,\n                "compound_label_source": resolution,\n                "compound_label_detail": detail,\n                "label": 1 if compound_label == "active" else (0 if compound_label in {"inactive", "inactive_like_high_concentration"} else np.nan),\n                "active_record_count": int((group["record_label"] == "active").sum()),\n                "inactive_record_count": int((group["record_label"] == "inactive").sum()),\n                "inactive_like_record_count": int((group["record_label"] == "inactive_like_high_concentration").sum()),\n                "excluded_unclear_record_count": int((group["record_label"] == "exclude_unclear").sum()),\n                "active_aids": compact_join(active_aids.astype(str), limit=80),\n                "inactive_aids": compact_join(inactive_aids.astype(str), limit=80),\n                "inactive_like_aids": compact_join(inactive_like_aids.astype(str), limit=80),\n                "standard_types": compact_join(group["standard_type"]),\n                "standard_relations": compact_join(group["standard_relation"]),\n                "standard_value_nM_min": float(group["standard_value_nM"].min(skipna=True))\n                if group["standard_value_nM"].notna().any()\n                else np.nan,\n                "standard_value_nM_median": float(group["standard_value_nM"].median(skipna=True))\n                if group["standard_value_nM"].notna().any()\n                else np.nan,\n                "assay_titles": compact_join(group["assay_title"], limit=12),\n            }\n        )\n    return pd.DataFrame(rows)\n\n\ndef load_training_mols(path: Path) -> pd.DataFrame:\n    df = pd.read_csv(path)\n    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"\n    if smiles_col not in df.columns:\n        raise SystemExit(f"Training dataset missing canonical_smiles/smiles column: {path}")\n    rows = []\n    for i, smiles in enumerate(df[smiles_col]):\n        mol, canonical, inchikey = standardize_mol(smiles)\n        if mol is None or canonical is None or inchikey is None:\n            continue\n        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)\n        if not scaffold:\n            scaffold = "acyclic::" + canonical\n        rows.append(\n            {\n                "training_index": i,\n                "training_canonical_smiles": canonical,\n                "training_inchikey": inchikey,\n                "training_murcko_scaffold": scaffold,\n                "mol": mol,\n            }\n        )\n    if not rows:\n        raise SystemExit("No parseable training molecules for similarity calculation.")\n    return pd.DataFrame(rows)\n\n\ndef add_training_context(compounds: pd.DataFrame, training: pd.DataFrame, radius: int, n_bits: int) -> pd.DataFrame:\n    if compounds.empty:\n        return compounds\n    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)\n    train_fps = [generator.GetFingerprint(mol) for mol in training["mol"]]\n    train_inchikeys = set(training["training_inchikey"])\n    train_scaffolds = set(training["training_murcko_scaffold"])\n    train_smiles = training["training_canonical_smiles"].tolist()\n    train_indices = training["training_index"].tolist()\n\n    rows = []\n    for _, row in compounds.iterrows():\n        mol = Chem.MolFromSmiles(row["canonical_smiles"])\n        if mol is None:\n            rows.append((np.nan, np.nan, "", "", False, "", False))\n            continue\n        fp = generator.GetFingerprint(mol)\n        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)\n        best_idx = int(np.argmax(sims))\n        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)\n        if not scaffold:\n            scaffold = "acyclic::" + row["canonical_smiles"]\n        rows.append(\n            (\n                float(sims[best_idx]),\n                int(train_indices[best_idx]),\n                train_smiles[best_idx],\n                scaffold,\n                bool(scaffold in train_scaffolds),\n                row["inchikey"],\n                bool(row["inchikey"] in train_inchikeys),\n            )\n        )\n    out = compounds.copy()\n    out["nearest_train_similarity"] = [item[0] for item in rows]\n    out["nearest_train_index"] = [item[1] for item in rows]\n    out["nearest_train_smiles"] = [item[2] for item in rows]\n    out["murcko_scaffold"] = [item[3] for item in rows]\n    out["training_scaffold_overlap"] = [item[4] for item in rows]\n    out["exact_training_inchikey_match"] = [item[6] for item in rows]\n    out["similarity_bin"] = pd.cut(\n        out["nearest_train_similarity"],\n        bins=SIMILARITY_BINS,\n        labels=SIMILARITY_LABELS,\n        include_lowest=True,\n    ).astype(str)\n    return out\n\n\ndef build_final_set(compounds: pd.DataFrame) -> pd.DataFrame:\n    labeled = compounds.dropna(subset=["label"]).copy()\n    labeled["label"] = labeled["label"].astype(int)\n    labeled = labeled[~labeled["exact_training_inchikey_match"]].copy()\n    labeled["label_name"] = np.where(labeled["label"].eq(1), "active", "inactive")\n    labeled["priority"] = np.select(\n        [\n            labeled["assay_curation_labels"].str.contains("confirmed_G4_assay", na=False),\n            labeled["assay_curation_labels"].str.contains("probable_G4_assay", na=False),\n        ],\n        [0, 1],\n        default=2,\n    )\n    labeled = labeled.sort_values(\n        ["priority", "label", "nearest_train_similarity", "record_count"],\n        ascending=[True, False, True, False],\n    ).reset_index(drop=True)\n\n    active = labeled[labeled["label"].eq(1)].copy()\n    inactive = labeled[labeled["label"].eq(0)].copy()\n    if active.empty or inactive.empty:\n        return labeled.head(TARGET_MAX).copy()\n\n    per_class_cap = TARGET_MAX // 2\n    take_each = min(per_class_cap, len(active), len(inactive))\n    rows = [active.head(take_each), inactive.head(take_each)]\n    remainder_slots = TARGET_MAX - sum(len(x) for x in rows)\n    if remainder_slots > 0:\n        chosen = pd.concat(rows, ignore_index=True)\n        remaining = labeled[~labeled["compound_key"].isin(set(chosen["compound_key"]))].copy()\n        rows.append(remaining.head(remainder_slots))\n    return pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)\n\n\ndef assay_level_summary(raw: pd.DataFrame, assay_status: pd.DataFrame) -> pd.DataFrame:\n    if raw.empty:\n        return assay_status\n    grouped = raw.groupby("aid", as_index=False).agg(\n        assay_curation_label=("assay_curation_label", first_nonempty),\n        assay_title=("assay_title", first_nonempty),\n        activity_rows=("aid", "size"),\n        unique_cids=("pubchem_cid", lambda x: int(pd.Series(x).replace("", np.nan).nunique())),\n        active_records=("record_label", lambda x: int((x == "active").sum())),\n        inactive_records=("record_label", lambda x: int((x == "inactive").sum())),\n        inactive_like_high_concentration_records=(\n            "record_label",\n            lambda x: int((x == "inactive_like_high_concentration").sum()),\n        ),\n        excluded_unclear_records=("record_label", lambda x: int((x == "exclude_unclear").sum())),\n    )\n    return assay_status.merge(grouped, on="aid", how="left", suffixes=("_download", ""))\n\n\ndef compound_level_summary(compounds: pd.DataFrame, final_set: pd.DataFrame) -> pd.DataFrame:\n    rows = []\n    stages = [\n        ("standardized_compounds", compounds),\n        ("labeled_compounds", compounds.dropna(subset=["label"]) if not compounds.empty else compounds),\n        (\n            "after_exact_training_overlap_removed",\n            compounds.dropna(subset=["label"])[lambda x: ~x["exact_training_inchikey_match"]]\n            if not compounds.empty and "exact_training_inchikey_match" in compounds.columns\n            else compounds.iloc[0:0],\n        ),\n        ("final_external_set", final_set),\n    ]\n    for stage, df in stages:\n        if df.empty:\n            rows.append({"stage": stage, "compounds": 0, "active": 0, "inactive": 0})\n            continue\n        labels = df.get("label", pd.Series(dtype=float))\n        rows.append(\n            {\n                "stage": stage,\n                "compounds": int(len(df)),\n                "active": int((labels == 1).sum()),\n                "inactive": int((labels == 0).sum()),\n                "confirmed_evidence_compounds": int(df["assay_curation_labels"].str.contains("confirmed_G4_assay", na=False).sum())\n                if "assay_curation_labels" in df.columns\n                else 0,\n                "probable_evidence_compounds": int(df["assay_curation_labels"].str.contains("probable_G4_assay", na=False).sum())\n                if "assay_curation_labels" in df.columns\n                else 0,\n                "training_scaffold_overlap": int(df.get("training_scaffold_overlap", pd.Series(dtype=bool)).fillna(False).sum()),\n                "exact_training_inchikey_match": int(df.get("exact_training_inchikey_match", pd.Series(dtype=bool)).fillna(False).sum()),\n            }\n        )\n    return pd.DataFrame(rows)\n\n\ndef plot_diagnostics(final_set: pd.DataFrame, output: Path) -> list[str]:\n    files: list[str] = []\n    if final_set.empty:\n        return files\n    counts = final_set["label_name"].value_counts().reindex(["active", "inactive"], fill_value=0)\n    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)\n    counts.plot(kind="bar", ax=ax, color=["#2f7d52", "#686f7a"])\n    ax.set_xlabel("")\n    ax.set_ylabel("Compounds")\n    ax.set_title("PubChem G4 final label counts")\n    plt.tight_layout()\n    fig_path = output / "pubchem_g4_final_label_counts.png"\n    plt.savefig(fig_path)\n    plt.close()\n    files.append(fig_path.name)\n\n    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)\n    final_set["nearest_train_similarity"].dropna().plot(kind="hist", bins=24, ax=ax, color="#376996")\n    ax.set_xlabel("Nearest training Morgan Tanimoto")\n    ax.set_ylabel("Compounds")\n    ax.set_title("PubChem G4 final similarity distribution")\n    plt.tight_layout()\n    fig_path = output / "pubchem_g4_final_similarity_distribution.png"\n    plt.savefig(fig_path)\n    plt.close()\n    files.append(fig_path.name)\n    return files\n\n\ndef write_report(\n    output: Path,\n    counts: dict[str, Any],\n    selected: pd.DataFrame,\n    assay_status: pd.DataFrame,\n    raw: pd.DataFrame,\n    standardized: pd.DataFrame,\n    compounds: pd.DataFrame,\n    final_set: pd.DataFrame,\n    figure_files: list[str],\n) -> None:\n    selected_counts = selected["curation_label"].value_counts(dropna=False).to_dict()\n    download_counts = assay_status["download_status"].value_counts(dropna=False).to_dict()\n    label_counts = raw["record_label"].value_counts(dropna=False).to_dict() if not raw.empty else {}\n    compound_label_counts = compounds["compound_label"].value_counts(dropna=False).to_dict() if not compounds.empty else {}\n    final_label_counts = final_set["label_name"].value_counts(dropna=False).to_dict() if not final_set.empty else {}\n    final_sim_counts = final_set["similarity_bin"].value_counts(dropna=False).to_dict() if not final_set.empty else {}\n    failures = assay_status[~assay_status["download_status"].eq("ok")].copy()\n    target_status = (\n        "met"\n        if TARGET_MIN <= len(final_set) <= TARGET_MAX\n        else f"not met; {len(final_set)} compounds available after conservative labeling, exact-overlap removal, and balancing"\n    )\n    probable_reason = (\n        "Probable assays were included because confirmed assay-level total_count was below the 800-compound target before compound-level deduplication."\n        if selected.loc[selected["curation_label"].eq("confirmed_G4_assay"), "total_count"].sum() < TARGET_MIN\n        else "Probable assays were available but not required by assay-level counts."\n    )\n\n    lines = [\n        "# PubChem G4 Compound-Level External Checkpoint",\n        "",\n        f"Generated: {datetime.now().isoformat(timespec=\'seconds\')}",\n        "",\n        "## Scope",\n        "",\n        "- Built compound-level PubChem BioAssay records and a checkpoint external set.",\n        "- Did not run the locked GNN, retrain models, or tune thresholds for model performance.",\n        f"- {probable_reason}",\n        "- Labels are traceable to PubChem activity outcomes or high-concentration greater-than evidence for unspecified rows.",\n        "",\n        "## Counts",\n        "",\n        f"- Selected AIDs: {len(selected)} ({selected_counts})",\n        f"- Download statuses: {download_counts}",\n        f"- Raw activity records: {len(raw)}",\n        f"- Raw record labels: {label_counts}",\n        f"- Standardized activity records: {int((standardized[\'standardization_status\'] == \'ok\').sum()) if not standardized.empty else 0}",\n        f"- Deduplicated compounds: {len(compounds)}",\n        f"- Compound labels: {compound_label_counts}",\n        f"- Exact training InChIKey overlaps removed from final consideration: {int(compounds.get(\'exact_training_inchikey_match\', pd.Series(dtype=bool)).fillna(False).sum()) if not compounds.empty else 0}",\n        f"- Final external set: {len(final_set)} ({final_label_counts})",\n        f"- Final similarity bins: {final_sim_counts}",\n        f"- Target 800-1200 compounds: {target_status}",\n        "",\n        "## Label Rules",\n        "",\n        "- `active`: `PUBCHEM_ACTIVITY_OUTCOME == Active`.",\n        "- `inactive`: `PUBCHEM_ACTIVITY_OUTCOME == Inactive`.",\n        f"- `inactive_like_high_concentration`: only `Unspecified` rows with concentration-style standard type, `>` or `>=`, and value >= {HIGH_CONCENTRATION_NM:g} nM.",\n        "- Inconclusive, untested, missing, and otherwise unclear rows are excluded from the final labeled set but preserved in the raw table.",\n        "- Compound conflicts are resolved conservatively: active evidence marks the compound active, with conflict provenance retained.",\n        "",\n        "## Output Files",\n        "",\n        "- `pubchem_g4_raw_activity_records.csv`",\n        "- `pubchem_g4_standardized_compounds.csv`",\n        "- `pubchem_g4_final_external_set.csv`",\n        "- `pubchem_g4_compound_level_summary.csv`",\n        "- `pubchem_g4_assay_level_summary.csv`",\n        "- `pubchem_g4_compound_checkpoint_report.md`",\n    ]\n    for figure in figure_files:\n        lines.append(f"- `{figure}`")\n    if not failures.empty:\n        lines.extend(["", "## Download Or Parse Failures", ""])\n        for _, row in failures.iterrows():\n            lines.append(f"- AID {row[\'aid\']}: {row[\'download_status\']} - {row[\'download_error\']}")\n    lines.extend(["", "## Parameters", ""])\n    for key, value in counts.items():\n        lines.append(f"- {key}: {value}")\n    (output / "pubchem_g4_compound_checkpoint_report.md").write_text("\\n".join(lines) + "\\n", encoding="utf-8")\n\n\ndef main() -> None:\n    args = parse_args()\n    root = root_from_script()\n    output = resolve(args.output_dir, root)\n    output.mkdir(parents=True, exist_ok=True)\n    selected_path = resolve(args.selected_aids, root)\n    training_path = resolve(args.training_dataset, root)\n    selected = pd.read_csv(selected_path)\n    if "curation_label" not in selected.columns or "aid" not in selected.columns:\n        raise SystemExit("Selected AID table must contain aid and curation_label columns.")\n    selected = selected[selected["curation_label"].isin(["confirmed_G4_assay", "probable_G4_assay"])].copy()\n    if selected.empty:\n        raise SystemExit("No confirmed/probable selected AIDs found.")\n\n    raw_downloaded, assay_status = collect_records(selected, args.sleep, args.timeout, args.retries)\n    raw = normalize_raw_records(raw_downloaded)\n    raw.to_csv(output / "pubchem_g4_raw_activity_records.csv", index=False)\n\n    standardized_records = standardize_records(raw)\n    compounds = deduplicate_compounds(standardized_records)\n    training = load_training_mols(training_path)\n    compounds = add_training_context(compounds, training, args.fingerprint_radius, args.fingerprint_bits)\n    compounds.to_csv(output / "pubchem_g4_standardized_compounds.csv", index=False)\n\n    final_set = build_final_set(compounds)\n    final_set.to_csv(output / "pubchem_g4_final_external_set.csv", index=False)\n\n    assay_summary = assay_level_summary(raw, assay_status)\n    assay_summary.to_csv(output / "pubchem_g4_assay_level_summary.csv", index=False)\n    compound_summary = compound_level_summary(compounds, final_set)\n    compound_summary.to_csv(output / "pubchem_g4_compound_level_summary.csv", index=False)\n    figures = plot_diagnostics(final_set, output)\n\n    counts = {\n        "selected_aids_path": str(selected_path),\n        "training_dataset_path": str(training_path),\n        "sleep_seconds": args.sleep,\n        "timeout_seconds": args.timeout,\n        "retries": args.retries,\n        "fingerprint_radius": args.fingerprint_radius,\n        "fingerprint_bits": args.fingerprint_bits,\n        "high_concentration_inactive_like_threshold_nM": HIGH_CONCENTRATION_NM,\n    }\n    write_report(output, counts, selected, assay_status, raw, standardized_records, compounds, final_set, figures)\n\n    print("Wrote PubChem G4 compound-level checkpoint outputs to", output)\n    print("Raw activity records:", len(raw))\n    print("Deduplicated compounds:", len(compounds))\n    print("Final external set:", len(final_set), Counter(final_set["label_name"]) if not final_set.empty else {})\n\n\nif __name__ == "__main__":\n    main()\n'


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
