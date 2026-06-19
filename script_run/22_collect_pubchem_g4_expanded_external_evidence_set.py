#!/usr/bin/env python3
"""Standalone v.4 reproducibility script: Build expanded PubChem G4 evidence set."""

STEP_ID = '22'
STEP_TITLE = 'Build expanded PubChem G4 evidence set'
EMBEDDED_SOURCE_NAME = '23_collect_pubchem_g4_expanded_external_evidence_set.py'
SEED_MODE = 'none'
EMBEDDED_SOURCE = '#!/usr/bin/env python3\n"""Build an expanded PubChem-derived G4 evidence set.\n\nThis workflow creates traceable evidence tables only. It deliberately does not\nrun GNN prediction, retrain any model, or tune a model threshold.\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport io\nimport json\nimport math\nimport time\nfrom collections import Counter\nfrom datetime import datetime\nfrom pathlib import Path\nfrom typing import Any\nfrom urllib.error import HTTPError, URLError\nfrom urllib.parse import urlencode\nfrom urllib.request import Request, urlopen\n\nimport matplotlib\n\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n\ntry:\n    from rdkit import Chem, DataStructs, RDLogger\n    from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors\n    from rdkit.Chem.MolStandardize import rdMolStandardize\n    from rdkit.Chem.Scaffolds import MurckoScaffold\n\n    RDKIT_AVAILABLE = True\nexcept Exception:  # noqa: BLE001 - report availability in checkpoint.\n    RDKIT_AVAILABLE = False\n\n\nUSER_AGENT = "Nipah-G4-PubChem-expanded-evidence-set/1.0"\nDEFAULT_OUTPUT_DIR = "data/pubchem_g4_expanded_external_evidence_set"\nDEFAULT_TIER1 = "data/pubchem_g4_bioassay_external_validation/pubchem_g4_standardized_compounds.csv"\nDEFAULT_TIER1_STRICT = "data/pubchem_g4_bioassay_external_validation/pubchem_g4_final_external_set.csv"\nDEFAULT_SELECTED_AIDS = "data/pubchem_g4_bioassay_external_validation/selected_pubchem_g4_aids.csv"\nDEFAULT_TRAINING = "data/modeling/g4ldb_chembl_binary_dataset.csv"\nSIMILARITY_BINS = [-0.001, 0.4, 0.6, 0.8, 1.0]\nSIMILARITY_LABELS = ["lt_0_4", "0_4_to_0_6", "0_6_to_0_8", "ge_0_8"]\nPROPERTY_COLUMNS = [\n    "CanonicalSMILES",\n    "IsomericSMILES",\n    "InChIKey",\n    "MolecularWeight",\n    "XLogP",\n    "HBondDonorCount",\n    "HBondAcceptorCount",\n    "TPSA",\n    "RotatableBondCount",\n    "Title",\n]\nTIER2_QUERIES = [\n    "G-quadruplex binder",\n    "G-quadruplex ligand",\n    "G4 stabilizer",\n    "telomeric G-quadruplex ligand",\n    "c-MYC G-quadruplex ligand",\n    "BCL2 G-quadruplex ligand",\n    "KRAS G-quadruplex ligand",\n    "quadruplex stabilizer",\n    "FRET melting G-quadruplex",\n    "telomerase inhibitor",\n]\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)\n    parser.add_argument("--tier1-compounds", default=DEFAULT_TIER1)\n    parser.add_argument("--tier1-strict-set", default=DEFAULT_TIER1_STRICT)\n    parser.add_argument("--selected-aids", default=DEFAULT_SELECTED_AIDS)\n    parser.add_argument("--training-dataset", default=DEFAULT_TRAINING)\n    parser.add_argument("--tier2-retmax", type=int, default=250)\n    parser.add_argument("--background-retmax", type=int, default=350)\n    parser.add_argument("--sleep", type=float, default=0.34)\n    parser.add_argument("--timeout", type=int, default=45)\n    parser.add_argument("--fingerprint-radius", type=int, default=2)\n    parser.add_argument("--fingerprint-bits", type=int, default=2048)\n    return parser.parse_args()\n\n\ndef root_from_script() -> Path:\n    return Path(__file__).resolve().parents[1]\n\n\ndef resolve(path: str | Path, root: Path) -> Path:\n    path = Path(path)\n    return path if path.is_absolute() else root / path\n\n\ndef request_json(url: str, timeout: int) -> dict[str, Any]:\n    request = Request(url, headers={"User-Agent": USER_AGENT})\n    with urlopen(request, timeout=timeout) as response:\n        return json.loads(response.read().decode("utf-8", errors="replace"))\n\n\ndef request_text(url: str, timeout: int) -> str:\n    request = Request(url, headers={"User-Agent": USER_AGENT})\n    with urlopen(request, timeout=timeout) as response:\n        return response.read().decode("utf-8", errors="replace")\n\n\ndef entrez_esearch(term: str, retmax: int, timeout: int) -> dict[str, Any]:\n    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urlencode(\n        {"db": "pccompound", "term": term, "retmode": "json", "retmax": retmax}\n    )\n    return request_json(url, timeout=timeout)\n\n\ndef fetch_pubchem_properties(cids: list[str], timeout: int, sleep: float, batch_size: int = 100) -> pd.DataFrame:\n    frames: list[pd.DataFrame] = []\n    props = ",".join(PROPERTY_COLUMNS)\n    for start in range(0, len(cids), batch_size):\n        batch = [str(cid) for cid in cids[start : start + batch_size]]\n        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{\',\'.join(batch)}/property/{props}/CSV"\n        text = request_text(url, timeout=timeout)\n        frames.append(pd.read_csv(io.StringIO(text), dtype=str))\n        time.sleep(sleep)\n    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()\n\n\ndef first_nonempty(values: pd.Series) -> str:\n    for value in values.dropna():\n        text = str(value).strip()\n        if text and text.lower() != "nan":\n            return text\n    return ""\n\n\ndef compact_join(values: pd.Series, limit: int = 20) -> str:\n    seen: list[str] = []\n    for value in values.dropna():\n        text = str(value).strip()\n        if text and text.lower() != "nan" and text not in seen:\n            seen.append(text)\n        if len(seen) >= limit:\n            break\n    return " | ".join(seen)\n\n\ndef standardize_smiles(smiles: Any) -> tuple[Any, str, str]:\n    if not RDKIT_AVAILABLE or pd.isna(smiles):\n        return None, "", ""\n    mol = Chem.MolFromSmiles(str(smiles))\n    if mol is None:\n        return None, "", ""\n    try:\n        mol = rdMolStandardize.Cleanup(mol)\n        mol = rdMolStandardize.FragmentParent(mol)\n        mol = rdMolStandardize.Uncharger().uncharge(mol)\n    except Exception:\n        pass\n    try:\n        Chem.SanitizeMol(mol)\n    except Exception:\n        return None, "", ""\n    return mol, Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True), Chem.MolToInchiKey(mol)\n\n\ndef rdkit_descriptor_row(smiles: str) -> dict[str, Any]:\n    if not RDKIT_AVAILABLE:\n        return {}\n    mol, canonical, inchikey = standardize_smiles(smiles)\n    if mol is None:\n        return {"valid_molecule": False, "canonical_smiles": "", "inchikey": ""}\n    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)\n    if not scaffold:\n        scaffold = "acyclic::" + canonical\n    return {\n        "valid_molecule": True,\n        "canonical_smiles": canonical,\n        "inchikey": inchikey,\n        "mw": float(Descriptors.MolWt(mol)),\n        "logp": float(Crippen.MolLogP(mol)),\n        "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),\n        "hba": int(rdMolDescriptors.CalcNumHBA(mol)),\n        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),\n        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),\n        "murcko_scaffold": scaffold,\n    }\n\n\ndef add_rdkit_descriptors(df: pd.DataFrame) -> pd.DataFrame:\n    out = df.copy()\n    if out.empty:\n        return out\n    if not RDKIT_AVAILABLE:\n        out["valid_molecule"] = pd.notna(out.get("canonical_smiles", ""))\n        return out\n    RDLogger.DisableLog("rdApp.*")\n    smiles_col = "canonical_smiles" if "canonical_smiles" in out.columns else "smiles"\n    descriptors = [rdkit_descriptor_row(smiles) for smiles in out[smiles_col].fillna("")]\n    desc_df = pd.DataFrame(descriptors)\n    for col in ["canonical_smiles", "inchikey", "mw", "logp", "hbd", "hba", "tpsa", "rotatable_bonds", "murcko_scaffold"]:\n        if col in desc_df.columns:\n            out[col] = desc_df[col]\n    out["valid_molecule"] = desc_df.get("valid_molecule", False)\n    return out\n\n\ndef load_tier1(path: Path, strict_path: Path) -> pd.DataFrame:\n    df = pd.read_csv(path)\n    strict_keys: set[str] = set()\n    if strict_path.exists():\n        strict = pd.read_csv(strict_path)\n        strict_keys = set(strict.get("inchikey", pd.Series(dtype=str)).dropna().astype(str))\n    label_map = {\n        "active": "assay_confirmed_active",\n        "inactive": "assay_confirmed_inactive",\n        "inactive_like_high_concentration": "assay_derived_inactive_like_high_concentration",\n    }\n    rows = []\n    for _, row in df.iterrows():\n        compound_label = row.get("compound_label")\n        if pd.isna(compound_label):\n            conservative_label = "unresolved_assay_evidence"\n        else:\n            conservative_label = label_map.get(str(compound_label), "unresolved_assay_evidence")\n        rows.append(\n            {\n                "evidence_tier": "tier1_assay_confirmed_or_derived",\n                "conservative_label": conservative_label,\n                "binary_label_for_evaluation": row.get("label") if conservative_label.startswith("assay_") else np.nan,\n                "pubchem_cids": row.get("pubchem_cids", ""),\n                "pubchem_title": "",\n                "canonical_smiles": row.get("canonical_smiles", ""),\n                "inchikey": row.get("inchikey", ""),\n                "evidence_query": "existing_pubchem_bioassay_workflow",\n                "evidence_source": "PubChem BioAssay",\n                "evidence_detail": row.get("compound_label_detail", ""),\n                "evidence_record_count": row.get("record_count", np.nan),\n                "aid_count": row.get("aid_count", np.nan),\n                "aids": row.get("aids", ""),\n                "assay_titles": row.get("assay_titles", ""),\n                "tier1_was_in_previous_strict_external_set": str(row.get("inchikey", "")) in strict_keys,\n            }\n        )\n    return pd.DataFrame(rows)\n\n\ndef collect_tier2(output: Path, retmax: int, timeout: int, sleep: float) -> tuple[pd.DataFrame, list[str]]:\n    raw_dir = output / "raw_pubchem_queries"\n    raw_dir.mkdir(parents=True, exist_ok=True)\n    rows: list[dict[str, Any]] = []\n    blockers: list[str] = []\n    for query in TIER2_QUERIES:\n        try:\n            data = entrez_esearch(query, retmax=retmax, timeout=timeout)\n            safe = query.lower().replace(" ", "_").replace("/", "_")\n            (raw_dir / f"tier2_esearch_{safe}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")\n            result = data.get("esearchresult", {})\n            ids = [str(x) for x in result.get("idlist", [])]\n            for rank, cid in enumerate(ids, start=1):\n                rows.append(\n                    {\n                        "cid": cid,\n                        "evidence_query": query,\n                        "query_rank": rank,\n                        "entrez_count_for_query": result.get("count", ""),\n                    }\n                )\n        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:\n            blockers.append(f"Tier 2 query failed for {query!r}: {type(exc).__name__}: {exc}")\n        time.sleep(sleep)\n    if not rows:\n        return pd.DataFrame(), blockers\n    query_hits = pd.DataFrame(rows)\n    query_hits.to_csv(raw_dir / "tier2_query_cid_hits.csv", index=False)\n    cids = list(dict.fromkeys(query_hits["cid"].astype(str)))\n    try:\n        props = fetch_pubchem_properties(cids, timeout=timeout, sleep=sleep)\n        props.to_csv(raw_dir / "tier2_pubchem_properties_raw.csv", index=False)\n    except (HTTPError, URLError, TimeoutError, OSError) as exc:\n        blockers.append(f"Tier 2 property fetch failed: {type(exc).__name__}: {exc}")\n        return pd.DataFrame(), blockers\n    props["CID"] = props["CID"].astype(str)\n    merged = props.merge(query_hits.groupby("cid", as_index=False).agg(\n        evidence_query=("evidence_query", compact_join),\n        query_rank=("query_rank", "min"),\n        entrez_count_for_query=("entrez_count_for_query", compact_join),\n    ), left_on="CID", right_on="cid", how="left")\n    out = pd.DataFrame(\n        {\n            "evidence_tier": "tier2_pubchem_annotation_literature_supported",\n            "conservative_label": "g4_active_like_annotation",\n            "binary_label_for_evaluation": np.nan,\n            "pubchem_cids": merged["CID"],\n            "pubchem_title": merged.get("Title", ""),\n            "canonical_smiles": merged.get("SMILES", merged.get("CanonicalSMILES", "")),\n            "inchikey": merged.get("InChIKey", ""),\n            "evidence_query": merged["evidence_query"],\n            "evidence_source": "NCBI Entrez pccompound / PubChem Compound properties",\n            "evidence_detail": "Retrieved from PubChem Compound text search terms; active-like annotation only, not assay-confirmed here.",\n            "evidence_record_count": 1,\n            "aid_count": np.nan,\n            "aids": "",\n            "assay_titles": "",\n            "tier1_was_in_previous_strict_external_set": False,\n        }\n    )\n    return out, blockers\n\n\ndef property_query_from_reference(reference: pd.DataFrame, retmax: int) -> str:\n    if reference.empty:\n        return f"300:800[MW] AND 0:8[XLogP]"\n    mw = pd.to_numeric(reference.get("mw"), errors="coerce").dropna()\n    logp = pd.to_numeric(reference.get("logp"), errors="coerce").dropna()\n    if mw.empty or logp.empty:\n        return f"300:800[MW] AND 0:8[XLogP]"\n    mw_low = max(150, math.floor(mw.quantile(0.05) / 25) * 25)\n    mw_high = min(1000, math.ceil(mw.quantile(0.95) / 25) * 25)\n    logp_low = max(-3, math.floor(logp.quantile(0.05)))\n    logp_high = min(12, math.ceil(logp.quantile(0.95)))\n    return f"{mw_low}:{mw_high}[MW] AND {logp_low}:{logp_high}[XLogP]"\n\n\ndef collect_tier3_background(\n    output: Path,\n    reference: pd.DataFrame,\n    exclude_cids: set[str],\n    retmax: int,\n    timeout: int,\n    sleep: float,\n) -> tuple[pd.DataFrame, list[str], str]:\n    raw_dir = output / "raw_pubchem_queries"\n    raw_dir.mkdir(parents=True, exist_ok=True)\n    blockers: list[str] = []\n    query = property_query_from_reference(reference, retmax=retmax)\n    try:\n        data = entrez_esearch(query, retmax=retmax, timeout=timeout)\n        (raw_dir / "tier3_background_esearch.json").write_text(json.dumps(data, indent=2), encoding="utf-8")\n        ids = [str(cid) for cid in data.get("esearchresult", {}).get("idlist", []) if str(cid) not in exclude_cids]\n    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:\n        return pd.DataFrame(), [f"Tier 3 background query failed: {type(exc).__name__}: {exc}"], query\n    if not ids:\n        return pd.DataFrame(), blockers, query\n    try:\n        props = fetch_pubchem_properties(ids[:retmax], timeout=timeout, sleep=sleep)\n        props.to_csv(raw_dir / "tier3_background_pubchem_properties_raw.csv", index=False)\n    except (HTTPError, URLError, TimeoutError, OSError) as exc:\n        return pd.DataFrame(), [f"Tier 3 property fetch failed: {type(exc).__name__}: {exc}"], query\n    out = pd.DataFrame(\n        {\n            "evidence_tier": "tier3_property_matched_pubchem_background",\n            "conservative_label": "background_putative_decoy",\n            "binary_label_for_evaluation": np.nan,\n            "pubchem_cids": props["CID"].astype(str),\n            "pubchem_title": props.get("Title", ""),\n            "canonical_smiles": props.get("SMILES", props.get("CanonicalSMILES", "")),\n            "inchikey": props.get("InChIKey", ""),\n            "evidence_query": query,\n            "evidence_source": "NCBI Entrez pccompound property-range query / PubChem Compound properties",\n            "evidence_detail": "Property-shaped PubChem background only; not an experimentally confirmed G4 inactive class.",\n            "evidence_record_count": 1,\n            "aid_count": np.nan,\n            "aids": "",\n            "assay_titles": "",\n            "tier1_was_in_previous_strict_external_set": False,\n        }\n    )\n    return out, blockers, query\n\n\ndef load_training(path: Path) -> pd.DataFrame:\n    df = pd.read_csv(path)\n    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"\n    if smiles_col not in df.columns:\n        return pd.DataFrame()\n    rows: list[dict[str, Any]] = []\n    for i, smiles in enumerate(df[smiles_col].fillna("")):\n        mol, canonical, inchikey = standardize_smiles(smiles)\n        if mol is None:\n            continue\n        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)\n        if not scaffold:\n            scaffold = "acyclic::" + canonical\n        rows.append(\n            {\n                "training_index": i,\n                "training_canonical_smiles": canonical,\n                "training_inchikey": inchikey,\n                "training_murcko_scaffold": scaffold,\n                "training_label": df.get("label", pd.Series(dtype=object)).iloc[i] if "label" in df.columns else np.nan,\n                "mol": mol,\n            }\n        )\n    return pd.DataFrame(rows)\n\n\ndef add_training_context(df: pd.DataFrame, training: pd.DataFrame, radius: int, n_bits: int) -> pd.DataFrame:\n    out = df.copy()\n    if out.empty:\n        return out\n    if not RDKIT_AVAILABLE or training.empty:\n        out["exact_training_inchikey_match"] = False\n        out["nearest_train_similarity"] = np.nan\n        out["similarity_bin"] = "not_available"\n        return out\n    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)\n    train_fps = [generator.GetFingerprint(mol) for mol in training["mol"]]\n    train_inchikeys = set(training["training_inchikey"])\n    train_smiles = training["training_canonical_smiles"].tolist()\n    train_indices = training["training_index"].tolist()\n    train_labels = training["training_label"].tolist()\n    records = []\n    for _, row in out.iterrows():\n        mol = Chem.MolFromSmiles(str(row.get("canonical_smiles", "")))\n        if mol is None:\n            records.append((False, np.nan, np.nan, "", np.nan))\n            continue\n        fp = generator.GetFingerprint(mol)\n        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)\n        best_idx = int(np.argmax(sims))\n        records.append(\n            (\n                str(row.get("inchikey", "")) in train_inchikeys,\n                float(sims[best_idx]),\n                int(train_indices[best_idx]),\n                train_smiles[best_idx],\n                train_labels[best_idx],\n            )\n        )\n    out["exact_training_inchikey_match"] = [x[0] for x in records]\n    out["nearest_train_similarity"] = [x[1] for x in records]\n    out["nearest_train_index"] = [x[2] for x in records]\n    out["nearest_train_smiles"] = [x[3] for x in records]\n    out["nearest_train_label"] = [x[4] for x in records]\n    out["similarity_bin"] = pd.cut(\n        out["nearest_train_similarity"],\n        bins=SIMILARITY_BINS,\n        labels=SIMILARITY_LABELS,\n        include_lowest=True,\n    ).astype(str)\n    return out\n\n\ndef deduplicate_evidence(df: pd.DataFrame) -> pd.DataFrame:\n    if df.empty:\n        return df\n    df = df[df["valid_molecule"].fillna(False)].copy()\n    df["dedupe_key"] = df["inchikey"].replace("", np.nan).fillna(df["canonical_smiles"])\n    tier_order = {\n        "tier1_assay_confirmed_or_derived": 1,\n        "tier2_pubchem_annotation_literature_supported": 2,\n        "tier3_property_matched_pubchem_background": 3,\n    }\n    df["_tier_order"] = df["evidence_tier"].map(tier_order).fillna(9)\n    rows = []\n    for _, group in df.sort_values(["_tier_order"]).groupby("dedupe_key", sort=False):\n        first = group.iloc[0].to_dict()\n        first["merged_evidence_tiers"] = compact_join(group["evidence_tier"])\n        first["merged_conservative_labels"] = compact_join(group["conservative_label"])\n        first["merged_evidence_queries"] = compact_join(group["evidence_query"], limit=40)\n        first["merged_pubchem_cids"] = compact_join(group["pubchem_cids"].astype(str), limit=40)\n        first["merged_evidence_sources"] = compact_join(group["evidence_source"])\n        if (group["evidence_tier"] == "tier1_assay_confirmed_or_derived").any():\n            tier1 = group[group["evidence_tier"] == "tier1_assay_confirmed_or_derived"].iloc[0]\n            first["evidence_tier"] = tier1["evidence_tier"]\n            first["conservative_label"] = tier1["conservative_label"]\n            first["binary_label_for_evaluation"] = tier1["binary_label_for_evaluation"]\n        elif (group["evidence_tier"] == "tier2_pubchem_annotation_literature_supported").any():\n            first["evidence_tier"] = "tier2_pubchem_annotation_literature_supported"\n            first["conservative_label"] = "g4_active_like_annotation"\n            first["binary_label_for_evaluation"] = np.nan\n        else:\n            first["evidence_tier"] = "tier3_property_matched_pubchem_background"\n            first["conservative_label"] = "background_putative_decoy"\n            first["binary_label_for_evaluation"] = np.nan\n        rows.append(first)\n    out = pd.DataFrame(rows).drop(columns=["_tier_order"], errors="ignore")\n    return out.sort_values(["evidence_tier", "conservative_label", "nearest_train_similarity"], na_position="last").reset_index(drop=True)\n\n\ndef summarize(df: pd.DataFrame) -> pd.DataFrame:\n    if df.empty:\n        return pd.DataFrame()\n    return (\n        df.groupby(["evidence_tier", "conservative_label"], dropna=False)\n        .agg(\n            compounds=("dedupe_key", "count"),\n            exact_training_overlaps=("exact_training_inchikey_match", "sum"),\n            median_nearest_training_similarity=("nearest_train_similarity", "median"),\n            previous_strict_tier1_members=("tier1_was_in_previous_strict_external_set", "sum"),\n        )\n        .reset_index()\n    )\n\n\ndef plot_outputs(df: pd.DataFrame, output: Path) -> list[str]:\n    files: list[str] = []\n    if df.empty:\n        return files\n    tier_counts = df["evidence_tier"].value_counts()\n    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)\n    tier_counts.plot(kind="bar", ax=ax, color=["#376996", "#2f7d52", "#8a6f3d"])\n    ax.set_xlabel("")\n    ax.set_ylabel("Compounds")\n    ax.set_title("Expanded PubChem G4 evidence tiers")\n    plt.tight_layout()\n    path = output / "pubchem_g4_expanded_tier_counts.png"\n    plt.savefig(path)\n    plt.close()\n    files.append(path.name)\n\n    overlap = df["exact_training_inchikey_match"].value_counts().rename(index={True: "exact overlap", False: "no exact overlap"})\n    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)\n    overlap.plot(kind="bar", ax=ax, color=["#8f4d4d", "#4e7d67"])\n    ax.set_xlabel("")\n    ax.set_ylabel("Compounds")\n    ax.set_title("Training InChIKey overlap")\n    plt.tight_layout()\n    path = output / "pubchem_g4_expanded_training_overlap_counts.png"\n    plt.savefig(path)\n    plt.close()\n    files.append(path.name)\n\n    if df["nearest_train_similarity"].notna().any():\n        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)\n        df["nearest_train_similarity"].dropna().plot(kind="hist", bins=30, ax=ax, color="#455a64")\n        ax.set_xlabel("Nearest training Morgan Tanimoto")\n        ax.set_ylabel("Compounds")\n        ax.set_title("Expanded evidence nearest-training similarity")\n        plt.tight_layout()\n        path = output / "pubchem_g4_expanded_similarity_distribution.png"\n        plt.savefig(path)\n        plt.close()\n        files.append(path.name)\n\n    props = [("mw", "Molecular weight"), ("logp", "RDKit LogP"), ("tpsa", "TPSA")]\n    available = [p for p in props if p[0] in df.columns and df[p[0]].notna().any()]\n    if available:\n        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4), dpi=180)\n        if len(available) == 1:\n            axes = [axes]\n        for ax, (col, label) in zip(axes, available):\n            df.boxplot(column=col, by="evidence_tier", ax=ax, rot=45)\n            ax.set_title(label)\n            ax.set_xlabel("")\n        fig.suptitle("Property distributions by evidence tier")\n        plt.tight_layout()\n        path = output / "pubchem_g4_expanded_property_distributions.png"\n        plt.savefig(path)\n        plt.close()\n        files.append(path.name)\n    return files\n\n\ndef write_report(\n    output: Path,\n    all_df: pd.DataFrame,\n    strict_df: pd.DataFrame,\n    summary: pd.DataFrame,\n    blockers: list[str],\n    figures: list[str],\n    args: argparse.Namespace,\n    tier3_query: str,\n) -> None:\n    counts = all_df["evidence_tier"].value_counts().to_dict() if not all_df.empty else {}\n    label_counts = all_df["conservative_label"].value_counts().to_dict() if not all_df.empty else {}\n    sim_bins = all_df["similarity_bin"].value_counts().to_dict() if "similarity_bin" in all_df else {}\n    text = [\n        "# Expanded PubChem G4 Evidence Set Checkpoint",\n        "",\n        f"Generated: {datetime.now().isoformat(timespec=\'seconds\')}",\n        "",\n        "## Scope",\n        "",\n        "- Built a PubChem-derived evidence dataset for G4 model evaluation/checkpointing.",\n        "- Did not run GNN prediction, retrain models, or tune thresholds.",\n        "- This is separate from `data/pubchem_g4_bioassay_external_validation/`.",\n        "- Tier 2 rows are annotation/literature-search active-like candidates, not assay-confirmed rows.",\n        "- Tier 3 rows are property-shaped PubChem background candidates, not true inactive compounds.",\n        "",\n        "## Counts",\n        "",\n        f"- All nonredundant compounds: {len(all_df)}",\n        f"- Strict no-training-overlap compounds: {len(strict_df)}",\n        f"- Evidence tier counts: {counts}",\n        f"- Conservative label counts: {label_counts}",\n        f"- Exact training InChIKey overlaps: {int(all_df[\'exact_training_inchikey_match\'].sum()) if not all_df.empty else 0}",\n        f"- Similarity bins: {sim_bins}",\n        f"- Tier 3 background query: `{tier3_query}`",\n        "",\n        "## Evidence Tiers",\n        "",\n        "- Tier 1: assay-confirmed / assay-derived G4 active, inactive, or high-concentration inactive-like evidence reused from the existing PubChem BioAssay workflow.",\n        "- Tier 2: PubChem Compound/Entrez text-search hits for G4 binder/ligand/stabilizer/telomerase terms. These are active-like annotations only.",\n        "- Tier 3: PubChem property-range background candidates. These can serve as putative decoys/background only.",\n        "",\n        "## Manuscript-Safe Interpretation",\n        "",\n        "Use this dataset as an expanded PubChem-derived evidence checkpoint and chemical-space stress test. "\n        "Only Tier 1 should be described as assay-confirmed or assay-derived. Tier 2 supports literature/annotation-aware active-like enrichment, "\n        "and Tier 3 supports background contrast without claiming experimental G4 inactivity.",\n        "",\n        "## Output Files",\n        "",\n        "- `pubchem_g4_expanded_all_evidence.csv`",\n        "- `pubchem_g4_expanded_strict_no_training_overlap.csv`",\n        "- `pubchem_g4_expanded_tier_summary.csv`",\n        "- `pubchem_g4_expanded_decoy_background.csv`",\n        "- `pubchem_g4_expanded_checkpoint_report.md`",\n    ]\n    for figure in figures:\n        text.append(f"- `{figure}`")\n    if blockers:\n        text.extend(["", "## Blockers / Partial Collection Notes", ""])\n        text.extend([f"- {blocker}" for blocker in blockers])\n    text.extend(\n        [\n            "",\n            "## Parameters",\n            "",\n            f"- tier1_compounds: {args.tier1_compounds}",\n            f"- tier1_strict_set: {args.tier1_strict_set}",\n            f"- selected_aids: {args.selected_aids}",\n            f"- training_dataset: {args.training_dataset}",\n            f"- tier2_retmax: {args.tier2_retmax}",\n            f"- background_retmax: {args.background_retmax}",\n            f"- sleep_seconds: {args.sleep}",\n            f"- timeout_seconds: {args.timeout}",\n            f"- rdkit_available: {RDKIT_AVAILABLE}",\n        ]\n    )\n    output.joinpath("pubchem_g4_expanded_checkpoint_report.md").write_text("\\n".join(text) + "\\n", encoding="utf-8")\n\n\ndef main() -> None:\n    args = parse_args()\n    root = root_from_script()\n    output = resolve(args.output_dir, root)\n    output.mkdir(parents=True, exist_ok=True)\n\n    tier1 = load_tier1(resolve(args.tier1_compounds, root), resolve(args.tier1_strict_set, root))\n    tier1 = add_rdkit_descriptors(tier1)\n    tier2, tier2_blockers = collect_tier2(output, args.tier2_retmax, args.timeout, args.sleep)\n    tier2 = add_rdkit_descriptors(tier2) if not tier2.empty else tier2\n    reference_for_background = pd.concat([tier1, tier2], ignore_index=True, sort=False)\n    exclude_cids = set()\n    for value in reference_for_background.get("pubchem_cids", pd.Series(dtype=str)).dropna().astype(str):\n        exclude_cids.update([part.strip() for part in value.split("|") if part.strip()])\n    tier3, tier3_blockers, tier3_query = collect_tier3_background(\n        output, reference_for_background, exclude_cids, args.background_retmax, args.timeout, args.sleep\n    )\n    tier3 = add_rdkit_descriptors(tier3) if not tier3.empty else tier3\n\n    combined = pd.concat([tier1, tier2, tier3], ignore_index=True, sort=False)\n    combined = add_rdkit_descriptors(combined)\n    training = load_training(resolve(args.training_dataset, root)) if RDKIT_AVAILABLE else pd.DataFrame()\n    combined = add_training_context(combined, training, args.fingerprint_radius, args.fingerprint_bits)\n    all_df = deduplicate_evidence(combined)\n    strict_df = all_df[~all_df["exact_training_inchikey_match"].fillna(False)].copy()\n    summary = summarize(all_df)\n\n    all_df.to_csv(output / "pubchem_g4_expanded_all_evidence.csv", index=False)\n    strict_df.to_csv(output / "pubchem_g4_expanded_strict_no_training_overlap.csv", index=False)\n    summary.to_csv(output / "pubchem_g4_expanded_tier_summary.csv", index=False)\n    if not tier3.empty:\n        tier3.to_csv(output / "pubchem_g4_expanded_decoy_background.csv", index=False)\n    else:\n        pd.DataFrame().to_csv(output / "pubchem_g4_expanded_decoy_background.csv", index=False)\n\n    figures = plot_outputs(all_df, output)\n    blockers = tier2_blockers + tier3_blockers\n    write_report(output, all_df, strict_df, summary, blockers, figures, args, tier3_query)\n\n    print(f"Wrote {len(all_df)} nonredundant PubChem evidence compounds to {output}")\n    print(f"Strict no-training-overlap rows: {len(strict_df)}")\n    print(f"Tier counts: {Counter(all_df[\'evidence_tier\']) if not all_df.empty else {}}")\n    print("GNN prediction was not run; no training or threshold tuning was performed.")\n\n\nif __name__ == "__main__":\n    main()\n'


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
