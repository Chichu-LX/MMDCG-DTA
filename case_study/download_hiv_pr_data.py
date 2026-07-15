#!/usr/bin/env python3
"""
Download real HIV-1 Protease complex data from public databases.
Sources: RCSB PDB, ChEMBL, BindingDB

This script:
1. Queries PDB for HIV-1 PR structures (EC 3.4.23.16)
2. Gets metadata and ligand info for each structure
3. Queries ChEMBL for binding affinity data (Ki/Kd)
4. Queries BindingDB for supplementary affinity data
5. Cross-references PDB structures with binding data
6. Downloads PDB files and ligand SDF files
7. Compiles a 50+ complex dataset with complete provenance
"""

import json
import os
import sys
import time
import hashlib
from pathlib import Path
from datetime import datetime

import requests
from io import StringIO

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path('/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study/hiv_pr_real_data')
DATA_DIR.mkdir(parents=True, exist_ok=True)

PDB_DIR = DATA_DIR / 'pdb_files'
SDF_DIR = DATA_DIR / 'sdf_files'
PDB_DIR.mkdir(exist_ok=True)
SDF_DIR.mkdir(exist_ok=True)

# Rate limiting
REQUEST_DELAY = 0.05  # seconds between API calls
MAX_RETRIES = 3

# Logging
def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


def fetch_with_retry(url, params=None, max_retries=MAX_RETRIES):
    """Fetch URL with retry logic and rate limiting."""
    for attempt in range(max_retries):
        try:
            time.sleep(REQUEST_DELAY)
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 404:
                return None
            else:
                log(f"  HTTP {resp.status_code} for {url[:80]}... (attempt {attempt+1})")
                time.sleep(2 * (attempt + 1))
        except Exception as e:
            log(f"  Error: {e} (attempt {attempt+1})")
            time.sleep(2 * (attempt + 1))
    return None


# ============================================================
# Step 1: Get all HIV-1 PR PDB entries
# ============================================================
def get_hiv_pr_pdb_ids():
    """Search PDB for HIV-1 protease structures using EC 3.4.23.16."""
    log("Step 1: Searching PDB for HIV-1 PR structures...")

    cache_file = DATA_DIR / 'pdb_ids.json'
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        log(f"  Loaded {len(data)} PDB IDs from cache")
        return data

    # Search with multiple queries to maximize coverage
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

    # Query 1: EC number search
    query_ec = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity.rcsb_ec_lineage.id",
                        "operator": "exact_match",
                        "value": "3.4.23.16"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.structure_determination_methodology",
                        "operator": "exact_match",
                        "value": "experimental"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 1000}}
    }

    resp = requests.post(search_url, json=query_ec, timeout=60)
    pdb_ids = set()

    if resp.status_code == 200:
        result = resp.json()
        for entry in result.get('result_set', []):
            pdb_ids.add(entry['identifier'].upper())
        log(f"  EC search: {len(pdb_ids)} entries")

    # Query 2: Text search for "HIV-1 protease" to catch unannotated entries
    query_text = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {
                        "value": "HIV-1 protease"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.structure_determination_methodology",
                        "operator": "exact_match",
                        "value": "experimental"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 1000}}
    }

    resp = requests.post(search_url, json=query_text, timeout=60)
    if resp.status_code == 200:
        result = resp.json()
        for entry in result.get('result_set', []):
            pdb_ids.add(entry['identifier'].upper())
        log(f"  After text search: {len(pdb_ids)} total entries")

    # Query 3: Also search for "HIV protease" variant spelling
    query_text2 = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {
                        "value": "human immunodeficiency virus protease"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.structure_determination_methodology",
                        "operator": "exact_match",
                        "value": "experimental"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 1000}}
    }

    resp = requests.post(search_url, json=query_text2, timeout=60)
    if resp.status_code == 200:
        result = resp.json()
        for entry in result.get('result_set', []):
            pdb_ids.add(entry['identifier'].upper())
        log(f"  After HIV text search: {len(pdb_ids)} total entries")

    pdb_ids = sorted(pdb_ids)
    log(f"  Final: {len(pdb_ids)} unique HIV-1 PR PDB entries")

    with open(cache_file, 'w') as f:
        json.dump(pdb_ids, f, indent=2)

    return pdb_ids


# ============================================================
# Step 2: Get PDB metadata (resolution, ligands, etc.)
# ============================================================
def get_pdb_metadata(pdb_ids):
    """Fetch metadata for each PDB entry."""
    log(f"Step 2: Fetching metadata for {len(pdb_ids)} PDB entries...")

    cache_file = DATA_DIR / 'pdb_metadata.json'
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        log(f"  Loaded {len(data)} metadata entries from cache")
        return data

    metadata = {}
    for i, pid in enumerate(pdb_ids):
        resp = fetch_with_retry(f"https://data.rcsb.org/rest/v1/core/entry/{pid}")
        if resp:
            data = resp.json()
            entry_info = data.get('rcsb_entry_info', {})
            metadata[pid] = {
                'resolution': entry_info.get('resolution_combined', [None])[0],
                'deposition_date': data.get('rcsb_accession_info', {}).get('deposit_date'),
                'title': data.get('struct', {}).get('title'),
                'method': data.get('exptl', [{}])[0].get('method'),
                'polymer_entities': entry_info.get('polymer_entity_count', 0),
                'nonpolymer_entities': entry_info.get('nonpolymer_entity_count', 0),
                'doi': data.get('rcsb_accession_info', {}).get('doi'),
            }

        if (i + 1) % 100 == 0:
            log(f"  Processed {i+1}/{len(pdb_ids)}... Got {len(metadata)}")

    log(f"  Fetched metadata for {len(metadata)} entries")
    with open(cache_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    return metadata


# ============================================================
# Step 3: Get ligand/nonpolymer entity details
# ============================================================
def get_ligand_data(pdb_ids, metadata):
    """Fetch nonpolymer entity data for PDB entries with ligands."""
    log("Step 3: Fetching ligand data...")

    cache_file = DATA_DIR / 'pdb_ligands.json'
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        log(f"  Loaded {len(data)} entries from cache")
        return data

    entries_with_ligands = [pid for pid, m in metadata.items()
                            if m.get('nonpolymer_entities', 0) > 0]
    log(f"  {len(entries_with_ligands)} entries have nonpolymer entities")

    ligand_data = {}
    for i, pid in enumerate(entries_with_ligands):
        resp = fetch_with_retry(
            f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pid}"
        )
        if resp:
            data = resp.json()
            # Normalize: API might return dict or list
            entities = data if isinstance(data, list) else [data]
            ligands = []
            for ent in entities:
                nonpoly = ent.get('pdbx_entity_nonpoly', {})
                ligands.append({
                    'entity_id': ent.get('entity_id'),
                    'comp_id': nonpoly.get('comp_id'),
                    'name': nonpoly.get('name'),
                    'formula': ent.get('chem_comp', {}).get('formula'),
                    'mw': ent.get('chem_comp', {}).get('formula_weight'),
                })
            ligand_data[pid] = ligands

        if (i + 1) % 100 == 0:
            log(f"  Processed {i+1}/{len(entries_with_ligands)}...")

    log(f"  Fetched ligand data for {len(ligand_data)} entries")
    with open(cache_file, 'w') as f:
        json.dump(ligand_data, f, indent=2)
    return ligand_data


# ============================================================
# Step 4: Query ChEMBL for binding data
# ============================================================
def get_chembl_binding_data():
    """Get all Ki/Kd/IC50 measurements for HIV-1 PR (CHEMBL243)
    and cross-reference with PDB via ligand ChEMBL IDs."""
    log("Step 4: Querying ChEMBL for HIV-1 PR binding data...")

    cache_file = DATA_DIR / 'chembl_binding.json'
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        log(f"  Loaded binding data from cache ({len(data)} measurements)")
        return data

    # Get all activities for CHEMBL243 with Ki, Kd, IC50
    activities = []
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

    for std_type in ['Ki', 'Kd', 'IC50']:
        params = {
            'target_chembl_id': 'CHEMBL243',
            'standard_type': std_type,
            'standard_units': 'nM',
            'limit': 500,
            'offset': 0
        }

        while True:
            resp = fetch_with_retry(base_url, params=params)
            if resp is None:
                break

            data = resp.json()
            batch = data.get('activities', [])
            if not batch:
                break

            for act in batch:
                std_val = act.get('standard_value')
                if std_val is None:
                    continue
                try:
                    std_val = float(std_val)
                except (ValueError, TypeError):
                    continue

                activities.append({
                    'activity_id': act.get('activity_id'),
                    'molecule_chembl_id': act.get('molecule_chembl_id'),
                    'molecule_pref_name': act.get('molecule_pref_name'),
                    'standard_type': std_type,
                    'standard_value': std_val,
                    'standard_units': 'nM',
                    'standard_relation': act.get('standard_relation', '='),
                    'assay_chembl_id': act.get('assay_chembl_id'),
                    'document_chembl_id': act.get('document_chembl_id'),
                    'pdb_id': act.get('pdb_id'),
                })

            params['offset'] += params['limit']
            total = data.get('page_meta', {}).get('total_count', 0)
            if params['offset'] >= total:
                break

        log(f"  {std_type}: {len(activities)} total so far")

    log(f"  Total ChEMBL activities: {len(activities)}")

    with open(cache_file, 'w') as f:
        json.dump(activities, f, indent=2)
    return activities


# ============================================================
# Step 5: Query BindingDB for supplementary data
# ============================================================
def get_bindingdb_data():
    """Query BindingDB for HIV-1 PR binding data with PDB cross-references."""
    log("Step 5: Querying BindingDB...")

    cache_file = DATA_DIR / 'bindingdb_data.json'
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        log(f"  Loaded BindingDB data from cache ({len(data)} entries)")
        return data

    # BindingDB has a direct PDB-to-Ki mapping via their API
    # Target: HIV-1 protease (UniProt: P03367, P04585)
    base_url = "https://bindingdb.org/rest/v1/targets/P03367"

    results = []

    # Get target info
    resp = fetch_with_retry(base_url)
    if resp:
        target_data = resp.json()
        log(f"  BindingDB target found: {target_data.get('name', 'Unknown')[:80]}")

    # Search for ligands with PDB cross-references
    # Use the BindingDB search API
    search_url = "https://bindingdb.org/rest/v1/search"

    # Query by protein name
    for query in ['HIV-1 protease', 'HIV protease', 'human immunodeficiency virus protease']:
        params = {
            'query': query,
            'limit': 500,
            'offset': 0
        }

        while True:
            resp = fetch_with_retry(search_url, params=params)
            if resp is None:
                break

            try:
                data = resp.json()
            except:
                break

            entries = data.get('entries', data if isinstance(data, list) else [])
            if not entries:
                break

            for entry in entries:
                if isinstance(entry, dict):
                    ki_val = entry.get('ki_value') or entry.get('kd_value') or entry.get('ic50_value')
                    if ki_val:
                        try:
                            ki_nm = float(ki_val)
                        except (ValueError, TypeError):
                            continue

                        results.append({
                            'bindingdb_id': entry.get('bindingdb_id') or entry.get('id'),
                            'ligand_name': entry.get('ligand_name') or entry.get('name'),
                            'smiles': entry.get('smiles') or entry.get('canonical_smiles'),
                            'ki_nm': ki_nm,
                            'ki_type': entry.get('ki_type') or entry.get('measurement_type'),
                            'pdb_id': entry.get('pdb_id') or entry.get('pdb_code'),
                            'pubmed_id': entry.get('pubmed_id'),
                            'doi': entry.get('doi'),
                        })

            params['offset'] += params['limit']
            if params['offset'] >= data.get('total', params['offset']):
                break

    log(f"  BindingDB results: {len(results)}")

    with open(cache_file, 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================
# Step 6: Cross-reference PDB structures with binding data
# ============================================================
def cross_reference(metadata, ligand_data, chembl_activities, bindingdb_data):
    """Cross-reference PDB entries with binding data to find complexes with known affinities."""
    log("Step 6: Cross-referencing PDB with binding data...")

    cache_file = DATA_DIR / 'cross_referenced_dataset.json'

    # First, try direct PDB IDs from BindingDB and ChEMBL
    pdb_to_binding = {}  # pdb_id -> [{binding info}]

    # From ChEMBL (some have pdb_id filled)
    for act in chembl_activities:
        pid = act.get('pdb_id')
        if pid:
            pid = pid.upper()
            if pid not in pdb_to_binding:
                pdb_to_binding[pid] = []
            pkb = -999
            if act['standard_value'] > 0:
                import math
                pkb = -math.log10(act['standard_value'] * 1e-9)
            pdb_to_binding[pid].append({
                'source': 'ChEMBL',
                'type': act['standard_type'],
                'value_nm': act['standard_value'],
                'pKd': pkb,
                'chembl_id': act['molecule_chembl_id'],
                'molecule_name': act.get('molecule_pref_name'),
            })

    # From BindingDB
    for entry in bindingdb_data:
        pid = entry.get('pdb_id')
        if pid:
            pid = pid.upper()
            if pid not in pdb_to_binding:
                pdb_to_binding[pid] = []
            import math
            pkb = -math.log10(entry['ki_nm'] * 1e-9) if entry['ki_nm'] > 0 else -999
            pdb_to_binding[pid].append({
                'source': 'BindingDB',
                'type': entry.get('ki_type', 'Ki'),
                'value_nm': entry['ki_nm'],
                'pKd': pkb,
                'molecule_name': entry.get('ligand_name'),
                'smiles': entry.get('smiles'),
                'pubmed_id': entry.get('pubmed_id'),
                'doi': entry.get('doi'),
            })

    log(f"  Found {len(pdb_to_binding)} PDB entries with direct binding data")

    # Build final dataset
    # Priority: X-ray structures, resolution < 3.0Å, with binding data
    dataset = []

    for pid, bindings in pdb_to_binding.items():
        if pid not in metadata:
            continue

        meta = metadata[pid]
        resolution = meta.get('resolution')

        # Skip non-X-ray structures
        if meta.get('method') != 'X-RAY DIFFRACTION':
            continue

        # Skip low-resolution
        if resolution is None or resolution > 3.0:
            continue

        # Get best binding data (prefer Ki, then Kd, then IC50)
        ki_vals = [b for b in bindings if b['type'] == 'Ki']
        kd_vals = [b for b in bindings if b['type'] == 'Kd']
        ic50_vals = [b for b in bindings if b['type'] == 'IC50']

        best = None
        if ki_vals:
            best = ki_vals[0]
        elif kd_vals:
            best = kd_vals[0]
        elif ic50_vals:
            best = ic50_vals[0]
        else:
            best = bindings[0]

        ligands = ligand_data.get(pid, [])
        ligand_name = None
        if ligands:
            # Find the largest ligand (most likely the inhibitor)
            ligands_sorted = sorted(ligands,
                                    key=lambda x: x.get('mw', 0) or 0,
                                    reverse=True)
            if ligands_sorted:
                ligand_name = ligands_sorted[0].get('name') or ligands_sorted[0].get('comp_id')

        dataset.append({
            'pdb_id': pid,
            'resolution': resolution,
            'deposition_date': meta.get('deposition_date'),
            'title': meta.get('title'),
            'binding_type': best['type'],
            'binding_value_nm': best['value_nm'],
            'pKd': best['pKd'],
            'binding_source': best['source'],
            'ligand_name': ligand_name or best.get('molecule_name', 'Unknown'),
            'molecule_name': best.get('molecule_name'),
            'smiles': best.get('smiles'),
            'pubmed_id': best.get('pubmed_id'),
            'doi': meta.get('doi'),
        })

    # Sort by resolution (best first)
    dataset.sort(key=lambda x: (x['resolution'] or 999))

    log(f"  Cross-referenced dataset: {len(dataset)} complexes with binding data")

    # Filter to high-quality entries: resolution <= 2.5Å, Ki or Kd preferred
    high_quality = [d for d in dataset
                    if d['resolution'] and d['resolution'] <= 2.5
                    and d['binding_type'] in ('Ki', 'Kd')]
    log(f"  High-quality (<=2.5Å, Ki/Kd): {len(high_quality)} complexes")

    # Also get Ki-only (most reliable for affinity)
    ki_only = [d for d in high_quality if d['binding_type'] == 'Ki']
    log(f"  Ki-only high quality: {len(ki_only)} complexes")

    # Take top 50 by resolution
    top50 = high_quality[:50]
    log(f"  Selected top {len(top50)} complexes")

    # pKd range stats
    pkd_vals = [d['pKd'] for d in top50 if d['pKd'] > -900]
    if pkd_vals:
        log(f"  pKd range: {min(pkd_vals):.2f} - {max(pkd_vals):.2f}")
        log(f"  Median pKd: {sorted(pkd_vals)[len(pkd_vals)//2]:.2f}")

    result = {
        'full_dataset': dataset,
        'high_quality': high_quality,
        'selected_50': top50,
        'pdb_to_binding': {k: v for k, v in pdb_to_binding.items()},
    }

    with open(cache_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ============================================================
# Step 7: Download PDB files and ligand SDFs
# ============================================================
def download_structures(dataset_entries, pdb_dir, sdf_dir):
    """Download PDB and SDF files for selected complexes."""
    log(f"Step 7: Downloading structures for {len(dataset_entries)} complexes...")

    downloaded = []

    for i, entry in enumerate(dataset_entries):
        pid = entry['pdb_id']

        # Download PDB file
        pdb_path = pdb_dir / f"{pid}.pdb"
        if not pdb_path.exists():
            pdb_url = f"https://files.rcsb.org/download/{pid}.pdb"
            resp = fetch_with_retry(pdb_url)
            if resp:
                with open(pdb_path, 'w') as f:
                    f.write(resp.text)
            else:
                # Try .pdb1.gz compressed format
                gz_url = f"https://files.rcsb.org/download/{pid}.pdb1.gz"
                resp = fetch_with_retry(gz_url)
                if resp:
                    import gzip
                    with open(pdb_path, 'wb') as f:
                        f.write(gzip.decompress(resp.content))

        # Download SDF file for ligand
        # PDB chemical component dictionary
        if pdb_path.exists():
            # Get the ligand ID from the PDB file
            try:
                ligand_ids = set()
                with open(pdb_path) as f:
                    for line in f:
                        if line.startswith('HETATM') or line.startswith('HET'):
                            resn = line[17:20].strip()
                            if resn and resn != 'HOH':
                                ligand_ids.add(resn)

                # Filter to likely inhibitors (not buffer/solvent)
                skip_ids = {'HOH', 'DOD', 'WAT', 'NA', 'CL', 'K', 'CA', 'MG', 'ZN',
                            'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'ACT', 'ACY', 'BU1',
                            'DMS', 'DMF', 'DMSO', 'EOH', 'ETE', 'FMT', 'FOR', 'FLC'}
                real_ligands = [lid for lid in ligand_ids if lid not in skip_ids]

                if real_ligands:
                    # Get the largest ligand's SDF
                    for lid in real_ligands[:3]:  # Try first 3
                        sdf_path = sdf_dir / f"{pid}_{lid}.sdf"
                        if not sdf_path.exists():
                            # RCSB ligand export API
                            sdf_url = f"https://models.rcsb.org/v1/{pid}/ligand?auth_asym_id=&encoding=sdf&filename={pid}_{lid}.sdf"
                            # Also try: https://files.rcsb.org/ligands/download/{lid}_ideal.sdf
                            resp = fetch_with_retry(sdf_url)
                            if resp and len(resp.text) > 100:
                                with open(sdf_path, 'w') as f:
                                    f.write(resp.text)
                            else:
                                # Try the CCD SDF
                                ccd_url = f"https://files.rcsb.org/ligands/view/{lid}_ideal.sdf"
                                resp = fetch_with_retry(ccd_url)
                                if resp and len(resp.text) > 100:
                                    with open(sdf_path, 'w') as f:
                                        f.write(resp.text)
            except Exception as e:
                log(f"  Error processing {pid} ligands: {e}")

        entry['pdb_file'] = str(pdb_path) if pdb_path.exists() else None

        if (i + 1) % 10 == 0:
            downloaded_count = sum(1 for e in dataset_entries[:i+1]
                                   if e.get('pdb_file'))
            log(f"  Processed {i+1}/{len(dataset_entries)}... Downloaded {downloaded_count} PDB files")

    downloaded_count = sum(1 for e in dataset_entries if e.get('pdb_file'))
    log(f"  Total downloaded: {downloaded_count} PDB files")
    return dataset_entries


# ============================================================
# Step 8: Generate dataset documentation
# ============================================================
def generate_documentation(dataset):
    """Generate dataset README and provenance documentation."""
    log("Step 8: Generating dataset documentation...")

    entries = dataset.get('selected_50', [])

    # Build summary table
    lines = []
    lines.append("# HIV-1 Protease Complex Dataset")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Data Sources")
    lines.append("")
    lines.append("- **RCSB PDB**: Protein Data Bank (https://www.rcsb.org/)")
    lines.append("  - Search: EC 3.4.23.16 (HIV-1 retropepsin)")
    lines.append("  - Method: X-ray crystallography")
    lines.append("  - Downloaded: PDB format structure files")
    lines.append("")
    lines.append("- **ChEMBL**: European Bioinformatics Institute (https://www.ebi.ac.uk/chembl/)")
    lines.append("  - Target: CHEMBL243 (HIV-1 protease)")
    lines.append("  - Measurements: Ki, Kd, IC50")
    lines.append("")
    lines.append("- **BindingDB**: Binding Database (https://www.bindingdb.org/)")
    lines.append("  - Target: UniProt P03367 (HIV-1 protease)")
    lines.append("  - Cross-referenced PDB-to-affinity data")
    lines.append("")
    lines.append("## Dataset Statistics")
    lines.append("")
    lines.append(f"- Total complexes with binding data: {len(entries)}")

    resolutions = [e.get('resolution') for e in entries if e.get('resolution')]
    if resolutions:
        lines.append(f"- Resolution range: {min(resolutions):.2f} - {max(resolutions):.2f} Å")
        lines.append(f"- Median resolution: {sorted(resolutions)[len(resolutions)//2]:.2f} Å")

    pkd_vals = [e['pKd'] for e in entries if e.get('pKd', -999) > -900]
    if pkd_vals:
        lines.append(f"- Binding affinity (pKd) range: {min(pkd_vals):.2f} - {max(pkd_vals):.2f}")
        lines.append(f"- Median pKd: {sorted(pkd_vals)[len(pkd_vals)//2]:.2f}")

    binding_types = {}
    for e in entries:
        bt = e.get('binding_type', 'Unknown')
        binding_types[bt] = binding_types.get(bt, 0) + 1
    lines.append(f"- Measurement types: {binding_types}")

    sources = {}
    for e in entries:
        src = e.get('binding_source', 'Unknown')
        sources[src] = sources.get(src, 0) + 1
    lines.append(f"- Sources: {sources}")

    lines.append("")
    lines.append("## Selected Complexes")
    lines.append("")
    lines.append("| # | PDB ID | Resolution (Å) | Ligand | Binding Type | Value (nM) | pKd | Source |")
    lines.append("|---|--------|----------------|--------|--------------|------------|-----|--------|")

    for i, entry in enumerate(entries[:60]):
        lines.append(
            f"| {i+1} | {entry['pdb_id']} | {entry.get('resolution', 'N/A')} | "
            f"{entry.get('ligand_name', 'Unknown')[:40]} | "
            f"{entry.get('binding_type', 'N/A')} | "
            f"{entry.get('binding_value_nm', 'N/A')} | "
            f"{entry.get('pKd', 'N/A'):.2f} | "
            f"{entry.get('binding_source', 'N/A')} |"
        )

    lines.append("")
    lines.append("## File Structure")
    lines.append("")
    lines.append("```")
    lines.append("hiv_pr_real_data/")
    lines.append("├── README.md                  # This file")
    lines.append("├── pdb_ids.json               # All HIV-1 PR PDB IDs")
    lines.append("├── pdb_metadata.json          # Structure metadata")
    lines.append("├── pdb_ligands.json           # Ligand information")
    lines.append("├── chembl_binding.json        # ChEMBL binding data")
    lines.append("├── bindingdb_data.json        # BindingDB data")
    lines.append("├── cross_referenced_dataset.json  # Final cross-referenced dataset")
    lines.append("├── pdb_files/                 # Downloaded PDB structure files")
    lines.append("└── sdf_files/                 # Downloaded ligand SDF files")
    lines.append("```")

    with open(DATA_DIR / 'README.md', 'w') as f:
        f.write('\n'.join(lines))

    log("  Documentation saved to README.md")

    # Also save a clean JSON dataset for use in the pipeline
    clean_dataset = []
    for i, entry in enumerate(entries):
        clean_dataset.append({
            'index': i + 1,
            'pdb_id': entry['pdb_id'],
            'resolution': entry.get('resolution'),
            'title': entry.get('title'),
            'ligand_name': entry.get('ligand_name') or entry.get('molecule_name', 'Unknown'),
            'binding_type': entry.get('binding_type'),
            'binding_value_nm': entry.get('binding_value_nm'),
            'pKd': entry.get('pKd'),
            'binding_source': entry.get('binding_source'),
            'pubmed_id': entry.get('pubmed_id'),
            'doi': entry.get('doi'),
            'deposition_date': str(entry.get('deposition_date', '')),
            'pdb_file': entry.get('pdb_file'),
        })

    with open(DATA_DIR / 'clean_dataset.json', 'w') as f:
        json.dump(clean_dataset, f, indent=2, default=str)

    log(f"  Clean dataset saved: {len(clean_dataset)} entries")
    return clean_dataset


# ============================================================
# Main
# ============================================================
def main():
    log("=" * 60)
    log("HIV-1 PR Real Data Download Pipeline")
    log("=" * 60)

    # Step 1: Get PDB IDs
    pdb_ids = get_hiv_pr_pdb_ids()

    # Step 2: Get metadata
    metadata = get_pdb_metadata(pdb_ids)

    # Step 3: Get ligand data
    ligand_data = get_ligand_data(pdb_ids, metadata)

    # Step 4: Get ChEMBL binding data
    chembl_data = get_chembl_binding_data()

    # Step 5: Get BindingDB data
    bindingdb_data = get_bindingdb_data()

    # Step 6: Cross-reference
    dataset = cross_reference(metadata, ligand_data, chembl_data, bindingdb_data)

    # Step 7: Download structures
    selected = dataset.get('selected_50', dataset.get('high_quality', [])[:50])
    selected = download_structures(selected, PDB_DIR, SDF_DIR)

    # Step 8: Documentation
    clean = generate_documentation(dataset)

    log("=" * 60)
    log(f"PIPELINE COMPLETE - Generated dataset with {len(clean)} complexes")
    log(f"Data directory: {DATA_DIR}")
    log("=" * 60)

    # Print summary for immediate viewing
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    for entry in clean[:10]:
        print(f"  {entry['pdb_id']}: {entry['resolution']}A, "
              f"{entry['ligand_name'][:40]}, "
              f"pKd={entry['pKd']:.2f}")
    if len(clean) > 10:
        print(f"  ... and {len(clean) - 10} more complexes")
    print("=" * 60)


if __name__ == '__main__':
    main()
