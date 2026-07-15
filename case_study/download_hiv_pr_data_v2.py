#!/usr/bin/env python3
"""
Download real HIV-1 Protease data with robust cross-referencing.
Strategy:
1. Get all HIV-1 PR PDB IDs via RCSB search
2. Get ligand info for each via PDBe API (working, fast)
3. Download SDF files for unique ligands from RCSB
4. Generate InChIKeys with RDKit
5. Search ChEMBL by InChIKey for molecule matches
6. Get Ki/Kd activities for matched molecules
7. Download PDB files for matched complexes
"""

import json, os, sys, time, hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import requests
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study/hiv_pr_real_data')
DATA_DIR.mkdir(parents=True, exist_ok=True)
PDB_DIR = DATA_DIR / 'pdb_files'
SDF_DIR = DATA_DIR / 'sdf_files'
PDB_DIR.mkdir(exist_ok=True)
SDF_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 50
DELAY = 0.05

NON_DRUG = {
    'HOH', 'DOD', 'WAT', 'NA', 'CL', 'K', 'CA', 'MG', 'ZN', 'MN', 'FE',
    'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'PGE', 'ACT', 'ACY', 'BU1',
    'DMS', 'DMF', 'DMSO', 'EOH', 'ETE', 'FMT', 'FOR', 'FLC', 'CIT',
    'BME', 'BCT', 'MRD', 'MPD', 'PG4', 'PGO', 'TRS', 'TRS', 'NO3',
    'AZI', 'IOD', 'BR', 'CO3', 'BO3', 'SCN', 'CMO', 'CO2', 'XE',
    'HEM', 'FAD', 'NAP', 'NDP', 'FMN', 'ADP', 'ATP', 'GDP', 'GTP',
    'PLM', 'LMT', 'OLC', 'OLA', 'LDA', 'SDS', 'OCT', 'DMU', 'UNX',
}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(url, params=None, retries=3):
    for i in range(retries):
        try:
            time.sleep(DELAY)
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            time.sleep(2 * (i + 1))
        except Exception as e:
            time.sleep(2 * (i + 1))
    return None


# ================================================================
# Step 1: Get PDB IDs
# ================================================================
def get_pdb_ids():
    cache = DATA_DIR / 'pdb_ids.json'
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    queries = [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_polymer_entity.rcsb_ec_lineage.id",
            "operator": "exact_match", "value": "3.4.23.16"}},
        {"type": "terminal", "service": "full_text", "parameters": {
            "value": "HIV-1 protease"}},
    ]

    all_ids = set()
    for q in queries:
        payload = {
            "query": {
                "type": "group", "logical_operator": "and", "nodes": [
                    q,
                    {"type": "terminal", "service": "text", "parameters": {
                        "attribute": "rcsb_entry_info.structure_determination_methodology",
                        "operator": "exact_match", "value": "experimental"}}
                ]
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 2000}}
        }
        resp = requests.post(search_url, json=payload, timeout=60)
        if resp.status_code == 200:
            for entry in resp.json().get('result_set', []):
                all_ids.add(entry['identifier'].upper())

    ids = sorted(all_ids)
    log(f"Step 1: Found {len(ids)} HIV-1 PR PDB entries")
    with open(cache, 'w') as f:
        json.dump(ids, f, indent=2)
    return ids


# ================================================================
# Step 2: Get ligand info via PDBe API
# ================================================================
def get_ligand_info(pdb_ids):
    cache = DATA_DIR / 'pdb_ligand_info.json'
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    log(f"Step 2: Getting ligand info for {len(pdb_ids)} entries via PDBe...")
    ligand_info = {}

    for i in range(0, len(pdb_ids), BATCH_SIZE):
        batch = pdb_ids[i:i+BATCH_SIZE]
        for pid in batch:
            try:
                # Use PDBe API (works reliably)
                url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/ligand_monomers/{pid}"
                resp = fetch(url)
                if resp:
                    data = resp.json()
                    monomers = data.get(pid.lower(), [])
                    drug_ligands = []
                    for m in monomers:
                        comp_id = m.get('chem_comp_id', '')
                        if comp_id and comp_id not in NON_DRUG:
                            drug_ligands.append({
                                'comp_id': comp_id,
                                'name': m.get('chem_comp_name', ''),
                                'weight': m.get('weight'),
                                'chain_id': m.get('chain_id'),
                            })
                    if drug_ligands:
                        # Deduplicate by comp_id
                        seen = set()
                        unique = []
                        for d in drug_ligands:
                            if d['comp_id'] not in seen:
                                seen.add(d['comp_id'])
                                unique.append(d)
                        ligand_info[pid] = unique
            except Exception:
                pass

        if (i + BATCH_SIZE) % 100 == 0:
            log(f"  Processed {min(i+BATCH_SIZE, len(pdb_ids))}/{len(pdb_ids)}... "
                f"Found ligands in {len(ligand_info)} entries")

    log(f"  Got ligand info for {len(ligand_info)} entries")
    with open(cache, 'w') as f:
        json.dump(ligand_info, f, indent=2)
    return ligand_info


# ================================================================
# Step 3: Download SDF and generate InChIKeys
# ================================================================
def get_ligand_inchikeys(ligand_info):
    """Download SDF files for unique ligand 3-letter codes, generate InChIKeys."""
    cache = DATA_DIR / 'ligand_inchikeys.json'
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    from rdkit import Chem

    # Collect unique comp_ids
    comp_ids = set()
    for pid, ligands in ligand_info.items():
        for lig in ligands:
            comp_ids.add(lig['comp_id'])

    log(f"Step 3: Getting InChIKeys for {len(comp_ids)} unique ligand types...")

    inchikeys = {}
    for i, comp_id in enumerate(sorted(comp_ids)):
        sdf_path = SDF_DIR / f"{comp_id}.sdf"

        # Download SDF if needed
        if not sdf_path.exists():
            url = f"https://files.rcsb.org/ligands/view/{comp_id}_ideal.sdf"
            resp = fetch(url)
            if resp and len(resp.text) > 100:
                with open(sdf_path, 'w') as f:
                    f.write(resp.text)

        # Generate InChIKey
        if sdf_path.exists():
            try:
                suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
                mol = next(suppl, None)
                if mol is not None:
                    Chem.SanitizeMol(mol)
                    inchikey = Chem.inchi.MolToInchiKey(mol)
                    inchikeys[comp_id] = inchikey
            except Exception:
                pass

        if (i + 1) % 50 == 0:
            log(f"  Processed {i+1}/{len(comp_ids)}... Got {len(inchikeys)} InChIKeys")

    log(f"  Generated {len(inchikeys)} InChIKeys from {len(comp_ids)} ligands")
    with open(cache, 'w') as f:
        json.dump(inchikeys, f, indent=2)
    return inchikeys


# ================================================================
# Step 4: Search ChEMBL by InChIKey and get binding data
# ================================================================
def get_chembl_by_inchikey(inchikeys):
    """For each unique InChIKey, search ChEMBL for the molecule and its activities."""
    cache = DATA_DIR / 'chembl_inchikey_matches.json'
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    log(f"Step 4: Searching ChEMBL for {len(inchikeys)} InChIKeys...")

    # First, build InChIKey → ChEMBL molecule mapping
    inchikey_to_chembl = {}
    unique_inchikeys = set(inchikeys.values())

    for i, ik in enumerate(sorted(unique_inchikeys)):
        resp = fetch(
            "https://www.ebi.ac.uk/chembl/api/data/molecule.json",
            params={"molecule_structures__standard_inchi_key": ik, "limit": 5}
        )
        if resp:
            data = resp.json()
            molecules = data.get('molecules', [])
            if molecules:
                mol = molecules[0]
                inchikey_to_chembl[ik] = {
                    'chembl_id': mol.get('molecule_chembl_id'),
                    'pref_name': mol.get('pref_name'),
                    'smiles': mol.get('molecule_structures', {}).get('canonical_smiles'),
                }

        if (i + 1) % 50 == 0:
            log(f"  Processed {i+1}/{len(unique_inchikeys)}... "
                f"Matched {len(inchikey_to_chembl)} to ChEMBL")

    log(f"  Matched {len(inchikey_to_chembl)}/{len(unique_inchikeys)} InChIKeys to ChEMBL")

    # Now get Ki/Kd activities for matched molecules (target CHEMBL243 = HIV-1 PR)
    chembl_to_affinity = {}
    chembl_ids = list(set(m['chembl_id'] for m in inchikey_to_chembl.values()))

    for i, chembl_id in enumerate(chembl_ids):
        for std_type in ['Ki', 'Kd']:
            resp = fetch(
                "https://www.ebi.ac.uk/chembl/api/data/activity.json",
                params={
                    'molecule_chembl_id': chembl_id,
                    'target_chembl_id': 'CHEMBL243',
                    'standard_type': std_type,
                    'standard_units': 'nM',
                    'limit': 10,
                }
            )
            if resp:
                data = resp.json()
                activities = data.get('activities', [])
                if activities:
                    if chembl_id not in chembl_to_affinity:
                        chembl_to_affinity[chembl_id] = []
                    for act in activities:
                        val = act.get('standard_value')
                        if val is not None:
                            try:
                                val = float(val)
                            except (ValueError, TypeError):
                                continue
                            chembl_to_affinity[chembl_id].append({
                                'type': std_type,
                                'value_nm': val,
                                'relation': act.get('standard_relation', '='),
                                'assay_id': act.get('assay_chembl_id'),
                                'doc_id': act.get('document_chembl_id'),
                            })
            time.sleep(DELAY / 2)  # Shorter delay for activity lookups

        if (i + 1) % 100 == 0:
            log(f"  Got binding for {len(chembl_to_affinity)}/{i+1} molecules...")

    log(f"  Found binding data for {len(chembl_to_affinity)}/{len(chembl_ids)} molecules")

    result = {
        'inchikey_to_chembl': inchikey_to_chembl,
        'chembl_to_affinity': chembl_to_affinity,
    }
    with open(cache, 'w') as f:
        json.dump(result, f, indent=2)
    return result


# ================================================================
# Step 5: Build final cross-referenced dataset
# ================================================================
def build_final_dataset(pdb_ids, ligand_info, inchikeys, chembl_data):
    """Cross-reference PDB entries with binding data via InChIKey."""
    log("Step 5: Building final cross-referenced dataset...")

    inchikey_to_chembl = chembl_data['inchikey_to_chembl']
    chembl_to_affinity = chembl_data['chembl_to_affinity']

    dataset = []

    for pid in pdb_ids:
        ligands = ligand_info.get(pid, [])
        if not ligands:
            continue

        for lig in ligands:
            comp_id = lig['comp_id']
            ik = inchikeys.get(comp_id)
            if not ik:
                continue

            mol_match = inchikey_to_chembl.get(ik)
            if not mol_match:
                continue

            chembl_id = mol_match['chembl_id']
            activities = chembl_to_affinity.get(chembl_id, [])

            # Prefer Ki, then Kd
            ki_vals = [a for a in activities if a['type'] == 'Ki']
            kd_vals = [a for a in activities if a['type'] == 'Kd']

            best = None
            if ki_vals:
                best = sorted(ki_vals, key=lambda x: x['value_nm'])[0]
            elif kd_vals:
                best = sorted(kd_vals, key=lambda x: x['value_nm'])[0]
            elif activities:
                best = activities[0]

            if best:
                import math
                pkb = -math.log10(max(best['value_nm'] * 1e-9, 1e-16))

                dataset.append({
                    'pdb_id': pid,
                    'ligand_comp_id': comp_id,
                    'ligand_name': lig['name'],
                    'ligand_weight': lig['weight'],
                    'chembl_id': chembl_id,
                    'molecule_name': mol_match.get('pref_name'),
                    'smiles': mol_match.get('smiles'),
                    'binding_type': best['type'],
                    'binding_value_nm': best['value_nm'],
                    'pKd': pkb,
                    'source': 'ChEMBL (InChIKey match)',
                    'chembl_assay_id': best.get('assay_id'),
                    'chembl_doc_id': best.get('doc_id'),
                })

    # Remove duplicates (keep best resolution per PDB ID)
    # For each PDB ID, keep the ligand with the best (lowest) Ki/Kd
    deduped = {}
    for entry in dataset:
        pid = entry['pdb_id']
        if pid not in deduped or entry['binding_value_nm'] < deduped[pid]['binding_value_nm']:
            deduped[pid] = entry

    result = sorted(deduped.values(), key=lambda x: x['binding_value_nm'])

    log(f"  Final dataset: {len(result)} PDB complexes with binding data")

    # Stats
    pkd_vals = [d['pKd'] for d in result]
    if pkd_vals:
        log(f"  pKd range: {min(pkd_vals):.2f} - {max(pkd_vals):.2f}")
        log(f"  Median pKd: {sorted(pkd_vals)[len(pkd_vals)//2]:.2f}")

    types = defaultdict(int)
    for d in result:
        types[d['binding_type']] += 1
    log(f"  Measurement types: {dict(types)}")

    cache = DATA_DIR / 'final_dataset.json'
    with open(cache, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ================================================================
# Step 6: Download PDB files and add resolution metadata
# ================================================================
def download_and_enrich(dataset):
    """Download PDB files and fetch resolution metadata."""
    log(f"Step 6: Downloading PDB files for {len(dataset)} complexes...")

    # Fetch metadata for dataset entries
    for i, entry in enumerate(dataset):
        pid = entry['pdb_id']

        # Get metadata
        resp = fetch(f"https://data.rcsb.org/rest/v1/core/entry/{pid}")
        if resp:
            data = resp.json()
            entry['resolution'] = data.get('rcsb_entry_info', {}).get(
                'resolution_combined', [None])[0]
            entry['deposition_date'] = data.get('rcsb_accession_info', {}).get('deposit_date')
            entry['title'] = data.get('struct', {}).get('title')
            citations = data.get('citation', [])
            for cit in citations:
                if cit.get('rcsb_is_primary') == 'Y':
                    entry['pubmed_id'] = cit.get('pdbx_database_id_PubMed')
                    entry['doi'] = cit.get('pdbx_database_id_DOI')
                    break

        # Download PDB file
        pdb_path = PDB_DIR / f"{pid}.pdb"
        if not pdb_path.exists():
            pdb_resp = fetch(f"https://files.rcsb.org/download/{pid}.pdb")
            if pdb_resp:
                with open(pdb_path, 'w') as f:
                    f.write(pdb_resp.text)

        entry['pdb_file'] = str(pdb_path) if pdb_path.exists() else None

        if (i + 1) % 50 == 0:
            downloaded = sum(1 for d in dataset[:i+1] if d.get('pdb_file'))
            log(f"  {i+1}/{len(dataset)} processed, {downloaded} PDB files downloaded")

    downloaded = sum(1 for d in dataset if d.get('pdb_file'))
    log(f"  Downloaded {downloaded} PDB files")

    # Filter to high quality only
    high_quality = [d for d in dataset
                    if d.get('resolution') and d['resolution'] <= 2.5
                    and d['binding_type'] in ('Ki', 'Kd')]
    log(f"  High quality (<=2.5Å, Ki/Kd): {len(high_quality)} complexes")

    top50 = high_quality[:50]
    with open(DATA_DIR / 'clean_dataset.json', 'w') as f:
        json.dump(top50, f, indent=2, default=str)

    return top50


# ================================================================
# Step 7: Generate documentation
# ================================================================
def generate_docs(dataset):
    log("Step 7: Generating documentation...")

    lines = [
        "# HIV-1 Protease Complex Dataset",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Data Sources",
        "\n- **RCSB PDB** (https://www.rcsb.org/): Structure files (.pdb format)",
        "  - EC 3.4.23.16 (HIV-1 retropepsin), X-ray crystallography",
        "  - Ligand SDF from RCSB Chemical Component Dictionary",
        "\n- **PDBe** (https://www.ebi.ac.uk/pdbe/): Ligand monomer annotations",
        "\n- **ChEMBL** (https://www.ebi.ac.uk/chembl/): Binding affinity data",
        "  - Target: CHEMBL243 (HIV-1 protease)",
        "  - Cross-referenced via InChIKey generated from PDB ligand SDF files",
        "\n## Cross-Referencing Method",
        "\n1. Downloaded SDF files for all unique PDB ligand 3-letter codes",
        "2. Generated InChIKeys using RDKit",
        "3. Searched ChEMBL molecule database by InChIKey",
        "4. Filtered activities for target CHEMBL243 with Ki/Kd in nM",
        "5. Matched back to PDB entries via ligand comp_id",
        "\n## Dataset Statistics",
        f"\n- Total complexes with binding data: {len(dataset)}",
    ]

    resolutions = [d.get('resolution') for d in dataset if d.get('resolution')]
    if resolutions:
        lines.append(f"- Resolution range: {min(resolutions):.2f} - {max(resolutions):.2f} Å")
        lines.append(f"- Median resolution: {sorted(resolutions)[len(resolutions)//2]:.2f} Å")

    pkd = [d['pKd'] for d in dataset]
    if pkd:
        lines.append(f"- pKd range: {min(pkd):.2f} - {max(pkd):.2f}")
        lines.append(f"- Median pKd: {sorted(pkd)[len(pkd)//2]:.2f}")

    lines.append("\n## Top 50 Complexes\n")
    lines.append("| # | PDB ID | Resolution | Ligand | pKd | Type | ChEMBL ID |")
    lines.append("|---|--------|-----------|--------|-----|------|-----------|")

    for i, d in enumerate(dataset[:50]):
        lines.append(
            f"| {i+1} | {d['pdb_id']} | {d.get('resolution', 'N/A')} | "
            f"{d.get('ligand_name', d.get('molecule_name', 'Unknown'))[:40]} | "
            f"{d['pKd']:.2f} | {d['binding_type']} | {d.get('chembl_id', 'N/A')} |"
        )

    with open(DATA_DIR / 'README.md', 'w') as f:
        f.write('\n'.join(lines))

    log("  Documentation saved to README.md")


# ================================================================
# Main
# ================================================================
def main():
    log("=" * 60)
    log("HIV-1 PR Real Data Pipeline v2")
    log("=" * 60)

    # Step 1: PDB IDs
    pdb_ids = get_pdb_ids()

    # Step 2: Ligand info via PDBe
    ligand_info = get_ligand_info(pdb_ids)

    # Step 3: SDF → InChIKeys
    inchikeys = get_ligand_inchikeys(ligand_info)

    # Step 4: ChEMBL by InChIKey
    chembl_data = get_chembl_by_inchikey(inchikeys)

    # Step 5: Build dataset
    dataset = build_final_dataset(pdb_ids, ligand_info, inchikeys, chembl_data)

    # Step 6: Download and enrich
    final = download_and_enrich(dataset)

    # Step 7: Docs
    generate_docs(final)

    log("=" * 60)
    log(f"COMPLETE: {len(final)} high-quality complexes in final dataset")
    log(f"Data: {DATA_DIR}")
    log("=" * 60)

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    for i, d in enumerate(final[:20]):
        print(f"  {i+1:2d}. {d['pdb_id']}: {d.get('resolution', 'N/A')}Å | "
              f"{d.get('ligand_name', 'Unknown')[:45]} | "
              f"pKd={d['pKd']:.2f} ({d['binding_type']})")
    if len(final) > 20:
        print(f"  ... and {len(final) - 20} more")
    print("=" * 60)


if __name__ == '__main__':
    main()
