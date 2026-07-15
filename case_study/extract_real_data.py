#!/usr/bin/env python3
"""
Step 1: Extract HIV-1 Protease Complex Data from Downloaded Real PDB Files.

Reads PDB files downloaded from RCSB PDB, extracts ligand/pocket/protein
in the format expected by build_hiv_graphs.py.

Input:
  - hiv_pr_real_data/pdb_files/*.pdb (downloaded from RCSB)
  - hiv_pr_real_data/selected_50_diverse.json (selection config)

Output:
  - hiv_protease_raw.pkl: Dictionary of complex_id -> {ligand, pocket, protein, label}
  - hiv_protease_metadata.csv: Metadata including PDB ID, resolution, affinity, ligand name
"""

import os, sys, json, pickle, csv
from pathlib import Path
from datetime import datetime

from Bio.PDB import PDBParser
from rdkit import Chem
import numpy as np


def extract_ligand_sdf(pdb_path, pdb_id):
    """Extract HETATM records from PDB and convert to SDF using RDKit."""
    het_lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('HETATM'):
                het_lines.append(line)

    if not het_lines:
        # Try ATOM records for non-standard residues
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM') and line[17:20].strip() not in ('ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','HOH','DOD','WAT','ACE','NH2','NME'):
                    het_lines.append(line)

    if not het_lines:
        print(f"    WARNING: No HETATM found in {pdb_path}")
        return None

    # Write temporary PDB with just the ligand
    tmp_pdb = f"/tmp/{pdb_id}_ligand.pdb"
    with open(tmp_pdb, 'w') as f:
        for line in het_lines:
            f.write(line)
        f.write("END\n")

    # Try to read with RDKit
    mol = Chem.MolFromPDBFile(tmp_pdb, sanitize=True, removeHs=False)
    if mol is None:
        mol = Chem.MolFromPDBFile(tmp_pdb, sanitize=False, removeHs=False)

    if mol is not None:
        # Remove explicit hydrogens for cleaner SDF
        try:
            mol = Chem.RemoveHs(mol)
        except:
            pass
        sdf = Chem.MolToMolBlock(mol)
        os.unlink(tmp_pdb)
        return sdf

    # Fallback: return HETATM as PDB string (some parsers handle this)
    os.unlink(tmp_pdb)
    return ''.join(het_lines)


def extract_pocket_pdb(pdb_path, ligand_atoms, cutoff=8.0):
    """Extract protein atoms within cutoff distance of ligand atoms."""
    from Bio.PDB import PDBParser
    import io

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_path)

    # Collect ligand atom coordinates
    lig_coords = []
    for atom in structure.get_atoms():
        resname = atom.get_parent().get_resname().strip()
        if resname not in ('ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','HOH','DOD','WAT','ACE','NH2','NME'):
            lig_coords.append(atom.get_coord())

    if not lig_coords:
        print(f"    WARNING: No ligand atoms found in structure")
        return ""

    lig_coords = np.array(lig_coords)

    # Find protein atoms within cutoff
    pocket_atoms = set()
    for atom in structure.get_atoms():
        resname = atom.get_parent().get_resname().strip()
        if resname in ('ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'):
            coord = atom.get_coord()
            dists = np.linalg.norm(lig_coords - coord, axis=1)
            if np.min(dists) <= cutoff:
                pocket_atoms.add(atom.get_serial_number())

    # Filter original PDB lines to pocket residues
    pocket_residues = set()
    for atom in structure.get_atoms():
        if atom.get_serial_number() in pocket_atoms:
            parent = atom.get_parent()
            resname = parent.get_resname().strip()
            resnum = parent.get_id()[1]
            chain = parent.get_parent().get_id()
            pocket_residues.add((chain, resnum, resname))

    # Extract from original PDB file
    pocket_lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                chain = line[21:22].strip() or ' '
                try:
                    resnum = int(line[22:26].strip())
                except:
                    continue
                resname = line[17:20].strip()
                if (chain, resnum, resname) in pocket_residues:
                    pocket_lines.append(line)

    return ''.join(pocket_lines) if pocket_lines else ""


def extract_protein_pdb(pdb_path):
    """Extract protein ATOM records from PDB file."""
    atom_lines = []
    standard_res = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM'):
                resname = line[17:20].strip()
                if resname in standard_res:
                    atom_lines.append(line)
    return ''.join(atom_lines)


def main():
    print("=" * 60)
    print("Real HIV-1 PR Data Extraction")
    print("=" * 60)

    # Paths
    case_dir = Path('/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study')
    pdb_dir = case_dir / 'hiv_pr_real_data' / 'pdb_files'

    # Load selected 50 complexes
    with open('/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study/hiv_pr_real_data/selected_50_diverse.json') as f:
        selected = json.load(f)

    print(f"\nProcessing {len(selected)} selected complexes...\n")

    raw_data = {}
    metadata = []

    for i, entry in enumerate(selected):
        pid = entry['pdb_id']
        pKd = entry['pKd']
        pdb_path = pdb_dir / f"{pid}.pdb"

        if not pdb_path.exists():
            print(f"  SKIP {pid}: PDB file not found at {pdb_path}")
            continue

        try:
            # Extract ligand
            ligand_content = extract_ligand_sdf(str(pdb_path), pid)
            if ligand_content is None:
                print(f"  SKIP {pid}: No ligand found")
                continue

            # Extract pocket
            pocket_content = extract_pocket_pdb(str(pdb_path), None, cutoff=8.0)

            # Extract full protein
            protein_content = extract_protein_pdb(str(pdb_path))

            raw_data[pid] = {
                'ligand': ligand_content,
                'pocket': pocket_content,
                'protein': protein_content,
                'label': pKd,
                'source': 'ChEMBL/RCSB cross-reference',
            }

            metadata.append({
                'pdb_id': pid,
                'resolution': entry.get('resolution'),
                'affinity': pKd,
                'Kd_Ki': f"Ki={10**(-pKd):.2e}M",
                'ligand_name': entry.get('molecule_name') or entry.get('ligand_name', 'unknown')[:80],
                'source': 'RCSB PDB + ChEMBL',
                'chembl_id': entry.get('chembl_id', ''),
                'doi': entry.get('doi', ''),
                'pubmed_id': entry.get('pubmed_id', ''),
            })

            print(f"  OK  {pid}: pKd={pKd:.2f} "
                  f"({entry.get('molecule_name') or entry.get('ligand_name', '?')[:40]})")

        except Exception as e:
            print(f"  ERR {pid}: {e}")
            import traceback
            traceback.print_exc()

    # Save raw data
    output_pkl = case_dir / "hiv_protease_raw.pkl"
    with open(output_pkl, 'wb') as f:
        pickle.dump(raw_data, f)
    print(f"\nSaved {len(raw_data)} complexes to {output_pkl}")

    # Save metadata CSV
    output_csv = case_dir / "hiv_protease_metadata.csv"
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pdb_id', 'resolution', 'affinity', 'Kd_Ki', 'ligand_name',
            'source', 'chembl_id', 'doi', 'pubmed_id'
        ])
        writer.writeheader()
        for m in sorted(metadata, key=lambda x: x['affinity']):
            writer.writerow(m)
    print(f"Saved metadata to {output_csv}")

    # Summary
    affinities = [d['label'] for d in raw_data.values()]
    print(f"\n{'='*60}")
    print(f"HIV-1 Protease Real Dataset Summary:")
    print(f"  Total complexes: {len(raw_data)}")
    print(f"  Affinity range: {min(affinities):.2f} - {max(affinities):.2f} pKd")
    print(f"  Mean affinity: {sum(affinities)/len(affinities):.2f} pKd")
    low = sum(1 for a in affinities if a < 7)
    med = sum(1 for a in affinities if 7 <= a < 9)
    high = sum(1 for a in affinities if a >= 9)
    print(f"  Low (pKd<7): {low}, Med (7<=pKd<9): {med}, High (pKd>=9): {high}")
    print(f"{'='*60}")

    return raw_data


if __name__ == "__main__":
    main()
