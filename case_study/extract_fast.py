#!/usr/bin/env python3
"""Fast extraction of HIV-1 PR data from PDB files (no BioPython)."""

import os, sys, json, pickle, csv
from pathlib import Path
from rdkit import Chem
import numpy as np

STANDARD_RES = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
NON_STANDARD = {"HOH","DOD","WAT","ACE","NH2","NME"}

def parse_pdb_lines(pdb_path):
    """Parse PDB file into structured arrays (fast text-based)."""
    atom_lines = []
    het_lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_lines.append(line)
            elif line.startswith("HETATM"):
                het_lines.append(line)
    return atom_lines, het_lines

def parse_atom_coords(lines):
    """Extract coordinates and residue info from ATOM/HETATM lines."""
    coords = []
    resnames = []
    chains = []
    resnums = []
    for line in lines:
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            resname = line[17:20].strip()
            chain = line[21:22].strip() or " "
            resnum = int(line[22:26].strip())
            coords.append([x, y, z])
            resnames.append(resname)
            chains.append(chain)
            resnums.append(resnum)
        except (ValueError, IndexError):
            continue
    return np.array(coords), resnames, chains, resnums

def extract_ligand_sdf(pdb_path, pdb_id):
    """Extract HETATM and convert to SDF using RDKit."""
    _, het_lines = parse_pdb_lines(pdb_path)
    
    if not het_lines:
        # Try ATOM records for non-standard residues
        atom_lines, _ = parse_pdb_lines(pdb_path)
        het_lines = [l for l in atom_lines if l[17:20].strip() not in STANDARD_RES]
    
    if not het_lines:
        return None
    
    tmp_pdb = f"/tmp/{pdb_id}_ligand.pdb"
    with open(tmp_pdb, "w") as f:
        for line in het_lines:
            f.write(line)
        f.write("END\n")
    
    mol = Chem.MolFromPDBFile(tmp_pdb, sanitize=True, removeHs=False)
    if mol is None:
        mol = Chem.MolFromPDBFile(tmp_pdb, sanitize=False, removeHs=False)
    
    if mol is not None:
        try:
            mol = Chem.RemoveHs(mol)
        except:
            pass
        sdf = Chem.MolToMolBlock(mol)
        os.unlink(tmp_pdb)
        return sdf
    
    os.unlink(tmp_pdb)
    return "".join(het_lines)

def extract_pocket_pdb(pdb_path, cutoff=8.0):
    """Extract protein atoms within cutoff of ligand using fast numpy."""
    atom_lines, het_lines = parse_pdb_lines(pdb_path)
    
    # Get ligand atoms (HETATM not in standard residues)
    lig_lines = het_lines.copy()
    if not lig_lines:
        lig_lines = [l for l in atom_lines if l[17:20].strip() not in STANDARD_RES]
    
    if not lig_lines:
        return ""
    
    lig_coords, _, _, _ = parse_atom_coords(lig_lines)
    if len(lig_coords) == 0:
        return ""
    
    # Get protein atoms
    prot_lines = [l for l in atom_lines if l[17:20].strip() in STANDARD_RES]
    prot_coords, prot_resnames, prot_chains, prot_resnums = parse_atom_coords(prot_lines)
    
    if len(prot_coords) == 0:
        return ""
    
    # Vectorized distance calculation
    pocket_lines = []
    pocket_residues = set()
    
    for i, (coord, resname, chain, resnum) in enumerate(zip(prot_coords, prot_resnames, prot_chains, prot_resnums)):
        dists = np.sqrt(np.sum((lig_coords - coord) ** 2, axis=1))
        if np.min(dists) <= cutoff:
            pocket_lines.append(prot_lines[i])
            pocket_residues.add((chain, resnum, resname))
    
    return "".join(pocket_lines) if pocket_lines else ""

def extract_protein_pdb(pdb_path):
    """Extract protein ATOM records."""
    atom_lines, _ = parse_pdb_lines(pdb_path)
    prot_lines = [l for l in atom_lines if l[17:20].strip() in STANDARD_RES]
    return "".join(prot_lines)

def main():
    print("=" * 60)
    print("Fast HIV-1 PR Data Extraction - ALL COMPLEXES")
    print("=" * 60)
    
    case_dir = Path("/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study")
    pdb_dir = case_dir / "hiv_pr_real_data" / "pdb_files"
    
    with open(str(case_dir / "hiv_pr_real_data" / "final_dataset.json")) as f:
        all_entries = json.load(f)
    
    print(f"\nProcessing {len(all_entries)} complexes...\n")
    
    raw_data = {}
    metadata = []
    success = 0
    skip = 0
    
    for i, entry in enumerate(all_entries):
        pid = entry["pdb_id"]
        pKd = entry["pKd"]
        pdb_path = pdb_dir / f"{pid}.pdb"
        
        if not pdb_path.exists():
            skip += 1
            continue
        
        try:
            ligand_content = extract_ligand_sdf(str(pdb_path), pid)
            if ligand_content is None:
                skip += 1
                continue
            
            pocket_content = extract_pocket_pdb(str(pdb_path), cutoff=8.0)
            protein_content = extract_protein_pdb(str(pdb_path))
            
            raw_data[pid] = {
                "ligand": ligand_content,
                "pocket": pocket_content,
                "protein": protein_content,
                "label": pKd,
                "source": "ChEMBL/RCSB cross-reference",
            }
            
            metadata.append({
                "pdb_id": pid,
                "resolution": entry.get("resolution"),
                "affinity": pKd,
                "Kd_Ki": f"Ki={10**(-pKd):.2e}M",
                "ligand_name": (entry.get("molecule_name") or entry.get("ligand_name", "unknown"))[:80],
                "source": "RCSB PDB + ChEMBL",
                "chembl_id": entry.get("chembl_id", ""),
                "doi": entry.get("doi", ""),
                "pubmed_id": entry.get("pubmed_id", ""),
            })
            success += 1
            
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(all_entries)}] {pid}: pKd={pKd:.2f} (success={success}, skip={skip})")
        
        except Exception as e:
            skip += 1
            if skip <= 5:
                print(f"  ERR {pid}: {e}")
    
    # Save
    output_pkl = case_dir / "hiv_protease_raw.pkl"
    with open(output_pkl, "wb") as f:
        pickle.dump(raw_data, f)
    print(f"\nSaved {len(raw_data)} complexes to {output_pkl}")
    
    output_csv = case_dir / "hiv_protease_metadata.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pdb_id", "resolution", "affinity", "Kd_Ki", "ligand_name",
            "source", "chembl_id", "doi", "pubmed_id"
        ])
        writer.writeheader()
        for m in sorted(metadata, key=lambda x: x["affinity"]):
            writer.writerow(m)
    print(f"Saved metadata to {output_csv}")
    
    affinities = [d["label"] for d in raw_data.values()]
    if affinities:
        sep = "=" * 60; print("\n" + sep)
        print(f"Summary:")
        print(f"  Success: {success}, Skipped: {skip}")
        print(f"  Affinity range: {min(affinities):.2f} - {max(affinities):.2f} pKd")
        print(f"  Mean affinity: {sum(affinities)/len(affinities):.2f} pKd")
        print(sep)

if __name__ == "__main__":
    main()
