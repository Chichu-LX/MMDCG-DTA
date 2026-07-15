"""
Step 1: Extract HIV-1 Protease Complex Data from PDBbind Dataset.

This script extracts raw protein-ligand complex data for HIV-1 protease
from the PDBbind v2016 refined-set and core-set, organizing them into
a clean case study dataset with known binding affinities.

Output:
  - hiv_protease_raw.pkl: Dictionary of complex_id -> {ligand, pocket, protein, label}
  - hiv_protease_metadata.csv: Metadata including PDB ID, resolution, affinity, ligand name
"""

import os
import pickle
import csv
import sys

# HIV-1 Protease PDB IDs from PDBbind v2016 INDEX with known binding affinities
HIV_PR_COMPLEXES = {
    # PDB_ID: -logKd/Ki (higher = stronger binding)
    '2hb1': 3.80,   # Ki=160uM
    '1a30': 4.30,   # Ki=50uM
    '1h1h': 5.22,   # Ki=6uM
    '1hdq': 5.82,   # Ki=1.5uM
    '1bdq': 6.34,   # Ki=0.46uM
    '1hbv': 6.37,   # Ki=430nM
    '1drv': 6.54,   # Kd=0.29uM
    '1bdr': 6.68,   # Ki=0.21uM
    '1htf': 6.83,   # IC50=148nM
    '1a9m': 6.92,   # Ki=119nM
    '1ent': 6.96,   # Ki=110nM
    '1hii': 7.28,   # Ki=53nM
    '1cpi': 7.41,   # IC50=39nM
    '1gno': 7.70,   # Ki=20nM
    '1ajv': 7.72,   # Ki=19.1nM
    '1ajx': 7.91,   # Ki=12.2nM
    '1b6j': 7.92,   # Ki=12nM
    '1g2k': 7.96,   # Ki=11nM
    '1hih': 8.05,   # Ki=9nM
    '1erb': 8.10,   # Kd=8nM
    '3hiv': 8.12,   # Ki=7.5nM
    '1g35': 8.14,   # Ki=7.3nM
    '1b6l': 8.30,   # Ki=5nM
    '1hwr': 8.33,   # Ki=4.70nM
    '1d4j': 8.36,   # Ki=4.40nM
    '1aaq': 8.40,   # Ki=4.0nM
    '1bwb': 8.42,   # Ki=3.8nM
    '1ec0': 8.49,   # Ki=3.2nM
    '1hos': 8.55,   # Ki=2.8nM
    '1bwa': 8.60,   # Ki=2.5nM
    '1b6k': 8.74,   # Ki=1.8nM
    '1d4i': 8.85,   # Ki=1.40nM
    '1ec1': 8.92,   # Ki=1.20nM
    '1hiv': 9.00,   # Ki<1.0nM
    '1ec3': 9.04,   # Ki=0.92nM
    '1ebw': 9.05,   # Ki=0.90nM
    '1hpv': 9.22,   # Ki=0.60nM
    '1hsh': 9.42,   # Ki=0.38nM
    '1qbs': 9.47,   # Ki=0.34nM
    '1hvr': 9.51,   # Ki=0.31nM
    '1dmp': 9.55,   # Ki=0.28nM
    '1eby': 9.70,   # Ki=0.20nM
    '1nh0': 9.74,   # Ki=0.18nM
    '3nu3': 9.82,   # Ki=0.15nM
    '1hxb': 9.92,   # Ki=0.12nM
    '1bv9': 9.96,   # Ki=0.11nM
    '1d4h': 10.00,  # Ki=0.10nM
    '1ec2': 10.00,  # Ki=0.10nM
    '1bv7': 10.30,  # Ki=0.05nM
    '1kzk': 10.39,  # Ki=41pM
    '1qbr': 10.57,  # Ki=0.027nM
    '1dif': 10.66,  # Ki=22pM
    '1hxw': 10.82,  # Ki=15pM
    '1hvk': 10.96,  # Ki=11pM
    '1d4y': 11.10,  # Ki=0.008nM
    '1hpx': 11.26,  # Ki=5.5pM
    '1pro': 11.30,  # Ki=5pM
    '2hb3': 11.35,  # Ki=4.5pM
}

# Additional metadata
LIGAND_NAMES = {
    '2hb1': '512', '1a30': '3-mer', '1h1h': 'A2P', '1hdq': 'INF',
    '1bdq': 'IM1', '1hbv': 'GAN', '1drv': 'A3D', '1bdr': 'IM1',
    '1htf': 'G26', '1a9m': 'U0E', '1ent': '5-mer', '1hii': 'C20',
    '1cpi': '9-mer', '1gno': 'U0E', '1ajv': 'NMB', '1ajx': 'AH1',
    '1b6j': 'PI1', '1g2k': 'NM1', '1hih': 'C20', '1erb': 'ETR',
    '3hiv': 'TXN', '1g35': 'AHF', '1b6l': 'PI4', '1hwr': '216',
    '1d4j': 'MSC', '1aaq': 'PSI', '1bwb': '146', '1ec0': 'BED',
    '1hos': 'PHP', '1bwa': 'XV6', '1b6k': 'PI5', '1d4i': 'BEG',
    '1ec1': 'BEE', '1hiv': '5-mer', '1ec3': 'MS3', '1ebw': 'BEI',
    '1hpv': '478', '1hsh': 'MK1', '1qbs': 'DMP', '1hvr': 'XK2',
    '1dmp': '450', '1eby': 'BEB', '1nh0': '5-mer', '3nu3': '478',
    '1hxb': '5-mer', '1bv9': 'XV6', '1d4h': 'BEH', '1ec2': 'BEJ',
    '1bv7': 'XV6', '1kzk': 'JE2', '1qbr': '638', '1dif': 'A85',
    '1hxw': 'RIT', '1hvk': 'A79', '1d4y': 'TPV', '1hpx': 'KNI',
    '1pro': 'A88', '2hb3': 'GRL',
}

RESOLUTIONS = {
    '2hb1': 2.00, '1a30': 2.00, '1h1h': 2.00, '1hdq': 2.30,
    '1bdq': 2.50, '1hbv': 2.30, '1drv': 2.20, '1bdr': 2.80,
    '1htf': 2.20, '1a9m': 2.30, '1ent': 1.90, '1hii': 2.30,
    '1cpi': 2.05, '1gno': 2.30, '1ajv': 2.00, '1ajx': 2.00,
    '1b6j': 1.85, '1g2k': 1.95, '1hih': 2.20, '1erb': 1.90,
    '3hiv': 2.14, '1g35': 1.80, '1b6l': 1.75, '1hwr': 1.80,
    '1d4j': 1.81, '1aaq': 2.50, '1bwb': 1.80, '1ec0': 1.79,
    '1hos': 2.30, '1bwa': 1.90, '1b6k': 1.85, '1d4i': 1.81,
    '1ec1': 2.10, '1hiv': 2.00, '1ec3': 1.80, '1ebw': 1.81,
    '1hpv': 1.90, '1hsh': 1.90, '1qbs': 1.80, '1hvr': 1.80,
    '1dmp': 2.00, '1eby': 2.29, '1nh0': 1.03, '3nu3': 1.02,
    '1hxb': 2.30, '1bv9': 2.00, '1d4h': 1.81, '1ec2': 2.00,
    '1bv7': 2.00, '1kzk': 1.09, '1qbr': 1.80, '1dif': 1.70,
    '1hxw': 1.80, '1hvk': 1.80, '1d4y': 1.97, '1hpx': 2.00,
    '1pro': 1.80, '2hb3': 1.35,
}


def extract_hiv_protease_data(pdbbind_base, output_dir="."):
    """
    Extract HIV-1 protease complex data from PDBbind dataset.

    Searches both refined-set and core-set for the target complexes.
    """
    subsets = ["refined-set", "core-set"]
    raw_data = {}
    metadata = []

    for subset in subsets:
        subset_path = os.path.join(pdbbind_base, subset)
        if not os.path.exists(subset_path):
            print(f"Warning: {subset_path} not found")
            continue

        for pdb_id, affinity in HIV_PR_COMPLEXES.items():
            if pdb_id in raw_data:
                continue  # Already found in another subset

            complex_dir = os.path.join(subset_path, pdb_id)
            if not os.path.isdir(complex_dir):
                continue

            # Find the correct file names (may be .sdf or .mol2)
            ligand_sdf = os.path.join(complex_dir, f"{pdb_id}_ligand.sdf")
            ligand_mol2 = os.path.join(complex_dir, f"{pdb_id}_ligand.mol2")
            pocket_pdb = os.path.join(complex_dir, f"{pdb_id}_pocket.pdb")
            protein_pdb = os.path.join(complex_dir, f"{pdb_id}_protein.pdb")

            ligand_file = None
            if os.path.exists(ligand_sdf):
                ligand_file = ligand_sdf
            elif os.path.exists(ligand_mol2):
                ligand_file = ligand_mol2

            if ligand_file is None:
                print(f"  SKIP {pdb_id}: no ligand file found")
                continue
            if not os.path.exists(pocket_pdb):
                print(f"  SKIP {pdb_id}: no pocket file found")
                continue

            try:
                with open(ligand_file, 'r') as f:
                    ligand_content = f.read()
                with open(pocket_pdb, 'r') as f:
                    pocket_content = f.read()

                # Protein file is optional (pocket may be sufficient)
                protein_content = ""
                if os.path.exists(protein_pdb):
                    with open(protein_pdb, 'r') as f:
                        protein_content = f.read()

                raw_data[pdb_id] = {
                    'ligand': ligand_content,
                    'pocket': pocket_content,
                    'protein': protein_content,
                    'label': affinity,
                    'source_subset': subset,
                }

                metadata.append({
                    'pdb_id': pdb_id,
                    'resolution': RESOLUTIONS.get(pdb_id, None),
                    'affinity': affinity,
                    'Kd_Ki': f"Ki={10**(-affinity):.2e}M" if affinity > 0 else "N/A",
                    'ligand_name': LIGAND_NAMES.get(pdb_id, 'unknown'),
                    'source': subset,
                })

                print(f"  OK  {pdb_id}: pKd={affinity:.2f} "
                      f"({LIGAND_NAMES.get(pdb_id, '?')}) "
                      f"[{subset}]")

            except Exception as e:
                print(f"  ERR {pdb_id}: {e}")

    # Save data
    output_pkl = os.path.join(output_dir, "hiv_protease_raw.pkl")
    with open(output_pkl, 'wb') as f:
        pickle.dump(raw_data, f)
    print(f"\nSaved {len(raw_data)} complexes to {output_pkl}")

    # Save metadata CSV
    output_csv = os.path.join(output_dir, "hiv_protease_metadata.csv")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pdb_id', 'resolution', 'affinity', 'Kd_Ki', 'ligand_name', 'source'
        ])
        writer.writeheader()
        for m in sorted(metadata, key=lambda x: x['affinity']):
            writer.writerow(m)
    print(f"Saved metadata to {output_csv}")

    # Summary statistics
    affinities = [d['label'] for d in raw_data.values()]
    print(f"\n{'='*60}")
    print(f"HIV-1 Protease Dataset Summary:")
    print(f"  Total complexes: {len(raw_data)}")
    print(f"  Affinity range: {min(affinities):.2f} - {max(affinities):.2f} pKd/Ki")
    print(f"  Mean affinity: {sum(affinities)/len(affinities):.2f} pKd/Ki")
    print(f"  Low binders (pKd<7): {sum(1 for a in affinities if a<7)}")
    print(f"  Medium binders (7<=pKd<9): {sum(1 for a in affinities if 7<=a<9)}")
    print(f"  High binders (pKd>=9): {sum(1 for a in affinities if a>=9)}")
    print(f"{'='*60}")

    return raw_data


if __name__ == "__main__":
    # Default path on the server
    pdbbind_base = sys.argv[1] if len(sys.argv) > 1 else \
        "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/Data/PDBbind_dataset"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    print(f"Extracting HIV-1 Protease data from: {pdbbind_base}")
    print(f"Output directory: {output_dir}")
    print()
    extract_hiv_protease_data(pdbbind_base, output_dir)
