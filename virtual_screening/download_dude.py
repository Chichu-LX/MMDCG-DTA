#!/usr/bin/env python3
"""Download DUD-E HIV-1 protease benchmark dataset.

DUD-E (Directory of Useful Decoys, Enhanced) is the gold-standard benchmark
for virtual screening method evaluation. HIV-1 protease (hivpr) is one of
102 targets with experimentally validated actives and property-matched decoys.

Reference: Mysinger et al. (2012), J. Med. Chem. 55, 6582-6594.
http://dude.docking.org/
"""

import os, sys, urllib.request, gzip, shutil

TARGET = "hivpr"
DUD_E_BASE = "https://dude.docking.org/targets"
OUT_DIR = os.path.join(os.path.dirname(__file__), "dude_data")


def download_file(url, out_path):
    print(f"  Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)


def download_dude_target(target, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    base = f"{DUD_E_BASE}/{target}"

    files = {
        "actives_final.ism": f"{base}/actives_final.ism",
        "decoys_final.ism": f"{base}/decoys_final.ism",
        "crystal_ligand.mol2": f"{base}/crystal_ligand.mol2",
    }

    for fname, url in files.items():
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path):
            print(f"  {fname} already exists, skipping")
            continue
        try:
            download_file(url, out_path)
        except Exception as e:
            print(f"  Failed to download {fname}: {e}")
            # Try .gz variant
            try:
                gz_url = url + ".gz"
                gz_path = out_path + ".gz"
                download_file(gz_url, gz_path)
                with gzip.open(gz_path, "rb") as f_in:
                    with open(out_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)
            except Exception as e2:
                print(f"  Also failed .gz: {e2}")

    # Download receptor
    receptor_path = os.path.join(out_dir, "receptor.pdb")
    if not os.path.exists(receptor_path):
        # Try multiple formats
        for fmt in ["pdb", "pdbqt"]:
            try:
                url = f"{base}/receptor.{fmt}"
                download_file(url, receptor_path if fmt == "pdb"
                             else os.path.join(out_dir, f"receptor.{fmt}"))
                if fmt == "pdb":
                    break
            except Exception:
                pass


def parse_ism(filepath):
    """Parse DUD-E .ism file (SMILES + compound ID per line)."""
    compounds = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                smiles = parts[0]
                cid = parts[1]
                compounds.append({"smiles": smiles, "id": cid})
    return compounds


def main():
    print("=" * 60)
    print("DUD-E HIV-1 Protease Dataset Download")
    print("=" * 60)

    out_dir = OUT_DIR
    print(f"Target: {TARGET}")
    print(f"Output: {out_dir}")

    download_dude_target(TARGET, out_dir)

    # Parse and report
    for fname in ["actives_final.ism", "decoys_final.ism"]:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            compounds = parse_ism(fpath)
            print(f"{fname}: {len(compounds)} compounds")

    print("Done.")


if __name__ == "__main__":
    main()
