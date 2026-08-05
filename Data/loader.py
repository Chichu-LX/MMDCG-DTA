"""Parse PDBbind Core/Refined directories into raw structure dictionaries."""

import argparse
import pickle
from pathlib import Path


def load_complex_data(dataset_path):
    data = {}
    for complex_path in sorted(dataset_path.iterdir()):
        if not complex_path.is_dir():
            continue
        compound_id = complex_path.name
        paths = {
            "pocket": complex_path / f"{compound_id}_pocket.pdb",
            "ligand": complex_path / f"{compound_id}_ligand.sdf",
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            print(f"[{compound_id}] skipped; missing {missing}")
            continue
        data[compound_id] = {
            key: path.read_text(errors="replace") for key, path in paths.items()
        }
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("Data/PDBbind_dataset")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("Data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for directory_name, output_name in (
        ("core-set", "core_set.pkl"),
        ("refined-set", "refined_set.pkl"),
    ):
        source = args.dataset_dir / directory_name
        if not source.exists():
            raise FileNotFoundError(source)
        data = load_complex_data(source)
        destination = args.output_dir / output_name
        with destination.open("wb") as handle:
            pickle.dump(data, handle)
        print(f"Saved {len(data)} complexes to {destination}")


if __name__ == "__main__":
    main()
