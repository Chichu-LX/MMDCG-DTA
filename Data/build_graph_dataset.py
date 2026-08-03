"""Build paper-aligned graph caches from the parsed PDBbind dictionaries."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import yaml

from .graphs import (
    build_atom_interaction_graph,
    build_ligand_atom_graph,
    build_ligand_fragment_graph,
    build_protein_atom_graph,
    build_protein_residue_graph,
    build_substructure_interaction_graph,
)


def load_affinity_labels(index_file: Path) -> dict[str, float]:
    labels: dict[str, float] = {}
    if not index_file.exists():
        return labels
    for line in index_file.read_text().splitlines()[6:]:
        fields = line.split()
        if len(fields) >= 4:
            try:
                labels[fields[0]] = float(fields[3])
            except ValueError:
                continue
    return labels


def _clean_structure(content: str) -> str:
    return "\n".join(
        line
        for line in content.splitlines()
        if "nan" not in line.lower() and "inf" not in line.lower()
    )


def process_single_complex(compound_id, raw, labels, config):
    ligand = _clean_structure(raw["ligand"])
    pocket = _clean_structure(raw["pocket"])
    ligand_atom = build_ligand_atom_graph(ligand, compound_id)
    protein_atom = build_protein_atom_graph(pocket, compound_id)
    ligand_fragment = build_ligand_fragment_graph(ligand, compound_id)
    protein_residue = build_protein_residue_graph(pocket, config["d_res"], compound_id)
    graphs = (ligand_atom, protein_atom, ligand_fragment, protein_residue)
    if any(graph.num_nodes() == 0 for graph in graphs):
        raise ValueError("one or more intra-molecular graphs are empty")

    if ligand_atom.ndata["group"].max().item() >= ligand_fragment.num_nodes():
        raise ValueError("ligand atom-to-fragment mapping is out of range")
    if protein_atom.ndata["group"].max().item() >= protein_residue.num_nodes():
        raise ValueError("protein atom-to-residue mapping is out of range")

    atom_initial = build_atom_interaction_graph(
        ligand_atom, protein_atom, config["d_atom_initial"], compound_id
    )
    atom_candidate = build_atom_interaction_graph(
        ligand_atom, protein_atom, config["d_atom_candidate"], compound_id
    )
    substructure = build_substructure_interaction_graph(
        ligand_fragment, protein_residue, config["d_sub"], compound_id
    )
    return {
        "compound_id": compound_id,
        "ligand_atom_graph": ligand_atom,
        "protein_atom_graph": protein_atom,
        "atom_interaction_graph": atom_initial,
        "atom_candidate_graph": atom_candidate,
        "ligand_fragment_graph": ligand_fragment,
        "protein_residue_graph": protein_residue,
        "substructure_interaction_graph": substructure,
        "label": labels.get(compound_id),
    }


def build_sample_graphs(dataset, labels, config):
    samples = {}
    failed = 0
    for index, (compound_id, raw) in enumerate(dataset.items(), start=1):
        try:
            sample = process_single_complex(compound_id, raw, labels, config)
            if sample["label"] is not None:
                samples[compound_id] = sample
        except Exception as exc:
            failed += 1
            print(f"[{compound_id}] skipped: {exc}")
        if index % 100 == 0:
            print(f"Processed {index}/{len(dataset)} complexes")
    print(f"Built {len(samples)} labeled samples; skipped {failed}")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("Data"))
    parser.add_argument("--output-dir", type=Path, default=Path("Data"))
    parser.add_argument("--index-file", type=Path, default=Path("Data/INDEX_data.2016"))
    parser.add_argument("--config", type=Path, default=Path("default.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    labels = load_affinity_labels(args.index_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("core_set", "refined_set"):
        source = args.input_dir / f"{split}.pkl"
        if not source.exists():
            raise FileNotFoundError(
                f"Missing {source}; run `python -m Data.loader` first"
            )
        with source.open("rb") as handle:
            raw_dataset = pickle.load(handle)
        samples = build_sample_graphs(raw_dataset, labels, config)
        destination = args.output_dir / f"{split}_graphs.pkl"
        with destination.open("wb") as handle:
            pickle.dump(samples, handle)
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
