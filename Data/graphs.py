"""Construction of the six MMDCG-DTA graphs and the Stage-2 candidate graph."""

from __future__ import annotations

import io
from typing import Optional

import dgl
import numpy as np
import torch
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS

from .featurize import featurize_atom, featurize_substructure


def _finite(values: np.ndarray) -> np.ndarray:
    """Replace parser-generated NaN/Inf values without changing array shape."""
    return np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def safe_read_sdf(sdf_content: str, sample_name: str = "") -> Optional[Chem.Mol]:
    mol = Chem.MolFromMolBlock(sdf_content, removeHs=False)
    if mol is None:
        mol = Chem.MolFromMolBlock(sdf_content, removeHs=False, sanitize=False)
        if mol is None:
            return None
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SANITIZE_ALL
                ^ Chem.SANITIZE_PROPERTIES
                ^ Chem.SANITIZE_CLEANUP,
            )
            Chem.GetSymmSSSR(mol)
        except Exception:
            return None
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass
    return mol


def _empty_graph() -> dgl.DGLGraph:
    return dgl.graph(([], []), num_nodes=0)


def _brics_partition(mol: Chem.Mol) -> tuple[list[Chem.Mol], list[tuple[int, ...]]]:
    """Return BRICS fragments and their exact original-atom memberships.

    ``BRICSDecompose`` returns SMARTS-like fragments containing dummy atoms and
    loses a reliable mapping to the input conformer.  Fragmenting the original
    molecule in place preserves its atom indices, which are the group IDs used
    by cross-hierarchy fusion.
    """
    bond_ids: list[int] = []
    for (begin, end), _labels in BRICS.FindBRICSBonds(mol):
        bond = mol.GetBondBetweenAtoms(int(begin), int(end))
        if bond is not None:
            bond_ids.append(bond.GetIdx())

    fragmented = (
        Chem.FragmentOnBonds(mol, sorted(set(bond_ids)), addDummies=False)
        if bond_ids
        else Chem.Mol(mol)
    )
    memberships = list(Chem.GetMolFrags(fragmented, asMols=False, sanitizeFrags=False))
    fragment_mols = list(Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True))
    paired = sorted(zip(memberships, fragment_mols), key=lambda item: min(item[0]))
    return [frag for _, frag in paired], [tuple(group) for group, _ in paired]


def _residue_key(
    chain_id: str, number: int, insertion_code: str
) -> tuple[str, int, str]:
    return chain_id.strip(), int(number), insertion_code.strip()


def _protein_residues(protein_pdb_content: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", io.StringIO(protein_pdb_content))
    residues = []
    for chain in structure[0]:
        for residue in chain:
            if residue.id[0] == " " and "CA" in residue:
                key = _residue_key(chain.id, residue.id[1], residue.id[2])
                residues.append((key, residue))
    return residues


def build_ligand_atom_graph(
    ligand_sdf_content: str, sample_name: str = ""
) -> dgl.DGLGraph:
    mol = safe_read_sdf(ligand_sdf_content, sample_name)
    if mol is None or mol.GetNumConformers() == 0:
        return _empty_graph()

    _fragments, memberships = _brics_partition(mol)
    group = np.empty(mol.GetNumAtoms(), dtype=np.int64)
    for group_id, atom_ids in enumerate(memberships):
        group[list(atom_ids)] = group_id

    conformer = mol.GetConformer()
    features = _finite(np.asarray([featurize_atom(atom) for atom in mol.GetAtoms()]))
    if np.any(features[:, 3] < 0):
        return _empty_graph()
    positions = _finite(
        np.asarray(
            [
                [
                    conformer.GetAtomPosition(i).x,
                    conformer.GetAtomPosition(i).y,
                    conformer.GetAtomPosition(i).z,
                ]
                for i in range(mol.GetNumAtoms())
            ]
        )
    )

    src: list[int] = []
    dst: list[int] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.extend((i, j))
        dst.extend((j, i))

    graph = dgl.graph((src, dst), num_nodes=mol.GetNumAtoms())
    graph.ndata["h"] = torch.from_numpy(features)
    graph.ndata["pos"] = torch.from_numpy(positions)
    graph.ndata["group"] = torch.from_numpy(group)
    return graph


def build_protein_atom_graph(
    protein_pdb_content: str, sample_name: str = ""
) -> dgl.DGLGraph:
    mol = Chem.MolFromPDBBlock(protein_pdb_content, removeHs=False, sanitize=True)
    if mol is None:
        mol = Chem.MolFromPDBBlock(protein_pdb_content, removeHs=False, sanitize=False)
    if mol is None or mol.GetNumConformers() == 0:
        return _empty_graph()
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol,
            sanitizeOps=(
                Chem.SanitizeFlags.SANITIZE_PROPERTIES
                | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                | Chem.SanitizeFlags.SANITIZE_KEKULIZE
                | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            ),
        )
        AllChem.ComputeGasteigerCharges(mol)
        residues = _protein_residues(protein_pdb_content)
    except Exception:
        return _empty_graph()
    if not residues:
        return _empty_graph()

    residue_to_group = {
        key: group_id for group_id, (key, _residue) in enumerate(residues)
    }
    selected: list[int] = []
    groups: list[int] = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        key = _residue_key(
            info.GetChainId(), info.GetResidueNumber(), info.GetInsertionCode()
        )
        if key in residue_to_group:
            selected.append(atom.GetIdx())
            groups.append(residue_to_group[key])
    if not selected:
        return _empty_graph()

    old_to_new = {old: new for new, old in enumerate(selected)}
    conformer = mol.GetConformer()
    features = _finite(
        np.asarray([featurize_atom(mol.GetAtomWithIdx(i)) for i in selected])
    )
    if np.any(features[:, 3] < 0):
        return _empty_graph()
    positions = _finite(
        np.asarray(
            [
                [
                    conformer.GetAtomPosition(i).x,
                    conformer.GetAtomPosition(i).y,
                    conformer.GetAtomPosition(i).z,
                ]
                for i in selected
            ]
        )
    )
    src: list[int] = []
    dst: list[int] = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin in old_to_new and end in old_to_new:
            i, j = old_to_new[begin], old_to_new[end]
            src.extend((i, j))
            dst.extend((j, i))

    graph = dgl.graph((src, dst), num_nodes=len(selected))
    graph.ndata["h"] = torch.from_numpy(features)
    graph.ndata["pos"] = torch.from_numpy(positions)
    graph.ndata["group"] = torch.tensor(groups, dtype=torch.long)
    return graph


def _build_bipartite_contact_graph(
    ligand_positions: np.ndarray,
    protein_positions: np.ndarray,
    cutoff: float,
) -> dgl.DGLGraph:
    n_ligand = ligand_positions.shape[0]
    n_protein = protein_positions.shape[0]
    distances = np.linalg.norm(
        ligand_positions[:, None, :] - protein_positions[None, :, :], axis=-1
    )
    ligand_ids, protein_ids = np.where(distances <= cutoff)

    src: list[int] = []
    dst: list[int] = []
    edge_distances: list[float] = []
    pair_ids: list[int] = []
    for pair_id, (ligand_id, protein_id) in enumerate(zip(ligand_ids, protein_ids)):
        protein_node = n_ligand + int(protein_id)
        ligand_node = int(ligand_id)
        distance = float(distances[ligand_id, protein_id])
        src.extend((ligand_node, protein_node))
        dst.extend((protein_node, ligand_node))
        edge_distances.extend((distance, distance))
        pair_ids.extend((pair_id, pair_id))

    graph = dgl.graph((src, dst), num_nodes=n_ligand + n_protein)
    graph.ndata["side"] = torch.cat(
        [
            torch.zeros(n_ligand, dtype=torch.long),
            torch.ones(n_protein, dtype=torch.long),
        ]
    )
    graph.edata["dist"] = torch.tensor(edge_distances, dtype=torch.float32).reshape(
        -1, 1
    )
    graph.edata["pair_id"] = torch.tensor(pair_ids, dtype=torch.long)
    return graph


def build_atom_interaction_graph(
    ligand_graph: dgl.DGLGraph,
    protein_graph: dgl.DGLGraph,
    cutoff: float,
    sample_name: str = "",
) -> dgl.DGLGraph:
    if ligand_graph.num_nodes() == 0 or protein_graph.num_nodes() == 0:
        return _empty_graph()
    return _build_bipartite_contact_graph(
        ligand_graph.ndata["pos"].cpu().numpy(),
        protein_graph.ndata["pos"].cpu().numpy(),
        cutoff,
    )


def build_ligand_fragment_graph(
    ligand_sdf_content: str, sample_name: str = ""
) -> dgl.DGLGraph:
    mol = safe_read_sdf(ligand_sdf_content, sample_name)
    if mol is None or mol.GetNumConformers() == 0:
        return _empty_graph()
    fragments, memberships = _brics_partition(mol)
    conformer = mol.GetConformer()
    positions = _finite(
        np.asarray(
            [
                np.mean(
                    [
                        [
                            conformer.GetAtomPosition(i).x,
                            conformer.GetAtomPosition(i).y,
                            conformer.GetAtomPosition(i).z,
                        ]
                        for i in atom_ids
                    ],
                    axis=0,
                )
                for atom_ids in memberships
            ]
        )
    )
    features = _finite(
        np.asarray([featurize_substructure(frag, "ligand") for frag in fragments])
    )

    src = [i for i in range(len(fragments)) for j in range(len(fragments)) if i != j]
    dst = [j for i in range(len(fragments)) for j in range(len(fragments)) if i != j]
    graph = dgl.graph((src, dst), num_nodes=len(fragments))
    graph.ndata["h"] = torch.from_numpy(features)
    graph.ndata["pos"] = torch.from_numpy(positions)
    if src:
        graph.edata["dist"] = torch.linalg.norm(
            graph.ndata["pos"][torch.tensor(src)]
            - graph.ndata["pos"][torch.tensor(dst)],
            dim=-1,
            keepdim=True,
        )
    else:
        graph.edata["dist"] = torch.empty(0, 1)
    return graph


def build_protein_residue_graph(
    protein_pdb_content: str,
    cutoff: float,
    sample_name: str = "",
) -> dgl.DGLGraph:
    try:
        residues = _protein_residues(protein_pdb_content)
    except Exception:
        return _empty_graph()
    if not residues:
        return _empty_graph()

    residue_objects = [residue for _key, residue in residues]
    positions = _finite(
        np.asarray([residue["CA"].get_coord() for residue in residue_objects])
    )
    features = _finite(
        np.asarray(
            [featurize_substructure(residue, "protein") for residue in residue_objects]
        )
    )
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    src, dst = np.where((distances <= cutoff) & (distances > 0.0))
    graph = dgl.graph((src, dst), num_nodes=len(residue_objects))
    graph.ndata["h"] = torch.from_numpy(features)
    graph.ndata["pos"] = torch.from_numpy(positions)
    graph.edata["dist"] = torch.from_numpy(
        distances[src, dst].astype(np.float32)
    ).reshape(-1, 1)
    return graph


def build_substructure_interaction_graph(
    ligand_fragment_graph: dgl.DGLGraph,
    protein_residue_graph: dgl.DGLGraph,
    cutoff: float,
    sample_name: str = "",
) -> dgl.DGLGraph:
    if ligand_fragment_graph.num_nodes() == 0 or protein_residue_graph.num_nodes() == 0:
        return _empty_graph()
    return _build_bipartite_contact_graph(
        ligand_fragment_graph.ndata["pos"].cpu().numpy(),
        protein_residue_graph.ndata["pos"].cpu().numpy(),
        cutoff,
    )
