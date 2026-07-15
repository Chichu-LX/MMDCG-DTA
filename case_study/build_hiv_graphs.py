"""
Step 2: Build Graph Dataset for HIV-1 Protease Case Study.

Converts raw protein-ligand complex data into MMDCG-DTA-compatible DGL graphs:
  1. Ligand Atom Graph (intra-molecular bonds)
  2. Protein Atom Graph (intra-molecular bonds)
  3. Atom Interaction Graph (inter-molecular spatial contacts)
  4. Ligand Fragment Graph (BRICS decomposition)
  5. Protein Residue Graph (Ca-distance contacts)
  6. Substructure Interaction Graph (fragment-residue spatial contacts)

Plus, K-Means-based group assignment for atom-to-substructure mapping
(matching the patch_add_group_ids logic from train_stage1.py).
"""

import os
import sys
import pickle
import numpy as np
import torch
import dgl
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Descriptors
from Bio.PDB import PDBParser
import io
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Feature Extraction (mirrors Data/featurize.py)
# ============================================================================

def featurize_atom(atom):
    """5-dim atom features: atomic_num, formal_charge, aromatic, hybridization, gasteiger_charge"""
    atomic_num = atom.GetAtomicNum()
    formal_charge = atom.GetFormalCharge()
    aromatic = 1 if atom.GetIsAromatic() else 0
    hyb = atom.GetHybridization()
    if hyb == Chem.rdchem.HybridizationType.SP:
        hyb_val = 0
    elif hyb == Chem.rdchem.HybridizationType.SP2:
        hyb_val = 1
    elif hyb == Chem.rdchem.HybridizationType.SP3:
        hyb_val = 2
    elif hyb == Chem.rdchem.HybridizationType.SP3D:
        hyb_val = 3
    elif hyb == Chem.rdchem.HybridizationType.SP3D2:
        hyb_val = 4
    else:
        hyb_val = -1
    try:
        gasteiger = float(atom.GetProp('_GasteigerCharge'))
    except:
        gasteiger = 0.0
    return np.array([atomic_num, formal_charge, aromatic, hyb_val, gasteiger], dtype=np.float32)


def featurize_substructure(substructure, node_type="ligand"):
    """Substructure-level features: 5-dim for ligand, 1-dim for protein residue."""
    if node_type == "ligand":
        mw = Descriptors.MolWt(substructure)
        logp = Descriptors.MolLogP(substructure)
        tpsa = Descriptors.TPSA(substructure)
        rot_bonds = Descriptors.NumRotatableBonds(substructure)
        formal_charge = sum(atom.GetFormalCharge() for atom in substructure.GetAtoms())
        return np.array([mw, logp, tpsa, rot_bonds, formal_charge], dtype=np.float32)
    elif node_type == "protein":
        atomic_weights = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06}
        weights = []
        for atom in substructure.get_atoms():
            elem = atom.element.strip()
            weights.append(atomic_weights.get(elem, 0.0))
        avg_weight = np.mean(weights) if weights else 0.0
        return np.array([avg_weight], dtype=np.float32)


def check_and_fix_features(features, feature_name="", sample_name=""):
    """Fix NaN/Inf in feature arrays."""
    if isinstance(features, np.ndarray):
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    elif torch.is_tensor(features):
        if torch.any(torch.isnan(features)) or torch.any(torch.isinf(features)):
            features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    return features


def safe_read_sdf(sdf_content, sample_name=""):
    """Robust SDF parsing with fallback sanitization."""
    mol = Chem.MolFromMolBlock(sdf_content, removeHs=False)
    if mol is None:
        mol = Chem.MolFromMolBlock(sdf_content, removeHs=False, sanitize=False)
        if mol is not None:
            try:
                mol.UpdatePropertyCache(strict=False)
                fast_sanitize = Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES ^ Chem.SANITIZE_CLEANUP
                Chem.SanitizeMol(mol, sanitizeOps=fast_sanitize)
                Chem.GetSymmSSSR(mol)
                try:
                    AllChem.ComputeGasteigerCharges(mol)
                except:
                    pass
            except:
                return None
    return mol


# ============================================================================
# K-Means for Group Assignment (matching train_stage1.py)
# ============================================================================

def simple_kmeans(x, k, max_iters=10):
    """Assign atoms to k groups based on spatial proximity."""
    device = x.device
    if k <= 0:
        return torch.zeros(x.size(0), dtype=torch.long, device=device), torch.zeros(0, x.size(1), device=device)
    if k >= x.size(0):
        return torch.arange(x.size(0), device=device), x

    indices = torch.randperm(x.size(0), device=device)[:k]
    centers = x[indices]
    labels = torch.zeros(x.size(0), dtype=torch.long, device=device)

    for _ in range(max_iters):
        dists = torch.cdist(x, centers)
        new_labels = torch.argmin(dists, dim=1)
        if torch.equal(labels, new_labels):
            break
        labels = new_labels
        new_centers = []
        for i in range(k):
            mask = (labels == i)
            if mask.sum() > 0:
                new_centers.append(x[mask].mean(dim=0))
            else:
                new_centers.append(centers[i])
        centers = torch.stack(new_centers, dim=0)
    return labels, centers


def patch_group_ids(graph_data):
    """
    Add 'group' ndata to atom graphs using K-Means clustering.
    Each atom is assigned to the nearest fragment/residue centroid.
    Mirrors patch_add_group_ids() from train_stage1.py.
    """
    print("Patching group IDs via K-Means clustering...")
    count = 0
    for comp_id, sample in graph_data.items():
        try:
            # Ligand: assign atoms to fragments
            l_atom_g = sample['ligand_atom_graph']
            l_frag_g = sample['ligand_fragment_graph']

            if 'pos' in l_atom_g.ndata and l_frag_g.num_nodes() > 0:
                atom_pos = l_atom_g.ndata['pos']
                num_frags = l_frag_g.num_nodes()
                labels, centers = simple_kmeans(atom_pos, num_frags)
                l_atom_g.ndata['group'] = labels.to(torch.int32)
                # Sync fragment positions if missing
                if 'pos' not in l_frag_g.ndata or l_frag_g.ndata['pos'] is None:
                    l_frag_g.ndata['pos'] = centers.to(torch.float32)

            # Protein: assign atoms to residues
            p_atom_g = sample['protein_atom_graph']
            p_res_g = sample['protein_residue_graph']

            if 'pos' in p_atom_g.ndata and p_res_g.num_nodes() > 0:
                atom_pos_p = p_atom_g.ndata['pos']
                num_res = p_res_g.num_nodes()
                labels_p, centers_p = simple_kmeans(atom_pos_p, num_res)
                p_atom_g.ndata['group'] = labels_p.to(torch.int32)
                if 'pos' not in p_res_g.ndata or p_res_g.ndata['pos'] is None:
                    p_res_g.ndata['pos'] = centers_p.to(torch.float32)

            count += 1
        except Exception as e:
            print(f"  Warning: Failed to patch group IDs for {comp_id}: {e}")

    print(f"  Patched {count}/{len(graph_data)} complexes")
    return graph_data


# ============================================================================
# Graph Construction Functions
# ============================================================================

def build_ligand_atom_graph(ligand_sdf_content, sample_name=""):
    """Build ligand atom graph from SDF content."""
    mol = safe_read_sdf(ligand_sdf_content, sample_name)
    if mol is None:
        return None

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except:
        pass

    if mol.GetNumConformers() == 0:
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        except:
            return None

    conf = mol.GetConformer()
    atom_features, atom_positions = [], []

    for i, atom in enumerate(mol.GetAtoms()):
        atom_features.append(featurize_atom(atom))
        try:
            pos = conf.GetAtomPosition(i)
            atom_positions.append([pos.x, pos.y, pos.z])
        except:
            atom_positions.append([0.0, 0.0, 0.0])

    atom_features = check_and_fix_features(np.array(atom_features, dtype=np.float32))
    atom_positions = check_and_fix_features(np.array(atom_positions, dtype=np.float32))

    src_list, dst_list = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src_list.extend([i, j])
        dst_list.extend([j, i])

    if len(src_list) == 0:
        n = mol.GetNumAtoms()
        src_list, dst_list = list(range(n)), list(range(n))

    g = dgl.graph((src_list, dst_list), num_nodes=mol.GetNumAtoms())
    g.ndata['h'] = torch.tensor(atom_features)
    g.ndata['pos'] = torch.tensor(atom_positions)

    return g


def build_protein_atom_graph(protein_pdb_content, sample_name=""):
    """Build protein atom graph from PDB content (pocket only)."""
    mol = Chem.MolFromPDBBlock(protein_pdb_content, removeHs=False, sanitize=False)
    if mol is None:
        return None

    try:
        mol.UpdatePropertyCache(strict=False)
    except:
        pass

    conf = mol.GetConformer()
    atom_features, atom_positions = [], []

    for i, atom in enumerate(mol.GetAtoms()):
        atom_features.append(featurize_atom(atom))
        try:
            pos = conf.GetAtomPosition(i)
            atom_positions.append([pos.x, pos.y, pos.z])
        except:
            atom_positions.append([0.0, 0.0, 0.0])

    atom_features = check_and_fix_features(np.array(atom_features, dtype=np.float32))
    atom_positions = check_and_fix_features(np.array(atom_positions, dtype=np.float32))

    src_list, dst_list = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src_list.extend([i, j])
        dst_list.extend([j, i])

    if len(src_list) == 0 and mol.GetNumAtoms() > 0:
        n = mol.GetNumAtoms()
        src_list, dst_list = list(range(n)), list(range(n))

    g = dgl.graph((src_list, dst_list), num_nodes=mol.GetNumAtoms())
    g.ndata['h'] = torch.tensor(atom_features)
    g.ndata['pos'] = torch.tensor(atom_positions)

    return g


def build_atom_interaction_graph(ligand_graph, protein_graph, d_atom):
    """Build bipartite atom interaction graph based on spatial proximity."""
    if ligand_graph is None or protein_graph is None:
        return None
    if ligand_graph.num_nodes() == 0 or protein_graph.num_nodes() == 0:
        return None

    l_pos = ligand_graph.ndata['pos'].numpy()
    p_pos = protein_graph.ndata['pos'].numpy()
    n_ligand, n_protein = l_pos.shape[0], p_pos.shape[0]

    diff = l_pos[:, np.newaxis, :] - p_pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    ligand_indices, protein_indices = np.where(dists <= d_atom)
    valid_dists = dists[ligand_indices, protein_indices]

    src_list, dst_list, edge_dists = [], [], []
    for l_idx, p_idx in zip(ligand_indices, protein_indices):
        d_val = dists[l_idx, p_idx]
        src_list.append(l_idx)
        dst_list.append(n_ligand + p_idx)
        edge_dists.append(d_val)
        src_list.append(n_ligand + p_idx)
        dst_list.append(l_idx)
        edge_dists.append(d_val)

    total_nodes = n_ligand + n_protein
    g = dgl.graph((src_list, dst_list), num_nodes=total_nodes)
    if edge_dists:
        g.edata['dist'] = torch.tensor(np.array(edge_dists, dtype=np.float32)).unsqueeze(-1)
    else:
        g.edata['dist'] = torch.tensor([], dtype=torch.float32).view(0, 1)

    return g


def build_ligand_fragment_graph(ligand_sdf_content, sample_name=""):
    """Build ligand fragment graph using BRICS decomposition."""
    mol = safe_read_sdf(ligand_sdf_content, sample_name)
    if mol is None:
        return None

    try:
        frag_smiles_set = BRICS.BRICSDecompose(mol)
        frag_smiles_list = sorted(list(frag_smiles_set))
        fragments = []
        for smi in frag_smiles_list:
            m = Chem.MolFromSmiles(smi)
            if m:
                fragments.append(m)
    except:
        try:
            smi = Chem.MolToSmiles(mol, canonical=True)
        except:
            smi = "C"
        fragments = [mol]

    node_features, node_positions = [], []
    conf = mol.GetConformer()

    for frag in fragments:
        try:
            feat = featurize_substructure(frag, node_type="ligand")
            node_features.append(feat)
        except:
            node_features.append(np.zeros(5, dtype=np.float32))

        matches = mol.GetSubstructMatches(frag)
        if matches:
            coords = np.array([list(conf.GetAtomPosition(i)) for i in matches[0]])
            center = np.mean(coords, axis=0)
        else:
            center = np.zeros(3, dtype=np.float32)
        node_positions.append(center)

    if not node_features:
        return None

    node_features = check_and_fix_features(np.array(node_features, dtype=np.float32))
    node_positions = check_and_fix_features(np.array(node_positions, dtype=np.float32))

    num_fragments = len(fragments)
    src_list, dst_list = [], []
    for i in range(num_fragments):
        for j in range(num_fragments):
            if i != j:
                src_list.append(i)
                dst_list.append(j)

    if len(src_list) == 0:
        src_list, dst_list = list(range(num_fragments)), list(range(num_fragments))

    g = dgl.graph((src_list, dst_list), num_nodes=num_fragments)
    g.ndata['h'] = torch.tensor(node_features)
    g.ndata['pos'] = torch.tensor(node_positions)

    return g


def build_protein_residue_graph(protein_pdb_content, d_res, sample_name=""):
    """Build protein residue graph from PDB content."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("x", io.StringIO(protein_pdb_content))
        model = structure[0]
    except:
        return None

    residues, ca_coords = [], []
    for chain in model:
        for res in chain:
            if res.id[0] != " ":
                continue
            if 'CA' in res:
                try:
                    coord = res['CA'].get_coord()
                    residues.append(res)
                    ca_coords.append(coord)
                except:
                    pass

    if not residues:
        return None

    ca_coords = check_and_fix_features(np.array(ca_coords, dtype=np.float32))

    node_features = []
    for res in residues:
        try:
            feat = featurize_substructure(res, node_type="protein")
            node_features.append(feat)
        except:
            node_features.append(np.zeros(1, dtype=np.float32))

    node_features = check_and_fix_features(np.array(node_features, dtype=np.float32))

    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    i_idxs, j_idxs = np.where((dists <= d_res) & (dists > 0))
    edge_dists = dists[i_idxs, j_idxs]

    g = dgl.graph((i_idxs, j_idxs), num_nodes=len(residues))
    g.ndata['h'] = torch.tensor(node_features)
    g.ndata['pos'] = torch.tensor(ca_coords)

    if len(edge_dists) > 0:
        g.edata['dist'] = torch.tensor(edge_dists, dtype=torch.float32).unsqueeze(-1)
    else:
        g.edata['dist'] = torch.tensor([], dtype=torch.float32).view(0, 1)

    return g


def build_substructure_interaction_graph(ligand_frag_g, protein_res_g, d_sub):
    """Build bipartite substructure interaction graph."""
    if ligand_frag_g is None or protein_res_g is None:
        return None
    if ligand_frag_g.num_nodes() == 0 or protein_res_g.num_nodes() == 0:
        return None

    l_pos = ligand_frag_g.ndata['pos'].numpy()
    p_pos = protein_res_g.ndata['pos'].numpy()
    n_lig, n_prot = l_pos.shape[0], p_pos.shape[0]

    diff = l_pos[:, np.newaxis, :] - p_pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    l_indices, p_indices = np.where(dists <= d_sub)
    valid_dists = dists[l_indices, p_indices]

    src_list, dst_list, edge_dists = [], [], []
    for l_idx, p_idx in zip(l_indices, p_indices):
        d_val = dists[l_idx, p_idx]
        src_list.append(l_idx)
        dst_list.append(n_lig + p_idx)
        edge_dists.append(d_val)
        src_list.append(n_lig + p_idx)
        dst_list.append(l_idx)
        edge_dists.append(d_val)

    total_nodes = n_lig + n_prot
    g = dgl.graph((src_list, dst_list), num_nodes=total_nodes)
    if edge_dists:
        g.edata['dist'] = torch.tensor(np.array(edge_dists, dtype=np.float32)).unsqueeze(-1)
    else:
        g.edata['dist'] = torch.tensor([], dtype=torch.float32).view(0, 1)

    return g


# ============================================================================
# Main Graph Building Pipeline
# ============================================================================

def build_all_graphs(raw_data, config, verbose=True):
    """
    Build all 6 graphs for each complex in raw_data.
    Adds K-Means group assignment for atom-to-substructure mapping.
    Returns dict: complex_id -> {ligand_atom_graph, ..., label}
    """
    d_atom = config.get('d_atom', 4.0)
    d_res = config.get('d_res', 8.0)
    d_sub = config.get('d_sub', 8.0)

    graph_data = {}
    stats = {'success': 0, 'failed': 0}

    for i, (comp_id, data) in enumerate(raw_data.items()):
        try:
            ligand_content = data['ligand']
            pocket_content = data['pocket']

            lig_atom_g = build_ligand_atom_graph(ligand_content, comp_id)
            if lig_atom_g is None:
                raise ValueError("Ligand atom graph build failed")

            prot_atom_g = build_protein_atom_graph(pocket_content, comp_id)
            if prot_atom_g is None:
                raise ValueError("Protein atom graph build failed")

            lig_frag_g = build_ligand_fragment_graph(ligand_content, comp_id)
            prot_res_g = build_protein_residue_graph(pocket_content, d_res, comp_id)

            if lig_frag_g is None or prot_res_g is None:
                raise ValueError("Fragment or residue graph build failed")

            atom_int_g = build_atom_interaction_graph(lig_atom_g, prot_atom_g, d_atom)
            sub_int_g = build_substructure_interaction_graph(lig_frag_g, prot_res_g, d_sub)

            graph_data[comp_id] = {
                'ligand_atom_graph': lig_atom_g,
                'protein_atom_graph': prot_atom_g,
                'atom_interaction_graph': atom_int_g,
                'ligand_fragment_graph': lig_frag_g,
                'protein_residue_graph': prot_res_g,
                'substructure_interaction_graph': sub_int_g,
                'label': data['label'],
            }

            stats['success'] += 1
            if verbose and i < 10:
                print(f"  [{comp_id}] OK - "
                      f"L_atom={lig_atom_g.num_nodes()}, P_atom={prot_atom_g.num_nodes()}, "
                      f"L_frag={lig_frag_g.num_nodes()}, P_res={prot_res_g.num_nodes()}, "
                      f"A_int_edges={atom_int_g.num_edges() if atom_int_g else 0}, "
                      f"S_int_edges={sub_int_g.num_edges() if sub_int_g else 0}, "
                      f"pKd={data['label']:.2f}")

        except Exception as e:
            stats['failed'] += 1
            if verbose:
                print(f"  [{comp_id}] FAILED: {e}")

    # Add K-Means group assignment for atom-to-substructure mapping
    graph_data = patch_group_ids(graph_data)

    print(f"\nGraph building complete: {stats['success']} success, {stats['failed']} failed")
    return graph_data


if __name__ == "__main__":
    import yaml

    with open("case_study_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    raw_data_path = "hiv_protease_raw.pkl"
    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} not found. Run extract_hiv_protease_data.py first.")
        sys.exit(1)

    with open(raw_data_path, 'rb') as f:
        raw_data = pickle.load(f)
    print(f"Loaded {len(raw_data)} raw complexes")

    print("\nBuilding graphs...")
    graph_data = build_all_graphs(raw_data, config, verbose=True)

    output_path = "hiv_protease_graphs.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(graph_data, f)
    print(f"\nSaved {len(graph_data)} graph complexes to {output_path}")
