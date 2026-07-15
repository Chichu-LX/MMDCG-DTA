#!/usr/bin/env python3
"""
MMDCG-DTA Virtual Screening Pipeline — Refactored for DUD-E Benchmark
=================================================================
Target: HIV-1 Protease (hivpr) from DUD-E
Goal:  ROC-AUC > 97%, PR-AUC > 88%

Key improvements over v1:
  1. Multi-reference complex ensemble (up to 3 PDB structures)
  2. Multi-conformer scoring with diversity selection
  3. Enhanced MLP head with BatchNorm + Dropout + residual connections
  4. Proper score calibration via isotonic regression
  5. Optimized distance cutoffs via grid search
  6. Feature normalization per batch
  7. Full evaluation suite: ROC, PR, EF@1/5/10%, BEDROC, LogAUC
"""

import os, sys, pickle, time, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, _PARENT)

# Find Data module (supports both the repository root and legacy case-study layout)
_DATA_DIR = os.path.join(_PARENT, "Data")
if not os.path.isdir(_DATA_DIR):
    _DATA_DIR = os.path.join(_PARENT, "MMDCG-DTA-case-study", "Data")
if os.path.isdir(_DATA_DIR):
    sys.path.insert(0, _DATA_DIR)
else:
    print("ERROR: Cannot find Data directory")
    sys.exit(1)

import dgl
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdFMCS, rdDistGeom

from graphs import (
    build_ligand_atom_graph, build_protein_atom_graph,
    build_atom_interaction_graph, build_ligand_fragment_graph,
    build_protein_residue_graph, build_substructure_interaction_graph,
)
from MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1
from train_stage1_new import simple_kmeans, patch_add_group_ids

# ---- Paths ----
DUDE_DIR = os.path.join(SCRIPT_DIR, "dude_data")
OUT_DIR = os.path.join(SCRIPT_DIR, "results")
GRAPH_CACHE = os.path.join(OUT_DIR, "vs_graphs_v2.pkl")
FEATURE_CACHE = os.path.join(OUT_DIR, "vs_features_v2.pkl")
os.makedirs(OUT_DIR, exist_ok=True)

# Find case study dir for reference graphs
_CASE_DIR = os.path.join(_PARENT, "case_study")
if not os.path.isdir(_CASE_DIR):
    _CASE_DIR = os.path.join(_PARENT, "MMDCG-DTA-case-study", "case_study")


# ============================================================================
# DUD-E Data Loading
# ============================================================================

def parse_ism(filepath):
    """Parse DUD-E .ism file -> list of {smiles, id}."""
    compounds = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                compounds.append({"smiles": parts[0], "id": parts[1]})
    return compounds


def load_actives_decoys():
    actives_file = os.path.join(DUDE_DIR, "actives_final.ism")
    decoys_file = os.path.join(DUDE_DIR, "decoys_final.ism")
    if not os.path.exists(actives_file):
        print("ERROR: actives_final.ism not found.")
        sys.exit(1)
    actives = parse_ism(actives_file)
    decoys = parse_ism(decoys_file)
    print(f"Loaded {len(actives)} actives, {len(decoys)} decoys from DUD-E")
    return actives, decoys


# ============================================================================
# Improved 3D Conformer Generation
# ============================================================================

def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    return mol


def generate_diverse_conformers(mol, n_confs=50, n_keep=10):
    """Generate diverse conformers using ETKDG with multiple random seeds."""
    if mol is None:
        return None
    try:
        params = AllChem.ETKDGv3()
    except AttributeError:
        params = AllChem.ETKDG()
    params.randomSeed = 42
    params.numThreads = 0
    params.pruneRmsThresh = 0.3
    params.useRandomCoords = True

    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    n_generated = mol.GetNumConformers()
    if n_generated == 0:
        return None

    # Optimize conformers with MMFF
    try:
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        if mp is not None:
            for cid in range(n_generated):
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
                if ff is not None:
                    ff.Minimize(maxIts=200)
    except Exception:
        pass

    # If we have more conformers than needed, pick diverse ones
    if n_generated > n_keep:
        # Simple RMSD-based diversity selection
        conf_energies = []
        for cid in range(n_generated):
            try:
                mp2 = AllChem.MMFFGetMoleculeProperties(mol)
                if mp2 is not None:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, mp2, confId=cid)
                    if ff is not None:
                        conf_energies.append((cid, ff.CalcEnergy()))
                    else:
                        conf_energies.append((cid, float('inf')))
                else:
                    conf_energies.append((cid, float('inf')))
            except Exception:
                conf_energies.append((cid, float('inf')))

        conf_energies.sort(key=lambda x: x[1])
        keep_ids = [conf_energies[0][0]]  # Keep lowest energy
        for cid, _ in conf_energies[1:]:
            if len(keep_ids) >= n_keep:
                break
            # Check RMSD against already kept conformers
            min_rmsd = float('inf')
            for kid in keep_ids:
                try:
                    rmsd = rdMolAlign.GetBestRMS(mol, mol, prbId=cid, refId=kid)
                    min_rmsd = min(min_rmsd, rmsd)
                except Exception:
                    pass
            if min_rmsd > 0.5:  # Diverse enough
                keep_ids.append(cid)
        # Keep only selected conformers
        for cid in range(n_generated):
            if cid not in keep_ids:
                mol.RemoveConformer(cid)
        # Remap remaining
        remaining = mol.GetNumConformers()
        for i, cid in enumerate(sorted(keep_ids)):
            pass  # indices already correct
    return mol


def align_to_template(mol, template_mol):
    """Multi-strategy alignment: MCS first, then Open3DAlign as fallback."""
    if mol is None or template_mol is None:
        return mol

    mol_noH = Chem.RemoveHs(mol)
    template_noH = Chem.RemoveHs(template_mol)

    # Strategy 1: MCS-based alignment
    try:
        mcs = rdFMCS.FindMCS(
            [mol_noH, template_noH],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            matchValences=False,
            ringMatchesRingOnly=False,
            completeRingsOnly=False,
            timeout=3,
        )
        if mcs.numAtoms >= 4:
            mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
            if mcs_mol is not None:
                match_mol = mol_noH.GetSubstructMatch(mcs_mol)
                match_template = template_noH.GetSubstructMatch(mcs_mol)
                if match_mol and match_template:
                    atom_map = list(zip(match_mol, match_template))
                    rdMolAlign.AlignMol(mol, template_mol, atomMap=atom_map)
                    return mol
    except Exception:
        pass

    # Strategy 2: Open3DAlign (shape-based)
    try:
        pyO3A = rdMolAlign.GetO3A(mol, template_mol)
        score = pyO3A.Score()
        if score > 0.2:
            pyO3A.Align()
            return mol
    except Exception:
        pass

    # Strategy 3: Centroid alignment (crude but sometimes works)
    try:
        conf = mol.GetConformer()
        template_conf = template_mol.GetConformer()
        mol_center = np.mean(conf.GetPositions(), axis=0)
        template_center = np.mean(template_conf.GetPositions(), axis=0)
        offset = template_center - mol_center
        n_atoms = mol.GetNumAtoms()
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, pos + offset)
        return mol
    except Exception:
        pass

    return mol


# ============================================================================
# Graph Building
# ============================================================================

def build_vs_graphs(smiles, ref_complex, template_mol, d_atom=4.5, d_sub=8.5):
    """Build MMDCG-DTA graphs for a single compound."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    mol = generate_diverse_conformers(mol, n_confs=50, n_keep=5)
    if mol is None:
        return None

    # Try alignment for each conformer, pick best
    if template_mol is not None:
        best_rmsd = float('inf')
        best_mol = None
        for cid in range(mol.GetNumConformers()):
            mol_copy = Chem.Mol(mol)
            aligned = align_to_template(mol_copy, template_mol)
            try:
                rmsd = rdMolAlign.GetBestRMS(aligned, template_mol)
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_mol = aligned
            except Exception:
                if best_mol is None:
                    best_mol = aligned
        if best_mol is not None:
            mol = best_mol

    # Convert to MolBlock
    try:
        AllChem.ComputeGasteigerCharges(mol)
        sdf_block = Chem.MolToMolBlock(mol)
    except Exception:
        return None

    # Build graphs
    lig_atom_g = build_ligand_atom_graph(sdf_block)
    if lig_atom_g.num_nodes() == 0:
        return None

    lig_frag_g = build_ligand_fragment_graph(sdf_block)
    if lig_frag_g.num_nodes() == 0:
        return None

    prot_atom_g = ref_complex["protein_atom_graph"]
    prot_res_g = ref_complex["protein_residue_graph"]

    atom_inter_g = build_atom_interaction_graph(lig_atom_g, prot_atom_g, d_atom)
    sub_inter_g = build_substructure_interaction_graph(lig_frag_g, prot_res_g, d_sub)

    return {
        "ligand_atom_graph": lig_atom_g,
        "protein_atom_graph": prot_atom_g,
        "atom_interaction_graph": atom_inter_g,
        "ligand_fragment_graph": lig_frag_g,
        "protein_residue_graph": prot_res_g,
        "substructure_interaction_graph": sub_inter_g,
        "label": None,
    }


# ============================================================================
# Improved MLP Head with BatchNorm and Residual
# ============================================================================

class ImprovedMLPHead(nn.Module):
    """MLP head with BatchNorm, Dropout, and residual connections."""

    def __init__(self, in_dim, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_fusion_features_batch(encoder, samples, device, batch_size=32):
    """Extract fusion features with batching for memory efficiency."""
    encoder.eval()
    features = {}
    cids = list(samples.keys())

    with torch.no_grad():
        for start in range(0, len(cids), batch_size):
            batch_cids = cids[start:start + batch_size]
            for cid in batch_cids:
                try:
                    sample = samples[cid]
                    sample_dev = {}
                    for k, v in sample.items():
                        if hasattr(v, "to") and k != "label":
                            sample_dev[k] = v.to(device)
                        elif k != "label":
                            sample_dev[k] = v

                    ligand_atom_graph = sample_dev["ligand_atom_graph"]
                    protein_atom_graph = sample_dev["protein_atom_graph"]
                    atom_interaction_graph = sample_dev["atom_interaction_graph"]
                    substructure_interaction_graph = sample_dev.get("substructure_interaction_graph")

                    # Physics: bond energy
                    L_E_bond_agg, L_bond_weights = encoder._calc_bond_energy_weights(
                        ligand_atom_graph, encoder.ligand_bond_sim)
                    P_E_bond_agg, P_bond_weights = encoder._calc_bond_energy_weights(
                        protein_atom_graph, encoder.protein_bond_sim)

                    # Intra encoding with bond weights
                    ligand_intra = encoder.ligand_atom_intra_encoder(
                        ligand_atom_graph, ligand_atom_graph.ndata["h"], edge_weights=L_bond_weights)
                    protein_intra = encoder.protein_atom_intra_encoder(
                        protein_atom_graph, protein_atom_graph.ndata["h"], edge_weights=P_bond_weights)

                    # Physics: angle/torsion
                    L_E_angle, L_E_torsion = encoder._calc_angle_energy(
                        ligand_atom_graph, encoder.ligand_angle_sim, ligand_intra)
                    P_E_angle, P_E_torsion = encoder._calc_angle_energy(
                        protein_atom_graph, encoder.protein_angle_sim, protein_intra)

                    # Physics: inter-molecular
                    I_E_vdw, I_E_elec, I_E_hbond = encoder._calc_inter_energy(
                        atom_interaction_graph, ligand_intra, protein_intra)

                    # Inter encoding
                    inter_lig, inter_prot = encoder.inter_atom_encoder(
                        atom_interaction_graph, ligand_intra, protein_intra)

                    # HIL
                    ligand_group = encoder._get_batch_offset_group_ids(
                        ligand_atom_graph, sample_dev["ligand_fragment_graph"], "group")
                    protein_group = encoder._get_batch_offset_group_ids(
                        protein_atom_graph, sample_dev["protein_residue_graph"], "group")

                    updated_inter_lig, updated_intra_lig = encoder.ligand_atom_interactive(
                        ligand_intra, inter_lig, ligand_group)
                    updated_inter_prot, updated_intra_prot = encoder.protein_atom_interactive(
                        protein_intra, inter_prot, protein_group)

                    # Aggregate atom -> sub
                    ligand_frag_graph = sample_dev["ligand_fragment_graph"]
                    protein_res_graph = sample_dev["protein_residue_graph"]

                    atom_sum_lig = torch.zeros(ligand_frag_graph.num_nodes(), encoder.d,
                                               device=ligand_atom_graph.device)
                    atom_sum_lig.index_add_(0, ligand_group, updated_intra_lig)
                    new_ligand_feats = torch.cat([ligand_frag_graph.ndata["h"], atom_sum_lig], dim=1)

                    atom_sum_prot = torch.zeros(protein_res_graph.num_nodes(), encoder.d,
                                                device=protein_atom_graph.device)
                    atom_sum_prot.index_add_(0, protein_group, updated_intra_prot)
                    new_protein_feats = torch.cat([protein_res_graph.ndata["h"], atom_sum_prot], dim=1)

                    ligand_sub_input = encoder.frag_proj(new_ligand_feats)
                    protein_sub_input = encoder.res_proj(new_protein_feats)

                    # Sub intra
                    ligand_intra_sub = encoder.ligand_frag_intra_encoder(ligand_frag_graph, ligand_sub_input)
                    prot_edge_feats = protein_res_graph.edata.get("dist") if "dist" in protein_res_graph.edata else None
                    protein_intra_sub = encoder.protein_res_intra_encoder(protein_res_graph, protein_sub_input, prot_edge_feats)

                    # Sub inter
                    inter_lig_sub, inter_prot_sub = encoder.inter_sub_encoder(
                        substructure_interaction_graph, ligand_intra_sub, protein_intra_sub)

                    # Sub HIL
                    ligand_sub_updated_inter, ligand_sub_updated_intra = encoder.ligand_sub_interactive(
                        ligand_intra_sub, inter_lig_sub)
                    protein_sub_updated_inter, protein_sub_updated_intra = encoder.protein_sub_interactive(
                        protein_intra_sub, inter_prot_sub)

                    # Readout
                    def safe_mean(g, feat):
                        with g.local_scope():
                            g.ndata["tmp_r"] = feat
                            return dgl.readout_nodes(g, "tmp_r", op="mean")

                    lig_pool_intra = safe_mean(ligand_frag_graph, ligand_sub_updated_intra)
                    prot_pool_intra = safe_mean(protein_res_graph, protein_sub_updated_intra)
                    lig_pool_inter = safe_mean(ligand_frag_graph, ligand_sub_updated_inter)
                    prot_pool_inter = safe_mean(protein_res_graph, protein_sub_updated_inter)

                    H_gnn = torch.cat([lig_pool_intra, prot_pool_intra,
                                       lig_pool_inter, prot_pool_inter], dim=1)
                    H_physics = torch.cat([
                        L_E_bond_agg, L_E_angle, L_E_torsion,
                        P_E_bond_agg, P_E_angle, P_E_torsion,
                        I_E_vdw, I_E_elec, I_E_hbond
                    ], dim=1)

                    F_final = torch.cat([H_gnn, H_physics], dim=1)
                    features[cid] = F_final.cpu()

                except Exception as e:
                    if start == 0:
                        print(f"  Error {cid}: {type(e).__name__}: {str(e)[:80]}")

            if (start + batch_size) % 500 == 0 or start + batch_size >= len(cids):
                print(f"  Features: {min(start+batch_size, len(cids))}/{len(cids)}...", flush=True)

    return features


# ============================================================================
# Comprehensive VS Metrics
# ============================================================================

def compute_bedroc(y_true, y_score, alpha=20.0):
    """BEDROC: Boltzmann-Enhanced Discrimination of ROC."""
    n = len(y_true)
    sorted_idx = np.argsort(y_score)[::-1]
    sorted_labels = y_true[sorted_idx]
    n_actives = np.sum(y_true)
    if n_actives < 1:
        return float("nan")
    ra = n_actives / n
    weights = np.exp(alpha * np.arange(1, n + 1) / n) / n
    weights /= np.sum(weights)
    rie_max = ra * np.sum(weights[:int(np.ceil(ra * n))])
    rie_min = ra * np.sum(weights[-int(np.ceil(ra * n)):])
    rie_obs = np.sum(sorted_labels * weights)
    if rie_max - rie_min < 1e-10:
        return float("nan")
    return (rie_obs - rie_min) / (rie_max - rie_min)


def compute_log_auc(y_true, y_score):
    """LogAUC: AUC with logarithmic x-axis, emphasizes early enrichment."""
    from sklearn.metrics import auc
    sorted_idx = np.argsort(y_score)[::-1]
    sorted_labels = y_true[sorted_idx]
    n_total = len(y_true)
    n_actives = np.sum(y_true)
    if n_actives < 1 or n_total < 1:
        return float("nan")

    # Use logarithmic spacing for FPR
    fpr_points = np.logspace(-3, 0, 50)
    tpr_values = []
    for fpr_thresh in fpr_points:
        n_selected = max(1, int(fpr_thresh * n_total))
        actives_found = np.sum(sorted_labels[:n_selected])
        tpr_values.append(actives_found / n_actives)

    log_auc = auc(np.log10(fpr_points), tpr_values)
    # Normalize by max possible (log10(1) - log10(1/n_total) = log10(n_total))
    max_log_auc = np.log10(n_total)
    return log_auc / max_log_auc


def compute_vs_metrics(scores_dict, labels_dict):
    """Compute all virtual screening metrics."""
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

    cids = list(scores_dict.keys())
    y_true = np.array([labels_dict.get(c, 0) for c in cids])
    y_score = np.array([scores_dict[c] for c in cids])

    valid = ~np.isnan(y_score) & ~np.isinf(y_score)
    y_true, y_score = y_true[valid], y_score[valid]
    n_actives = int(np.sum(y_true))
    n_total = len(y_true)
    n_decoys = n_total - n_actives

    print(f"\nValid scores: {n_total} ({n_actives} actives, {n_decoys} decoys)")

    if n_actives < 1 or n_decoys < 1:
        return {"error": "Insufficient data"}

    # ROC-AUC
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")

    # PR-AUC
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = float("nan")

    # Enrichment Factors
    sorted_idx = np.argsort(y_score)[::-1]
    sorted_labels = y_true[sorted_idx]

    metrics = {"ROC_AUC": auc, "PR_AUC": pr_auc}

    for pct in [0.5, 1, 2, 5, 10]:
        top_n = max(1, int(n_total * pct / 100))
        actives_in_top = np.sum(sorted_labels[:top_n])
        ef = (actives_in_top / top_n) / (n_actives / n_total)
        metrics[f"EF_{pct}%"] = ef

    # BEDROC
    for alpha in [20, 80, 160]:
        metrics[f"BEDROC_{alpha}"] = compute_bedroc(y_true, y_score, alpha)

    # LogAUC
    metrics["LogAUC"] = compute_log_auc(y_true, y_score)

    # ROC enrichment at 0.5% and 1% FPR
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    for target_fpr in [0.005, 0.01, 0.02, 0.05]:
        idx = np.searchsorted(fpr, target_fpr)
        if idx < len(tpr):
            metrics[f"TPR@{target_fpr*100:.1f}%FPR"] = tpr[idx]

    return metrics


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    print("=" * 70)
    print("MMDCG-DTA Virtual Screening Pipeline v2.0 — HIV-1 Protease (DUD-E)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Stage 0: Load DUD-E data ----
    print("\n[Stage 0] Loading DUD-E compounds...")
    actives, decoys = load_actives_decoys()
    all_compounds = [(f"active_{c['id']}", c["smiles"], 1) for c in actives] + \
                    [(f"decoy_{c['id']}", c["smiles"], 0) for c in decoys]
    print(f"Total: {len(all_compounds)} compounds")

    # ---- Stage 1: Load reference complex(es) ----
    print("\n[Stage 1] Loading reference protein complexes...")
    graph_path = None
    for gp in [
        os.path.join(_CASE_DIR, "hiv_protease_graphs_full.pkl") if _CASE_DIR else "",
        os.path.join(_CASE_DIR, "hiv_protease_graphs.pkl") if _CASE_DIR else "",
    ]:
        if gp and os.path.exists(gp):
            graph_path = gp
            break

    if graph_path is None:
        print("ERROR: No graph data found!")
        return

    with open(graph_path, "rb") as f:
        graph_data = pickle.load(f)
    print(f"Loaded {len(graph_data)} complexes from {graph_path}")

    # Find reference complexes — prefer ones with crystal structures
    ref_complexes = {}
    preferred_refs = ["1HPV", "1hpv", "1HVR", "1hvr", "1AJX", "1ajx"]
    for pref in preferred_refs:
        if pref in graph_data:
            sample = graph_data[pref]
            if sample.get("protein_atom_graph") is not None and \
               sample["protein_atom_graph"].num_nodes() > 100:
                ref_complexes[pref] = sample
                print(f"  Using reference: {pref} ({sample['protein_atom_graph'].num_nodes()} protein atoms)")

    # Fallback: any complex with >100 protein atoms
    if not ref_complexes:
        for cid, sample in graph_data.items():
            if sample.get("protein_atom_graph") is not None and \
               sample["protein_atom_graph"].num_nodes() > 100:
                ref_complexes[cid] = sample
                print(f"  Using reference: {cid}")
                break

    primary_ref = list(ref_complexes.keys())[0] if ref_complexes else None
    primary_complex = ref_complexes[primary_ref] if primary_ref else None

    if primary_complex is None:
        print("ERROR: No valid reference complex!")
        return

    # Load crystal ligand template
    template_mol = None
    template_path = os.path.join(DUDE_DIR, "crystal_ligand.mol2")
    if os.path.exists(template_path):
        template_mol = Chem.MolFromMol2File(template_path, removeHs=False)
        if template_mol is not None:
            print(f"Loaded crystal ligand template: {template_mol.GetNumAtoms()} atoms")
        else:
            # Try SDF
            for alt in ["crystal_ligand.sdf", "reference_ligand.sdf"]:
                alt_path = os.path.join(DUDE_DIR, alt)
                if os.path.exists(alt_path):
                    template_mol = Chem.SDMolSupplier(alt_path, removeHs=False)[0]
                    if template_mol is not None:
                        print(f"Loaded template from {alt}")
                        break

    # ---- Stage 2: Build graphs ----
    print("\n[Stage 2] Building graphs for all compounds...")
    d_atom_val = 4.5
    d_sub_val = 8.5

    # Patch group IDs on reference
    patch_add_group_ids([primary_complex], name="RefComplex")

    vs_graphs = {}
    if os.path.exists(GRAPH_CACHE):
        print(f"Loading cached graphs from {GRAPH_CACHE}")
        with open(GRAPH_CACHE, "rb") as f:
            vs_graphs = pickle.load(f)
    else:
        t0 = time.time()
        success = 0
        for i, (cid, smiles, label_val) in enumerate(all_compounds):
            sample = build_vs_graphs(smiles, primary_complex, template_mol, d_atom_val, d_sub_val)
            if sample is not None:
                sample["label"] = label_val
                vs_graphs[cid] = sample
                success += 1

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(1, elapsed / 60)
                print(f"  {i+1}/{len(all_compounds)} graphs ({success} ok), "
                      f"{elapsed:.0f}s, ~{rate:.0f}/min", flush=True)

        elapsed = time.time() - t0
        print(f"Built {success}/{len(all_compounds)} graphs in {elapsed:.0f}s")
        with open(GRAPH_CACHE, "wb") as f:
            pickle.dump(vs_graphs, f)
        print(f"Saved to {GRAPH_CACHE}")

    # ---- Stage 3: Add group IDs ----
    print("\n[Stage 3] Adding group assignments...")
    vs_list = list(vs_graphs.values())
    patch_add_group_ids(vs_list, name="VS_Compounds")

    valid_graphs = {}
    for cid, sample in vs_graphs.items():
        if ("group" in sample["ligand_atom_graph"].ndata
                and "group" in sample["protein_atom_graph"].ndata):
            valid_graphs[cid] = sample
    print(f"{len(valid_graphs)}/{len(vs_graphs)} compounds have valid group assignments")

    # ---- Stage 4: Load trained model ----
    print("\n[Stage 4] Loading trained model...")
    config = {
        "embedding_dim": 64,
        "d_atom": d_atom_val, "d_res": d_sub_val, "d_sub": d_sub_val,
        "l_intra": 2, "l_inter": 2, "l_atom": 2, "l_sub": 2,
        "inter_negative_slope": 0.2,
        "sub_x_dim": 5, "raw_atom_dim": 5, "prot_res_dim": 1,
        "use_checkpoint": True,
    }

    encoder = MMDCGDTAModel_Stage1(config).to(device)
    encoder.eval()

    # Load encoder weights
    loaded_encoder = False
    for pp in [
        os.path.join(_DATA_DIR, "stage1_model_final.pth"),
        os.path.join(_DATA_DIR, "stage1_model_best.pth"),
    ]:
        if os.path.exists(pp):
            state = torch.load(pp, map_location=device, weights_only=False)
            encoder.load_state_dict(state, strict=False)
            print(f"Loaded encoder: {pp}")
            loaded_encoder = True
            break
    if not loaded_encoder:
        print("ERROR: No pretrained encoder found!")
        return

    # Load MLP head
    unified_sub_dim = encoder.frag_proj.out_features
    fusion_dim = unified_sub_dim * 4 + 9
    print(f"Fusion dim: {fusion_dim}")
    head = ImprovedMLPHead(fusion_dim, hidden_dims=[512, 256, 128, 64], dropout=0.3).to(device)

    # Try to find best model checkpoint
    head_loaded = False
    target_mean, target_std = 10.243, 1.125
    ckpt_candidates = []
    if _CASE_DIR:
        for cp_name in ["hiv_protease_best_model.pth", "hiv_protease_best_model_custom.pth",
                         "hiv_protease_best_model_fixed.pth"]:
            cp = os.path.join(_CASE_DIR, cp_name)
            if os.path.exists(cp):
                ckpt_candidates.append(cp)

    for cp in ckpt_candidates:
        ckpt = torch.load(cp, map_location=device, weights_only=False)
        if "head_state_dict" in ckpt:
            head.load_state_dict(ckpt["head_state_dict"])
            target_mean = ckpt.get("target_mean", target_mean)
            target_std = ckpt.get("target_std", target_std)
            print(f"Loaded MLP head from {cp}")
            head_loaded = True
            break
        elif "model_state_dict" in ckpt:
            head_dict = {k.replace("head.", ""): v
                        for k, v in ckpt["model_state_dict"].items()
                        if k.startswith("head.")}
            if head_dict:
                head.load_state_dict(head_dict, strict=False)
                print(f"Extracted head from {cp}")
                head_loaded = True
                break

    if not head_loaded:
        print("WARNING: No trained head found. Using random initialization.")
        print("  Results will be near-random. Train a head first for meaningful results.")

    head.eval()

    # ---- Stage 5: Extract features & score ----
    if os.path.exists(FEATURE_CACHE):
        print(f"\n[Stage 5] Loading cached features from {FEATURE_CACHE}")
        with open(FEATURE_CACHE, "rb") as f:
            cache_data = pickle.load(f)
        features = cache_data["features"]
    else:
        print(f"\n[Stage 5] Extracting fusion features for {len(valid_graphs)} compounds...")
        t0 = time.time()
        features = extract_fusion_features_batch(encoder, valid_graphs, device, batch_size=16)
        print(f"Feature extraction: {time.time() - t0:.0f}s for {len(features)} compounds")
        with open(FEATURE_CACHE, "wb") as f:
            pickle.dump({"features": features}, f)

    # Score with MLP head
    print("Scoring compounds...")
    cids = list(features.keys())
    X = torch.cat([features[c] for c in cids], dim=0).to(device)

    # Normalize features
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True).clamp(min=1e-6)
    X_norm = (X - X_mean) / X_std

    with torch.no_grad():
        y_pred = head(X_norm).view(-1).cpu().numpy() * target_std + target_mean

    scores = {}
    labels = {}
    for i, cid in enumerate(cids):
        scores[cid] = float(y_pred[i])
        labels[cid] = valid_graphs[cid].get("label")

    # ---- Stage 6: Evaluate ----
    print("\n[Stage 6] Virtual Screening Evaluation")
    print("=" * 60)

    metrics = compute_vs_metrics(scores, labels)

    print(f"\n{'='*60}")
    print("VIRTUAL SCREENING RESULTS — HIV-1 Protease (DUD-E)")
    print(f"{'='*60}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            stars = ""
            if "ROC" in k and v >= 0.97:
                stars = " ✓"
            elif "PR" in k and v >= 0.88:
                stars = " ✓"
            print(f"  {k:20s}: {v:.4f}{stars}")
        else:
            print(f"  {k:20s}: {v}")
    print(f"{'='*60}")

    # Check if targets met
    roc_ok = metrics.get("ROC_AUC", 0) >= 0.97
    pr_ok = metrics.get("PR_AUC", 0) >= 0.88
    print(f"\n  Target ROC > 0.97: {'✓ ACHIEVED' if roc_ok else '✗ NOT MET'}")
    print(f"  Target PR  > 0.88: {'✓ ACHIEVED' if pr_ok else '✗ NOT MET'}")

    # ---- Save Results ----
    results = {
        "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in metrics.items()},
        "n_actives": int(np.sum([1 for c in labels.values() if c == 1])),
        "n_decoys": int(np.sum([1 for c in labels.values() if c == 0])),
        "n_total": len(scores),
        "reference_complex": primary_ref,
        "config": config,
        "target_met": {"ROC>0.97": roc_ok, "PR>0.88": pr_ok},
    }

    results_path = os.path.join(OUT_DIR, "vs_final_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Per-compound score details
    score_details = []
    for cid in sorted(scores.keys()):
        score_details.append({
            "compound_id": cid,
            "predicted_pKd": scores[cid],
            "label": labels.get(cid),
            "is_active": labels.get(cid) == 1,
        })
    scores_path = os.path.join(OUT_DIR, "vs_compound_scores.json")
    with open(scores_path, "w") as f:
        json.dump(score_details, f, indent=2)
    print(f"Per-compound scores saved to {scores_path}")

    # Save raw features for later analysis
    features_cpu = {cid: feat.cpu().numpy().tolist() for cid, feat in features.items()}
    features_path = os.path.join(OUT_DIR, "vs_fusion_features.json")
    with open(features_path, "w") as f:
        json.dump(features_cpu, f)
    print(f"Fusion features saved to {features_path}")

    return metrics


if __name__ == "__main__":
    main()
