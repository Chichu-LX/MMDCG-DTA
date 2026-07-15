"""
Inference + Interpretability Analysis for HIV-1 Protease Case Study.

Wraps MMDCGDTAModel_Stage1 to capture intermediate representations:
  - Atom-level intra/inter encoding states
  - Physics energy components (bond, angle, vdw, elec, hbond)
  - Bridge node states from hierarchical interactive learning
  - Substructure-level encoding states
  - Fusion representations
  - Graph topology statistics
"""

import os
import sys
import pickle
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import defaultdict
import json
import dgl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Data'))

from Data.MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1

try:
    from Utils.metrics import evaluate_metrics
except ImportError:
    def evaluate_metrics(y_true, y_pred):
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        vx = y_true - np.mean(y_true)
        vy = y_pred - np.mean(y_pred)
        std_x = np.std(y_true)
        std_y = np.std(y_pred)
        if std_x < 1e-6 or std_y < 1e-6:
            pearson = 0.0
        else:
            pearson = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + 1e-8)
        error = y_true - y_pred
        sd = np.std(error)
        return {"RMSE": rmse, "MAE": mae, "Pearson": pearson, "SD": sd}


# ============================================================================
# Interpretability Wrapper for MMDCGDTAModel_Stage1
# ============================================================================

class MMDCGDTAInterpretable(MMDCGDTAModel_Stage1):
    """
    Extended MMDCGDTAModel_Stage1 that saves intermediate representations.
    """

    def __init__(self, config):
        super().__init__(config)
        self.save_representations = False
        self.representations = {}

    def forward(self, sample):
        reps = {} if self.save_representations else None

        # 1. Extract graphs
        ligand_atom_graph = sample["ligand_atom_graph"]
        protein_atom_graph = sample["protein_atom_graph"]
        atom_interaction_graph = sample["atom_interaction_graph"]
        substructure_interaction_graph = sample["substructure_interaction_graph"]

        # 2. Calculate intra physics weights (bond energy)
        L_E_bond_agg, L_bond_weights = self._calc_bond_energy_weights(
            ligand_atom_graph, self.ligand_bond_sim
        )
        P_E_bond_agg, P_bond_weights = self._calc_bond_energy_weights(
            protein_atom_graph, self.protein_bond_sim
        )

        # 3. Intra encoding (with physics-informed edge weights)
        ligand_intra = self.ligand_atom_intra_encoder(
            ligand_atom_graph,
            ligand_atom_graph.ndata["h"],
            edge_weights=L_bond_weights
        )
        protein_intra = self.protein_atom_intra_encoder(
            protein_atom_graph,
            protein_atom_graph.ndata["h"],
            edge_weights=P_bond_weights
        )

        if reps is not None:
            reps['atom_intra_ligand'] = ligand_intra.detach().cpu()
            reps['atom_intra_protein'] = protein_intra.detach().cpu()
            reps['ligand_bond_energy'] = L_E_bond_agg.detach().cpu()
            reps['protein_bond_energy'] = P_E_bond_agg.detach().cpu()

        # 4. Physics simulation (angle, torsion)
        L_E_angle, L_E_torsion = self._calc_angle_energy(
            ligand_atom_graph, self.ligand_angle_sim, ligand_intra
        )
        P_E_angle, P_E_torsion = self._calc_angle_energy(
            protein_atom_graph, self.protein_angle_sim, protein_intra
        )

        # Inter-force energy
        I_E_vdw, I_E_elec, I_E_hbond = self._calc_inter_energy(
            atom_interaction_graph, ligand_intra, protein_intra
        )

        if reps is not None:
            reps['L_angle_energy'] = L_E_angle.detach().cpu()
            reps['L_torsion_energy'] = L_E_torsion.detach().cpu()
            reps['P_angle_energy'] = P_E_angle.detach().cpu()
            reps['P_torsion_energy'] = P_E_torsion.detach().cpu()
            reps['I_vdw_energy'] = I_E_vdw.detach().cpu()
            reps['I_elec_energy'] = I_E_elec.detach().cpu()
            reps['I_hbond_energy'] = I_E_hbond.detach().cpu()

        # 5. Inter encoding
        inter_lig, inter_prot = self.inter_atom_encoder(
            atom_interaction_graph, ligand_intra, protein_intra
        )

        if reps is not None:
            reps['atom_inter_ligand'] = inter_lig.detach().cpu()
            reps['atom_inter_protein'] = inter_prot.detach().cpu()

        # 6. HIL & Substructure
        ligand_group = self._get_batch_offset_group_ids(
            ligand_atom_graph, sample["ligand_fragment_graph"], "group"
        )
        protein_group = self._get_batch_offset_group_ids(
            protein_atom_graph, sample["protein_residue_graph"], "group"
        )

        updated_inter_lig, updated_intra_lig = self.ligand_atom_interactive(
            ligand_intra, inter_lig, ligand_group
        )
        updated_inter_prot, updated_intra_prot = self.protein_atom_interactive(
            protein_intra, inter_prot, protein_group
        )

        if reps is not None:
            reps['atom_hil_ligand_intra'] = updated_intra_lig.detach().cpu()
            reps['atom_hil_ligand_inter'] = updated_inter_lig.detach().cpu()
            reps['atom_hil_protein_intra'] = updated_intra_prot.detach().cpu()
            reps['atom_hil_protein_inter'] = updated_inter_prot.detach().cpu()

        H_lig_atom_final = updated_intra_lig
        H_prot_atom_final = updated_intra_prot

        # Aggregate atom -> substructure
        ligand_frag_graph = sample["ligand_fragment_graph"]
        protein_res_graph = sample["protein_residue_graph"]

        num_frags_total = ligand_frag_graph.num_nodes()
        atom_sum_lig = torch.zeros(num_frags_total, self.d, device=ligand_atom_graph.device)
        atom_sum_lig.index_add_(0, ligand_group, H_lig_atom_final)
        new_ligand_feats = torch.cat([ligand_frag_graph.ndata["h"], atom_sum_lig], dim=1)

        num_res_total = protein_res_graph.num_nodes()
        atom_sum_prot = torch.zeros(num_res_total, self.d, device=protein_atom_graph.device)
        atom_sum_prot.index_add_(0, protein_group, H_prot_atom_final)
        new_protein_feats = torch.cat([protein_res_graph.ndata["h"], atom_sum_prot], dim=1)

        if reps is not None:
            reps['sub_ligand_aggregated'] = new_ligand_feats.detach().cpu()
            reps['sub_protein_aggregated'] = new_protein_feats.detach().cpu()

        # Project to unified dimension
        ligand_sub_input = self.frag_proj(new_ligand_feats)
        protein_sub_input = self.res_proj(new_protein_feats)

        # Substructure intra encoding
        ligand_intra_sub = self.ligand_frag_intra_encoder(ligand_frag_graph, ligand_sub_input)
        prot_edge_feats = protein_res_graph.edata['dist'] if 'dist' in protein_res_graph.edata else None
        protein_intra_sub = self.protein_res_intra_encoder(protein_res_graph, protein_sub_input, prot_edge_feats)

        if reps is not None:
            reps['sub_intra_ligand'] = ligand_intra_sub.detach().cpu()
            reps['sub_intra_protein'] = protein_intra_sub.detach().cpu()

        # Substructure inter encoding
        inter_lig_sub, inter_prot_sub = self.inter_sub_encoder(
            substructure_interaction_graph,
            ligand_intra_sub,
            protein_intra_sub
        )

        if reps is not None:
            reps['sub_inter_ligand'] = inter_lig_sub.detach().cpu()
            reps['sub_inter_protein'] = inter_prot_sub.detach().cpu()

        # Substructure HIL
        ligand_sub_updated_inter, ligand_sub_updated_intra = self.ligand_sub_interactive(
            ligand_intra_sub, inter_lig_sub
        )
        protein_sub_updated_inter, protein_sub_updated_intra = self.protein_sub_interactive(
            protein_intra_sub, inter_prot_sub
        )

        if reps is not None:
            reps['sub_hil_ligand_intra'] = ligand_sub_updated_intra.detach().cpu()
            reps['sub_hil_ligand_inter'] = ligand_sub_updated_inter.detach().cpu()
            reps['sub_hil_protein_intra'] = protein_sub_updated_intra.detach().cpu()
            reps['sub_hil_protein_inter'] = protein_sub_updated_inter.detach().cpu()

        # Readout
        def safe_mean_nodes(g, feat):
            with g.local_scope():
                g.ndata['tmp_readout'] = feat
                return dgl.readout_nodes(g, 'tmp_readout', op='mean')

        ligand_pool_intra = safe_mean_nodes(ligand_frag_graph, ligand_sub_updated_intra)
        protein_pool_intra = safe_mean_nodes(protein_res_graph, protein_sub_updated_intra)
        ligand_pool_inter = safe_mean_nodes(ligand_frag_graph, ligand_sub_updated_inter)
        protein_pool_inter = safe_mean_nodes(protein_res_graph, protein_sub_updated_inter)

        H_gnn = torch.cat([ligand_pool_intra, protein_pool_intra,
                          ligand_pool_inter, protein_pool_inter], dim=1)

        H_physics = torch.cat([
            L_E_bond_agg, L_E_angle, L_E_torsion,
            P_E_bond_agg, P_E_angle, P_E_torsion,
            I_E_vdw, I_E_elec, I_E_hbond
        ], dim=1)

        if reps is not None:
            reps['fusion_gnn'] = H_gnn.detach().cpu()
            reps['fusion_physics'] = H_physics.detach().cpu()

        F_final = torch.cat([H_gnn, H_physics], dim=1)

        if reps is not None:
            reps['fusion_final'] = F_final.detach().cpu()

        F_final = F_final.unsqueeze(1)
        gru_out, _ = self.gru(F_final)
        fusion_rep = gru_out.squeeze(1)

        if reps is not None:
            reps['gru_output'] = fusion_rep.detach().cpu()

        y_pred = self.pred_fc(fusion_rep)

        if self.save_representations:
            self.representations = reps

        return y_pred


# ============================================================================
# Graph Topology Analyzer (Reconstructed Graph Information)
# ============================================================================

def analyze_graph_topology(sample, complex_id):
    """Extract detailed graph topology information for a complex."""
    info = {'complex_id': complex_id}

    # Ligand Atom Graph
    g = sample['ligand_atom_graph']
    info['ligand_atom_graph'] = {
        'num_nodes': g.num_nodes(),
        'num_edges': g.num_edges(),
        'avg_degree': g.num_edges() / max(1, g.num_nodes()),
        'feature_dim': list(g.ndata['h'].shape),
    }

    # Protein Atom Graph
    g = sample['protein_atom_graph']
    info['protein_atom_graph'] = {
        'num_nodes': g.num_nodes(),
        'num_edges': g.num_edges(),
        'avg_degree': g.num_edges() / max(1, g.num_nodes()),
        'feature_dim': list(g.ndata['h'].shape),
    }

    # Atom Interaction Graph
    g = sample.get('atom_interaction_graph')
    if g is not None and g.num_edges() > 0:
        info['atom_interaction_graph'] = {
            'num_nodes': g.num_nodes(),
            'num_edges': g.num_edges(),
            'ligand_ratio': (
                sample['ligand_atom_graph'].num_nodes() / max(1, g.num_nodes())
            ),
        }
        if 'dist' in g.edata:
            dists = g.edata['dist'].cpu().numpy()
            info['atom_interaction_graph']['dist_stats'] = {
                'min': float(dists.min()), 'max': float(dists.max()),
                'mean': float(dists.mean()), 'std': float(dists.std()),
            }

    # Ligand Fragment Graph
    g = sample['ligand_fragment_graph']
    info['ligand_fragment_graph'] = {
        'num_nodes': g.num_nodes(),
        'num_edges': g.num_edges(),
    }

    # Protein Residue Graph
    g = sample['protein_residue_graph']
    info['protein_residue_graph'] = {
        'num_nodes': g.num_nodes(),
        'num_edges': g.num_edges(),
    }

    # Substructure Interaction Graph
    g = sample.get('substructure_interaction_graph')
    if g is not None and g.num_edges() > 0:
        info['substructure_interaction_graph'] = {
            'num_nodes': g.num_nodes(),
            'num_edges': g.num_edges(),
        }
        if 'dist' in g.edata:
            dists = g.edata['dist'].cpu().numpy()
            info['substructure_interaction_graph']['dist_stats'] = {
                'min': float(dists.min()), 'max': float(dists.max()),
                'mean': float(dists.mean()), 'std': float(dists.std()),
            }

    return info


# ============================================================================
# Interpretability Analyzer
# ============================================================================

def compute_interpretability_metrics(reps, sample, complex_id):
    """Compute interpretability metrics from saved representations."""
    interp = {'complex_id': complex_id}

    # 1. Atom-level embedding magnitudes
    if 'atom_intra_ligand' in reps:
        h = reps['atom_intra_ligand']
        interp['atom_ligand_intra_norm'] = float(torch.norm(h).item())
        interp['atom_ligand_intra_mean'] = h.mean().item()

    if 'atom_intra_protein' in reps:
        h = reps['atom_intra_protein']
        interp['atom_protein_intra_norm'] = float(torch.norm(h).item())

    if 'atom_inter_ligand' in reps:
        interp['atom_ligand_inter_norm'] = float(torch.norm(reps['atom_inter_ligand']).item())

    if 'atom_inter_protein' in reps:
        interp['atom_protein_inter_norm'] = float(torch.norm(reps['atom_inter_protein']).item())

    # 2. HIL information flow (change after interactive learning)
    if 'atom_intra_ligand' in reps and 'atom_hil_ligand_intra' in reps:
        delta = reps['atom_hil_ligand_intra'] - reps['atom_intra_ligand']
        interp['atom_ligand_hil_change'] = float(torch.norm(delta).item())

    if 'atom_intra_protein' in reps and 'atom_hil_protein_intra' in reps:
        delta = reps['atom_hil_protein_intra'] - reps['atom_intra_protein']
        interp['atom_protein_hil_change'] = float(torch.norm(delta).item())

    # 3. Physics energy components
    for key in ['ligand_bond_energy', 'protein_bond_energy',
                'L_angle_energy', 'L_torsion_energy',
                'P_angle_energy', 'P_torsion_energy',
                'I_vdw_energy', 'I_elec_energy', 'I_hbond_energy']:
        if key in reps:
            interp[key] = float(reps[key].mean().item())

    # 4. Substructure-level metrics
    if 'sub_intra_ligand' in reps:
        interp['sub_ligand_intra_norm'] = float(torch.norm(reps['sub_intra_ligand']).item())

    if 'sub_intra_protein' in reps:
        interp['sub_protein_intra_norm'] = float(torch.norm(reps['sub_intra_protein']).item())

    if 'sub_intra_ligand' in reps and 'sub_hil_ligand_intra' in reps:
        delta = reps['sub_hil_ligand_intra'] - reps['sub_intra_ligand']
        interp['sub_ligand_hil_change'] = float(torch.norm(delta).item())

    # 5. Fusion representation analysis
    if 'fusion_gnn' in reps and 'fusion_physics' in reps:
        interp['fusion_gnn_norm'] = float(torch.norm(reps['fusion_gnn']).item())
        interp['fusion_physics_norm'] = float(torch.norm(reps['fusion_physics']).item())
        interp['gnn_physics_ratio'] = float(
            torch.norm(reps['fusion_gnn']).item() / max(torch.norm(reps['fusion_physics']).item(), 1e-8)
        )

    # 6. Ligand-protein interaction index
    if 'I_vdw_energy' in interp and 'I_elec_energy' in interp:
        total_inter = abs(interp['I_vdw_energy']) + abs(interp['I_elec_energy']) + abs(interp.get('I_hbond_energy', 0))
        interp['total_interaction_energy'] = total_inter

    return interp


# ============================================================================
# Inference Loop
# ============================================================================

def run_inference(model, graph_data, device, config):
    """
    Run inference on all complexes, collecting predictions, representations,
    and interpretability information.
    """
    model.eval()
    model.save_representations = True

    results = {}
    all_predictions = []

    for complex_id, sample in graph_data.items():
        # Move graphs to device
        sample_on_device = {}
        for key, val in sample.items():
            if hasattr(val, 'to'):
                sample_on_device[key] = val.to(device)
            else:
                sample_on_device[key] = val

        with torch.no_grad():
            try:
                y_pred = model(sample_on_device)
                reps = model.representations.copy()
            except Exception as e:
                print(f"  [{complex_id}] Inference error: {e}")
                import traceback
                traceback.print_exc()
                continue

        y_pred_val = y_pred.item()
        y_true_val = sample['label']

        # Collect results
        graph_info = analyze_graph_topology(sample, complex_id)
        interp_metrics = compute_interpretability_metrics(reps, sample, complex_id)

        results[complex_id] = {
            'predicted_pKd': y_pred_val,
            'true_pKd': y_true_val,
            'error': y_pred_val - y_true_val,
            'graph_topology': graph_info,
            'interpretability': interp_metrics,
            'representations': {k: v.numpy() if hasattr(v, 'numpy') else v
                              for k, v in reps.items()},
        }

        all_predictions.append((y_true_val, y_pred_val, complex_id))

    return results, all_predictions
