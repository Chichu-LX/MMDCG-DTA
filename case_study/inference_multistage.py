"""
Multi-Stage Interpretability Inference for MMDCG-DTA Case Study.

Wraps MMDCGDTAModel_Stage1, Stage2, Stage3 to capture:
  - All intermediate representations (physics energies, HIL states, embeddings)
  - Stage 2/3 specific: edge reconstruction logits, ratios, edge weights
  - Graph topology statistics
  - Multi-stage comparison metrics

Matches the server path: /root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/Data/
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import dgl

# Path setup for server environment
SERVER_DATA_PATH = "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/Data"
sys.path.insert(0, SERVER_DATA_PATH)

from MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1
from MMDCG_DTA_Stage2 import MMDCGDTAModel_Stage2
from MMDCG_DTA_Stage3 import MMDCGDTAModel_Stage3


def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    vx = y_true - np.mean(y_true)
    vy = y_pred - np.mean(y_pred)
    std_x, std_y = np.std(y_true), np.std(y_pred)
    if std_x < 1e-6 or std_y < 1e-6:
        pearson = 0.0
    else:
        pearson = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + 1e-8)
    error = y_true - y_pred
    sd = np.std(error)
    return {"RMSE": rmse, "MAE": mae, "Pearson": pearson, "SD": sd}


# ============================================================================
# Stage 1 Interpretable Wrapper
# ============================================================================

class MMDCGDTAInterpretable_Stage1(MMDCGDTAModel_Stage1):
    """Extended Stage 1 that saves ALL intermediate representations."""

    def __init__(self, config):
        super().__init__(config)
        self.save_representations = False
        self.representations = {}

    def forward(self, sample):
        reps = {} if self.save_representations else None

        ligand_atom_graph = sample["ligand_atom_graph"]
        protein_atom_graph = sample["protein_atom_graph"]
        atom_interaction_graph = sample["atom_interaction_graph"]
        substructure_interaction_graph = sample["substructure_interaction_graph"]

        # Physics: bond energy
        L_E_bond_agg, L_bond_weights = self._calc_bond_energy_weights(
            ligand_atom_graph, self.ligand_bond_sim)
        P_E_bond_agg, P_bond_weights = self._calc_bond_energy_weights(
            protein_atom_graph, self.protein_bond_sim)

        # Intra encoding
        ligand_intra = self.ligand_atom_intra_encoder(
            ligand_atom_graph, ligand_atom_graph.ndata["h"], edge_weights=L_bond_weights)
        protein_intra = self.protein_atom_intra_encoder(
            protein_atom_graph, protein_atom_graph.ndata["h"], edge_weights=P_bond_weights)

        # Physics: angle/torsion
        L_E_angle, L_E_torsion = self._calc_angle_energy(
            ligand_atom_graph, self.ligand_angle_sim, ligand_intra)
        P_E_angle, P_E_torsion = self._calc_angle_energy(
            protein_atom_graph, self.protein_angle_sim, protein_intra)

        # Physics: inter-molecular
        I_E_vdw, I_E_elec, I_E_hbond = self._calc_inter_energy(
            atom_interaction_graph, ligand_intra, protein_intra)

        # Inter encoding
        inter_lig, inter_prot = self.inter_atom_encoder(
            atom_interaction_graph, ligand_intra, protein_intra)

        # HIL & Substructure
        ligand_group = self._get_batch_offset_group_ids(
            ligand_atom_graph, sample["ligand_fragment_graph"], "group")
        protein_group = self._get_batch_offset_group_ids(
            protein_atom_graph, sample["protein_residue_graph"], "group")

        updated_inter_lig, updated_intra_lig = self.ligand_atom_interactive(
            ligand_intra, inter_lig, ligand_group)
        updated_inter_prot, updated_intra_prot = self.protein_atom_interactive(
            protein_intra, inter_prot, protein_group)

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

        ligand_sub_input = self.frag_proj(new_ligand_feats)
        protein_sub_input = self.res_proj(new_protein_feats)

        # Sub intra encoding
        ligand_intra_sub = self.ligand_frag_intra_encoder(ligand_frag_graph, ligand_sub_input)
        prot_edge_feats = protein_res_graph.edata['dist'] if 'dist' in protein_res_graph.edata else None
        protein_intra_sub = self.protein_res_intra_encoder(protein_res_graph, protein_sub_input, prot_edge_feats)

        # Sub inter encoding
        inter_lig_sub, inter_prot_sub = self.inter_sub_encoder(
            substructure_interaction_graph, ligand_intra_sub, protein_intra_sub)

        # Sub HIL
        ligand_sub_updated_inter, ligand_sub_updated_intra = self.ligand_sub_interactive(
            ligand_intra_sub, inter_lig_sub)
        protein_sub_updated_inter, protein_sub_updated_intra = self.protein_sub_interactive(
            protein_intra_sub, inter_prot_sub)

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

        F_final = torch.cat([H_gnn, H_physics], dim=1)
        F_final = F_final.unsqueeze(1)
        gru_out, _ = self.gru(F_final)
        fusion_rep = gru_out.squeeze(1)
        y_pred = self.pred_fc(fusion_rep)

        if reps is not None:
            reps['ligand_intra'] = ligand_intra.detach().cpu()
            reps['protein_intra'] = protein_intra.detach().cpu()
            reps['atom_inter_ligand'] = inter_lig.detach().cpu()
            reps['atom_inter_protein'] = inter_prot.detach().cpu()
            reps['atom_hil_ligand_intra'] = updated_intra_lig.detach().cpu()
            reps['atom_hil_protein_intra'] = updated_intra_prot.detach().cpu()
            reps['sub_intra_ligand'] = ligand_intra_sub.detach().cpu()
            reps['sub_intra_protein'] = protein_intra_sub.detach().cpu()
            reps['sub_hil_ligand_intra'] = ligand_sub_updated_intra.detach().cpu()
            reps['sub_hil_protein_intra'] = protein_sub_updated_intra.detach().cpu()
            reps['L_bond_energy'] = L_E_bond_agg.detach().cpu()
            reps['P_bond_energy'] = P_E_bond_agg.detach().cpu()
            reps['L_angle_energy'] = L_E_angle.detach().cpu()
            reps['L_torsion_energy'] = L_E_torsion.detach().cpu()
            reps['P_angle_energy'] = P_E_angle.detach().cpu()
            reps['P_torsion_energy'] = P_E_torsion.detach().cpu()
            reps['I_vdw_energy'] = I_E_vdw.detach().cpu()
            reps['I_elec_energy'] = I_E_elec.detach().cpu()
            reps['I_hbond_energy'] = I_E_hbond.detach().cpu()
            reps['fusion_gnn'] = H_gnn.detach().cpu()
            reps['fusion_physics'] = H_physics.detach().cpu()
            reps['fusion_final'] = F_final.squeeze(1).detach().cpu()
            self.representations = reps

        return y_pred


# ============================================================================
# Stage 2 Interpretable Wrapper
# ============================================================================

class MMDCGDTAInterpretable_Stage2(MMDCGDTAModel_Stage2):
    """Extended Stage 2 that saves ALL intermediate representations including edge reconstruction."""

    def __init__(self, config):
        super().__init__(config)
        self.save_representations = False
        self.representations = {}

    def forward(self, sample):
        reps = {} if self.save_representations else None

        ligand_atom_graph = sample["ligand_atom_graph"]
        protein_atom_graph = sample["protein_atom_graph"]
        atom_interaction_graph = sample["atom_interaction_graph"]
        substructure_interaction_graph = sample["substructure_interaction_graph"]

        # Physics: bond energy
        L_E_bond_agg, L_bond_weights = self._calc_bond_energy_weights(
            ligand_atom_graph, self.ligand_bond_sim)
        P_E_bond_agg, P_bond_weights = self._calc_bond_energy_weights(
            protein_atom_graph, self.protein_bond_sim)

        # Intra encoding
        ligand_intra = self.ligand_atom_intra_encoder(
            ligand_atom_graph, ligand_atom_graph.ndata["h"], edge_weights=L_bond_weights)
        protein_intra = self.protein_atom_intra_encoder(
            protein_atom_graph, protein_atom_graph.ndata["h"], edge_weights=P_bond_weights)

        # Physics: angle/torsion/inter
        L_E_angle, L_E_torsion = self._calc_angle_energy(
            ligand_atom_graph, self.ligand_angle_sim, ligand_intra)
        P_E_angle, P_E_torsion = self._calc_angle_energy(
            protein_atom_graph, self.protein_angle_sim, protein_intra)
        I_E_vdw, I_E_elec, I_E_hbond = self._calc_inter_energy(
            atom_interaction_graph, ligand_intra, protein_intra)

        # [STAGE 2 KEY] Edge reconstruction
        edge_weights, recon_stats, edge_logits = self._run_edge_reconstruction(
            atom_interaction_graph, ligand_intra, protein_intra)
        flat_edge_energies = self._calc_inter_energy_for_loss(
            atom_interaction_graph, ligand_intra, protein_intra)

        if reps is not None:
            logits_cpu = edge_logits.detach().cpu()
            probs = F.softmax(edge_logits, dim=1).detach().cpu()
            reps['edge_logits'] = logits_cpu
            reps['edge_probs_remove'] = probs[:, 0]
            reps['edge_probs_keep'] = probs[:, 1]
            reps['edge_probs_add'] = probs[:, 2]
            reps['edge_weights'] = edge_weights.detach().cpu().squeeze(-1)
            reps['recon_stats'] = recon_stats
            reps['flat_edge_energies'] = flat_edge_energies.detach().cpu()

        # Inter encoding with reconstructed edge weights
        inter_lig, inter_prot = self.inter_atom_encoder(
            atom_interaction_graph, ligand_intra, protein_intra, edge_weights=edge_weights)

        # HIL & Substructure
        ligand_group = self._get_batch_offset_group_ids(
            ligand_atom_graph, sample["ligand_fragment_graph"], "group")
        protein_group = self._get_batch_offset_group_ids(
            protein_atom_graph, sample["protein_residue_graph"], "group")

        updated_inter_lig, updated_intra_lig = self.ligand_atom_interactive(
            ligand_intra, inter_lig, ligand_group)
        updated_inter_prot, updated_intra_prot = self.protein_atom_interactive(
            protein_intra, inter_prot, protein_group)

        H_lig_atom_final = updated_intra_lig
        H_prot_atom_final = updated_intra_prot

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

        ligand_sub_input = self.frag_proj(new_ligand_feats)
        protein_sub_input = self.res_proj(new_protein_feats)

        ligand_intra_sub = self.ligand_frag_intra_encoder(ligand_frag_graph, ligand_sub_input)
        prot_edge_feats = protein_res_graph.edata['dist'] if 'dist' in protein_res_graph.edata else None
        protein_intra_sub = self.protein_res_intra_encoder(protein_res_graph, protein_sub_input, prot_edge_feats)

        inter_lig_sub, inter_prot_sub = self.inter_sub_encoder(
            substructure_interaction_graph, ligand_intra_sub, protein_intra_sub)

        ligand_sub_updated_inter, ligand_sub_updated_intra = self.ligand_sub_interactive(
            ligand_intra_sub, inter_lig_sub)
        protein_sub_updated_inter, protein_sub_updated_intra = self.protein_sub_interactive(
            protein_intra_sub, inter_prot_sub)

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

        F_final = torch.cat([H_gnn, H_physics], dim=1)
        F_final = F_final.unsqueeze(1)
        gru_out, _ = self.gru(F_final)
        fusion_rep = gru_out.squeeze(1)
        y_pred = self.pred_fc(fusion_rep)

        if reps is not None:
            reps['ligand_intra'] = ligand_intra.detach().cpu()
            reps['protein_intra'] = protein_intra.detach().cpu()
            reps['atom_inter_ligand'] = inter_lig.detach().cpu()
            reps['atom_inter_protein'] = inter_prot.detach().cpu()
            reps['atom_hil_ligand_intra'] = updated_intra_lig.detach().cpu()
            reps['atom_hil_protein_intra'] = updated_intra_prot.detach().cpu()
            reps['sub_hil_ligand_intra'] = ligand_sub_updated_intra.detach().cpu()
            reps['sub_hil_protein_intra'] = protein_sub_updated_intra.detach().cpu()
            reps['L_bond_energy'] = L_E_bond_agg.detach().cpu()
            reps['P_bond_energy'] = P_E_bond_agg.detach().cpu()
            reps['L_angle_energy'] = L_E_angle.detach().cpu()
            reps['L_torsion_energy'] = L_E_torsion.detach().cpu()
            reps['P_angle_energy'] = P_E_angle.detach().cpu()
            reps['P_torsion_energy'] = P_E_torsion.detach().cpu()
            reps['I_vdw_energy'] = I_E_vdw.detach().cpu()
            reps['I_elec_energy'] = I_E_elec.detach().cpu()
            reps['I_hbond_energy'] = I_E_hbond.detach().cpu()
            reps['fusion_gnn'] = H_gnn.detach().cpu()
            reps['fusion_physics'] = H_physics.detach().cpu()
            self.representations = reps

        return y_pred


# ============================================================================
# Stage 3 Interpretable Wrapper
# ============================================================================

class MMDCGDTAInterpretable_Stage3(MMDCGDTAModel_Stage3):
    """Extended Stage 3 - frozen edge reconstructor, fine-tuned main network."""

    def __init__(self, config):
        super().__init__(config)
        self.save_representations = False
        self.representations = {}

    def forward(self, sample):
        # Identical to Stage 2 forward but with frozen edge_classifier
        reps = {} if self.save_representations else None

        ligand_atom_graph = sample["ligand_atom_graph"]
        protein_atom_graph = sample["protein_atom_graph"]
        atom_interaction_graph = sample["atom_interaction_graph"]
        substructure_interaction_graph = sample["substructure_interaction_graph"]

        L_E_bond_agg, L_bond_weights = self._calc_bond_energy_weights(
            ligand_atom_graph, self.ligand_bond_sim)
        P_E_bond_agg, P_bond_weights = self._calc_bond_energy_weights(
            protein_atom_graph, self.protein_bond_sim)

        ligand_intra = self.ligand_atom_intra_encoder(
            ligand_atom_graph, ligand_atom_graph.ndata["h"], edge_weights=L_bond_weights)
        protein_intra = self.protein_atom_intra_encoder(
            protein_atom_graph, protein_atom_graph.ndata["h"], edge_weights=P_bond_weights)

        L_E_angle, L_E_torsion = self._calc_angle_energy(
            ligand_atom_graph, self.ligand_angle_sim, ligand_intra)
        P_E_angle, P_E_torsion = self._calc_angle_energy(
            protein_atom_graph, self.protein_angle_sim, protein_intra)
        I_E_vdw, I_E_elec, I_E_hbond = self._calc_inter_energy(
            atom_interaction_graph, ligand_intra, protein_intra)

        edge_weights, recon_stats, edge_logits = self._run_edge_reconstruction(
            atom_interaction_graph, ligand_intra, protein_intra)

        if reps is not None:
            probs = F.softmax(edge_logits, dim=1).detach().cpu()
            reps['edge_logits'] = edge_logits.detach().cpu()
            reps['edge_probs_remove'] = probs[:, 0]
            reps['edge_probs_keep'] = probs[:, 1]
            reps['edge_probs_add'] = probs[:, 2]
            reps['edge_weights'] = edge_weights.detach().cpu().squeeze(-1)
            reps['recon_stats'] = recon_stats

        inter_lig, inter_prot = self.inter_atom_encoder(
            atom_interaction_graph, ligand_intra, protein_intra, edge_weights=edge_weights)

        ligand_group = self._get_batch_offset_group_ids(
            ligand_atom_graph, sample["ligand_fragment_graph"], "group")
        protein_group = self._get_batch_offset_group_ids(
            protein_atom_graph, sample["protein_residue_graph"], "group")

        updated_inter_lig, updated_intra_lig = self.ligand_atom_interactive(
            ligand_intra, inter_lig, ligand_group)
        updated_inter_prot, updated_intra_prot = self.protein_atom_interactive(
            protein_intra, inter_prot, protein_group)

        H_lig_atom_final = updated_intra_lig
        H_prot_atom_final = updated_intra_prot

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

        ligand_sub_input = self.frag_proj(new_ligand_feats)
        protein_sub_input = self.res_proj(new_protein_feats)

        ligand_intra_sub = self.ligand_frag_intra_encoder(ligand_frag_graph, ligand_sub_input)
        prot_edge_feats = protein_res_graph.edata['dist'] if 'dist' in protein_res_graph.edata else None
        protein_intra_sub = self.protein_res_intra_encoder(protein_res_graph, protein_sub_input, prot_edge_feats)

        inter_lig_sub, inter_prot_sub = self.inter_sub_encoder(
            substructure_interaction_graph, ligand_intra_sub, protein_intra_sub)

        ligand_sub_updated_inter, ligand_sub_updated_intra = self.ligand_sub_interactive(
            ligand_intra_sub, inter_lig_sub)
        protein_sub_updated_inter, protein_sub_updated_intra = self.protein_sub_interactive(
            protein_intra_sub, inter_prot_sub)

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

        F_final = torch.cat([H_gnn, H_physics], dim=1)
        F_final = F_final.unsqueeze(1)
        gru_out, _ = self.gru(F_final)
        fusion_rep = gru_out.squeeze(1)
        y_pred = self.pred_fc(fusion_rep)

        if reps is not None:
            reps['ligand_intra'] = ligand_intra.detach().cpu()
            reps['protein_intra'] = protein_intra.detach().cpu()
            reps['atom_hil_protein_intra'] = updated_intra_prot.detach().cpu()
            reps['sub_hil_protein_intra'] = protein_sub_updated_intra.detach().cpu()
            reps['I_vdw_energy'] = I_E_vdw.detach().cpu()
            reps['I_elec_energy'] = I_E_elec.detach().cpu()
            reps['I_hbond_energy'] = I_E_hbond.detach().cpu()
            reps['fusion_gnn'] = H_gnn.detach().cpu()
            reps['fusion_physics'] = H_physics.detach().cpu()
            self.representations = reps

        return y_pred


# ============================================================================
# Graph Topology Analyzer
# ============================================================================

def analyze_graph_topology(sample, complex_id):
    info = {'complex_id': complex_id}
    for gname in ['ligand_atom_graph', 'protein_atom_graph',
                   'ligand_fragment_graph', 'protein_residue_graph']:
        g = sample[gname]
        info[gname] = {
            'num_nodes': g.num_nodes(),
            'num_edges': g.num_edges(),
        }
    g = sample.get('atom_interaction_graph')
    if g is not None and g.num_edges() > 0:
        info['atom_interaction_graph'] = {
            'num_nodes': g.num_nodes(), 'num_edges': g.num_edges(),
        }
        if 'dist' in g.edata:
            dists = g.edata['dist'].cpu().numpy()
            info['atom_interaction_graph']['dist_stats'] = {
                'min': float(dists.min()), 'max': float(dists.max()),
                'mean': float(dists.mean()), 'std': float(dists.std()),
            }
    return info


# ============================================================================
# Interpretability Metric Computer
# ============================================================================

def compute_interpretability_metrics(reps, sample, complex_id):
    interp = {'complex_id': complex_id}
    if 'ligand_intra' in reps:
        interp['atom_ligand_intra_norm'] = float(torch.norm(reps['ligand_intra']).item())
    if 'protein_intra' in reps:
        interp['atom_protein_intra_norm'] = float(torch.norm(reps['protein_intra']).item())
    if 'atom_inter_ligand' in reps:
        interp['atom_ligand_inter_norm'] = float(torch.norm(reps['atom_inter_ligand']).item())
    if 'atom_inter_protein' in reps:
        interp['atom_protein_inter_norm'] = float(torch.norm(reps['atom_inter_protein']).item())
    if 'ligand_intra' in reps and 'atom_hil_ligand_intra' in reps:
        delta = reps['atom_hil_ligand_intra'] - reps['ligand_intra']
        interp['atom_ligand_hil_change'] = float(torch.norm(delta).item())
    if 'protein_intra' in reps and 'atom_hil_protein_intra' in reps:
        delta = reps['atom_hil_protein_intra'] - reps['protein_intra']
        interp['atom_protein_hil_change'] = float(torch.norm(delta).item())
    for key in ['L_bond_energy', 'P_bond_energy', 'L_angle_energy', 'L_torsion_energy',
                'P_angle_energy', 'P_torsion_energy',
                'I_vdw_energy', 'I_elec_energy', 'I_hbond_energy']:
        if key in reps:
            interp[key] = float(reps[key].mean().item())
    if 'fusion_gnn' in reps and 'fusion_physics' in reps:
        interp['fusion_gnn_norm'] = float(torch.norm(reps['fusion_gnn']).item())
        interp['fusion_physics_norm'] = float(torch.norm(reps['fusion_physics']).item())
        interp['gnn_physics_ratio'] = float(
            torch.norm(reps['fusion_gnn']).item() /
            max(torch.norm(reps['fusion_physics']).item(), 1e-8))
    if 'I_vdw_energy' in interp:
        interp['total_interaction_energy'] = (
            abs(interp['I_vdw_energy']) +
            abs(interp['I_elec_energy']) +
            abs(interp.get('I_hbond_energy', 0)))
    if 'recon_stats' in reps:
        rs = reps['recon_stats']
        interp['edge_keep_ratio'] = rs['ratio_keep']
        interp['edge_remove_ratio'] = rs['ratio_remove']
        interp['edge_add_ratio'] = rs['ratio_add']
        interp['edge_total'] = rs['total']
    return interp


# ============================================================================
# Multi-Stage Inference Runner
# ============================================================================

def run_multistage_inference(models_dict, graph_data, device):
    """
    Run inference with multiple stage models on all complexes.
    models_dict: {'stage1': model1, 'stage2': model2, 'stage3': model3}
    Returns nested results dict.
    """
    all_results = {}
    all_predictions = {}

    for stage_name, model in models_dict.items():
        model.eval()
        model.save_representations = True
        results = {}
        predictions = []

        for complex_id, sample in graph_data.items():
            sample_dev = {}
            for k, v in sample.items():
                if hasattr(v, 'to'):
                    sample_dev[k] = v.to(device)
                else:
                    sample_dev[k] = v

            with torch.no_grad():
                try:
                    y_pred = model(sample_dev)
                    y_pred_val = y_pred.item() if isinstance(y_pred, torch.Tensor) else y_pred[0].item()
                    reps = model.representations.copy()
                except Exception as e:
                    print(f"  [{stage_name}][{complex_id}] Error: {e}")
                    continue

            y_true_val = sample['label']
            graph_info = analyze_graph_topology(sample, complex_id)
            interp_metrics = compute_interpretability_metrics(reps, sample, complex_id)

            results[complex_id] = {
                'predicted_pKd': y_pred_val,
                'true_pKd': y_true_val,
                'error': y_pred_val - y_true_val,
                'graph_topology': graph_info,
                'interpretability': interp_metrics,
            }
            predictions.append((y_true_val, y_pred_val, complex_id))

        all_results[stage_name] = results
        all_predictions[stage_name] = predictions

    return all_results, all_predictions
