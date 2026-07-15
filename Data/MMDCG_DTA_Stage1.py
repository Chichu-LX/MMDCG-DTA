import torch
import torch.nn as nn
import dgl
from channels import (
    LigandAtomChannel, ProteinAtomChannel, InterAtomChannel,
    LigandFragmentChannel, ProteinResidueChannel, InterSubstructureChannel
)
from hil import (
    AtomLevelInteractiveLigand, AtomLevelInteractiveProtein,
    SubstructureLevelInteractiveLigand, SubstructureLevelInteractiveProtein
)
from mechanics import BondEnergyMLP, AngleDihedralEnergyMLP, InteractionForceMLP

class MMDCGDTAModel_Stage1(nn.Module):
    def __init__(self, config):
        super(MMDCGDTAModel_Stage1, self).__init__()
        # === 配置 ===
        self.l_intra = config["l_intra"]
        self.l_inter = config["l_inter"]
        self.l_atom = config["l_atom"]
        self.l_sub = config["l_sub"]
        self.d = config["embedding_dim"]
        self.neg_slope = config["inter_negative_slope"]
        self.d_atom = config["d_atom"]
        self.d_sub = config["d_sub"]
        self.use_checkpoint = config.get("use_checkpoint", True)

        self.raw_atom_dim = config.get("raw_atom_dim", 5) 
        self.sub_x_dim = config.get("sub_x_dim", 5)
        self.prot_res_dim = config.get("prot_res_dim", 1)

        # === 编码器 ===
        self.ligand_atom_intra_encoder = LigandAtomChannel(self.l_intra, self.raw_atom_dim, self.d, self.neg_slope)
        self.protein_atom_intra_encoder = ProteinAtomChannel(self.l_intra, self.raw_atom_dim, self.d, self.neg_slope)
        self.inter_atom_encoder = InterAtomChannel(self.l_inter, self.d, self.neg_slope)
        
        self.ligand_atom_interactive = AtomLevelInteractiveLigand(self.l_atom, self.d, use_checkpoint=self.use_checkpoint)
        self.protein_atom_interactive = AtomLevelInteractiveProtein(self.l_atom, self.d, use_checkpoint=self.use_checkpoint)

        # === Substructure 模块 ===
        raw_frag_in_dim = self.sub_x_dim + self.d
        raw_res_in_dim = self.prot_res_dim + self.d
        self.unified_sub_dim = self.d + self.sub_x_dim
        
        self.frag_proj = nn.Linear(raw_frag_in_dim, self.unified_sub_dim) 
        self.res_proj = nn.Linear(raw_res_in_dim, self.unified_sub_dim)   
        
        self.ligand_frag_intra_encoder = LigandFragmentChannel(self.l_intra, self.unified_sub_dim, self.unified_sub_dim, self.neg_slope)
        self.protein_res_intra_encoder = ProteinResidueChannel(self.l_intra, self.unified_sub_dim, self.unified_sub_dim, self.neg_slope)
        
        self.inter_sub_encoder = InterSubstructureChannel(self.l_inter, self.unified_sub_dim, self.neg_slope)
        
        self.ligand_sub_interactive = SubstructureLevelInteractiveLigand(self.l_sub, self.unified_sub_dim, self.neg_slope, use_checkpoint=self.use_checkpoint)
        self.protein_sub_interactive = SubstructureLevelInteractiveProtein(self.l_sub, self.unified_sub_dim, self.neg_slope, use_checkpoint=self.use_checkpoint)

        # === 分子力学模拟模块 ===
        self.ligand_bond_sim = BondEnergyMLP(hidden_dim=32)
        self.ligand_angle_sim = AngleDihedralEnergyMLP(in_dim=self.d, hidden_dim=32)
        self.protein_bond_sim = BondEnergyMLP(hidden_dim=32)
        self.protein_angle_sim = AngleDihedralEnergyMLP(in_dim=self.d, hidden_dim=32)
        self.inter_force_sim = InteractionForceMLP(atom_dim=self.d, hidden_dim=64)

        # === 融合与预测 ===
        self.fusion_dim = 4 * self.unified_sub_dim + 9 
        self.gru = nn.GRU(input_size=self.fusion_dim, hidden_size=self.fusion_dim, batch_first=True)
        self.pred_fc = nn.Linear(self.fusion_dim, 1)

    def _get_batch_offset_group_ids(self, atom_graph, sub_graph, group_key="group"):
        device = atom_graph.device
        sub_batch_num_nodes = sub_graph.batch_num_nodes().to(device)
        cumsum = torch.cumsum(sub_batch_num_nodes, dim=0)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), cumsum[:-1]])
        atom_batch_num_nodes = atom_graph.batch_num_nodes().to(device)
        atom_offsets = torch.repeat_interleave(offsets, atom_batch_num_nodes)
        if group_key not in atom_graph.ndata:
             raise ValueError(f"CRITICAL ERROR: '{group_key}' not found.")
        local_group_ids = atom_graph.ndata[group_key].long()
        return local_group_ids + atom_offsets

    # [辅助方法] 将能量转换为 0~1 权重 (Sigmoid)
    def _bond_energy_to_weight(self, bond_energy):
        return torch.sigmoid(-bond_energy)

    # [核心] 计算键能并返回用于 GAT 的权重
    def _calc_bond_energy_weights(self, graph, model, pos_key='pos'):
        src, dst = graph.edges()
        pos = graph.ndata[pos_key]
        dist = torch.norm(pos[src] - pos[dst], dim=-1, keepdim=True)
        
        # 计算每条边的能量
        energy_edges = model(dist)
        
        # 转换为 GAT 权重
        weights = self._bond_energy_to_weight(energy_edges)
        
        # 同时计算聚合能量 (用于最终特征融合)
        with graph.local_scope():
            graph.edata['e'] = energy_edges
            g_energy_agg = dgl.readout_edges(graph, 'e', weight=None, op='mean')
            
        return g_energy_agg, weights

    def _calc_angle_energy(self, graph, model, node_h):
        E_angle, E_torsion = model(node_h)
        with graph.local_scope():
            graph.ndata['tmp_angle_energy'] = E_angle
            graph.ndata['tmp_torsion_energy'] = E_torsion
            g_angle = dgl.readout_nodes(graph, 'tmp_angle_energy', op='mean')
            g_torsion = dgl.readout_nodes(graph, 'tmp_torsion_energy', op='mean')
        return g_angle, g_torsion

    def _calc_inter_energy(self, g, h_l, h_p):
        h_all = torch.cat([h_l, h_p], dim=0)
        with g.local_scope():
            src, dst = g.edges()
            h_src = h_all[src]
            h_dst = h_all[dst]
            d_val = g.edata['dist']
            
            E_vdw_edge, E_elec_edge, E_hbond_edge = self.inter_force_sim(h_src, h_dst, d_val)
            
            g.edata['E_vdw'] = E_vdw_edge
            g.edata['E_elec'] = E_elec_edge
            g.edata['E_hbond'] = E_hbond_edge
            
            E_vdw_batch = dgl.readout_edges(g, 'E_vdw', op='sum') 
            E_elec_batch = dgl.readout_edges(g, 'E_elec', op='sum')
            E_hbond_batch = dgl.readout_edges(g, 'E_hbond', op='sum')
            
        return E_vdw_batch, E_elec_batch, E_hbond_batch

    def forward(self, sample):
        # 1. 提取图
        ligand_atom_graph = sample["ligand_atom_graph"]
        protein_atom_graph = sample["protein_atom_graph"]
        atom_interaction_graph = sample["atom_interaction_graph"]
        substructure_interaction_graph = sample["substructure_interaction_graph"]
        
        # === 2. [调整] 先计算内轨物理权重 ===
        
        # 配体内部键能权重
        L_E_bond_agg, L_bond_weights = self._calc_bond_energy_weights(
            ligand_atom_graph, self.ligand_bond_sim
        )
        
        # 蛋白内部键能权重
        P_E_bond_agg, P_bond_weights = self._calc_bond_energy_weights(
            protein_atom_graph, self.protein_bond_sim
        )

        # === 3. Intra 编码 (融入内部物理权重) ===
        ligand_intra = self.ligand_atom_intra_encoder(
            ligand_atom_graph, 
            ligand_atom_graph.ndata["h"],
            edge_weights=L_bond_weights # 传入权重
        )
        protein_intra = self.protein_atom_intra_encoder(
            protein_atom_graph, 
            protein_atom_graph.ndata["h"],
            edge_weights=P_bond_weights # 传入权重
        )
        
        # === 4. 其他物理模拟 ===
        L_E_angle, L_E_torsion = self._calc_angle_energy(ligand_atom_graph, self.ligand_angle_sim, ligand_intra)
        P_E_angle, P_E_torsion = self._calc_angle_energy(protein_atom_graph, self.protein_angle_sim, protein_intra)
        
        I_E_vdw, I_E_elec, I_E_hbond = self._calc_inter_energy(atom_interaction_graph, ligand_intra, protein_intra)

        # === 5. 交互编码 ===
        inter_lig, inter_prot = self.inter_atom_encoder(atom_interaction_graph, ligand_intra, protein_intra)

        # === 6. HIL & Substructure ===
        ligand_group = self._get_batch_offset_group_ids(ligand_atom_graph, sample["ligand_fragment_graph"], "group")
        protein_group = self._get_batch_offset_group_ids(protein_atom_graph, sample["protein_residue_graph"], "group")
        
        updated_inter_lig, updated_intra_lig = self.ligand_atom_interactive(ligand_intra, inter_lig, ligand_group)
        updated_inter_prot, updated_intra_prot = self.protein_atom_interactive(protein_intra, inter_prot, protein_group)
        
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
            substructure_interaction_graph,
            ligand_intra_sub, 
            protein_intra_sub
        )

        ligand_sub_updated_inter, ligand_sub_updated_intra = self.ligand_sub_interactive(ligand_intra_sub, inter_lig_sub)
        protein_sub_updated_inter, protein_sub_updated_intra = self.protein_sub_interactive(protein_intra_sub, inter_prot_sub)

        def safe_mean_nodes(g, feat):
            with g.local_scope():
                g.ndata['tmp_readout'] = feat
                return dgl.readout_nodes(g, 'tmp_readout', op='mean')

        ligand_pool_intra = safe_mean_nodes(ligand_frag_graph, ligand_sub_updated_intra)
        protein_pool_intra = safe_mean_nodes(protein_res_graph, protein_sub_updated_intra)
        ligand_pool_inter = safe_mean_nodes(ligand_frag_graph, ligand_sub_updated_inter)
        protein_pool_inter = safe_mean_nodes(protein_res_graph, protein_sub_updated_inter)

        H_gnn = torch.cat([ligand_pool_intra, protein_pool_intra, ligand_pool_inter, protein_pool_inter], dim=1)
        
        H_physics = torch.cat([
            L_E_bond_agg, L_E_angle, L_E_torsion,  # 使用聚合后的键能
            P_E_bond_agg, P_E_angle, P_E_torsion,
            I_E_vdw, I_E_elec, I_E_hbond
        ], dim=1)

        F_final = torch.cat([H_gnn, H_physics], dim=1)
        
        F_final = F_final.unsqueeze(1) 
        gru_out, _ = self.gru(F_final)
        fusion_rep = gru_out.squeeze(1)

        y_pred = self.pred_fc(fusion_rep)
        return y_pred