import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

# 继承 Stage 1 模型
# 确保你使用的 MMDCG_DTA_Stage1 是包含 _calc_bond_energy_weights 方法的新版本
from MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1
from edge_reconstructor import EdgeReconstructor

class MMDCGDTAModel_Stage2(MMDCGDTAModel_Stage1):
    def __init__(self, config):
        # 初始化 Stage 1 (包含所有 GNN 编码器, 分子力学 MLP, HIL 模块)
        super(MMDCGDTAModel_Stage2, self).__init__(config)
        
        # [Stage 2 新增] 边重构器 (3分类任务)
        # 输入维度: h_src(d) + h_dst(d) + dist(1) = 2d + 1
        # 输出维度: 3 (Class 0:Remove, 1:Keep, 2:Add)
        self.edge_classifier = EdgeReconstructor(input_dim=self.d, hidden_dim=64)

    def _run_edge_reconstruction(self, g, h_l, h_p):
        """
        [修改] 针对预构建图 g 的边进行三分类预测。
        输入: 
            g: atom_interaction_graph (DGLGraph)
            h_l: 配体原子特征 [N_l, d]
            h_p: 蛋白原子特征 [N_p, d]
        输出:
            edge_weights: 用于 GNN 的标量权重 [E, 1]
            recon_stats: 统计信息 (各类别比例)
            logits: 用于计算 CrossEntropy Loss [E, 3]
        """
        # 1. 拼接特征 (因为 g 是二部图，节点索引是连续的)
        h_all = torch.cat([h_l, h_p], dim=0)
        
        with g.local_scope():
            # 获取边的源/目标节点索引
            src, dst = g.edges()
            
            # 提取特征
            h_src = h_all[src]
            h_dst = h_all[dst]
            
            # 获取预计算的距离特征 [E, 1]
            d_val = g.edata['dist'] 
            
            # 2. 预测 Logits [E, 3]
            logits = self.edge_classifier(h_src, h_dst, d_val)
            
            # 3. 计算概率 (Softmax)
            probs = F.softmax(logits, dim=1) # [E, 3]
            
            # 4. 生成 GNN 权重 (Soft Attention Mask)
            # 策略: 
            #   Class 0 (Remove): 权重贡献 0 (抑制)
            #   Class 1 (Keep):   权重贡献 1.0 * prob (保持)
            #   Class 2 (Add):    权重贡献 2.0 * prob (增强)
            
            p_keep = probs[:, 1]
            p_add = probs[:, 2]
            
            edge_weights = (p_keep * 1.0 + p_add * 2.0).unsqueeze(-1) # [E, 1]
            
            # 5. 统计信息 (用于监控训练状态)
            pred_labels = torch.argmax(probs, dim=1)
            total = probs.size(0) + 1e-8
            n_remove = (pred_labels == 0).sum().item()
            n_keep = (pred_labels == 1).sum().item()
            n_add = (pred_labels == 2).sum().item()
            
            recon_stats = {
                "total": int(total),
                "n_remove": n_remove,
                "n_keep": n_keep,
                "n_add": n_add,
                "ratio_remove": n_remove / total,
                "ratio_keep": n_keep / total,
                "ratio_add": n_add / total
            }
            
            return edge_weights, recon_stats, logits

    def _calc_inter_energy_for_loss(self, g, h_l, h_p):
        """
        [修改] 为 Loss 计算每条边的物理能量。
        直接利用预构建图 g 的边和距离，不再重新计算。
        用于 Stage 2/3 的物理一致性辅助 Loss。
        """
        h_all = torch.cat([h_l, h_p], dim=0)
        
        with g.local_scope():
            src, dst = g.edges()
            h_src = h_all[src]
            h_dst = h_all[dst]
            d_val = g.edata['dist']
            
            # 调用 Stage 1 中定义的物理力学模块 (InteractionForceMLP)
            # 输出: E_vdw, E_elec, E_hbond
            E_vdw, E_elec, E_hbond = self.inter_force_sim(h_src, h_dst, d_val)
            
            # 总能量用于 PhysicsConsistencyLoss (能量越低越稳定)
            # 这将被用于监督 edge_logits，使物理上稳定的边更有可能被分类为 Keep/Add
            E_total = E_vdw + E_elec + E_hbond
            
            return E_total.squeeze(-1) # [E]

    def forward(self, sample):
        # === 1. 提取图对象 ===
        ligand_atom_graph = sample["ligand_atom_graph"]
        protein_atom_graph = sample["protein_atom_graph"]
        atom_interaction_graph = sample["atom_interaction_graph"]
        substructure_interaction_graph = sample["substructure_interaction_graph"]
        
        # === 2. [关键修复] 先计算内轨物理权重 (Intra Physics) ===
        # 这里不再调用 _calc_bond_energy，而是调用 Stage 1 新增的 _calc_bond_energy_weights
        
        # A. 配体内部键能权重
        L_E_bond_agg, L_bond_weights = self._calc_bond_energy_weights(
            ligand_atom_graph, self.ligand_bond_sim
        )
        
        # B. 蛋白内部键能权重
        P_E_bond_agg, P_bond_weights = self._calc_bond_energy_weights(
            protein_atom_graph, self.protein_bond_sim
        )

        # === 3. Intra 编码 (融入内部物理权重) ===
        ligand_intra = self.ligand_atom_intra_encoder(
            ligand_atom_graph, 
            ligand_atom_graph.ndata["h"],
            edge_weights=L_bond_weights # 传入
        )
        protein_intra = self.protein_atom_intra_encoder(
            protein_atom_graph, 
            protein_atom_graph.ndata["h"],
            edge_weights=P_bond_weights # 传入
        )
        
        # === 4. 其他物理模拟 (用于 Feature Fusion) ===
        # Intra Physics (Angle/Torsion)
        L_E_angle, L_E_torsion = self._calc_angle_energy(ligand_atom_graph, self.ligand_angle_sim, ligand_intra)
        P_E_angle, P_E_torsion = self._calc_angle_energy(protein_atom_graph, self.protein_angle_sim, protein_intra)
        
        # Inter Physics (聚合后的，用于亲和力预测)
        I_E_vdw, I_E_elec, I_E_hbond = self._calc_inter_energy(atom_interaction_graph, ligand_intra, protein_intra)

        # === 5. [Stage 2] 边重构预测 ===
        # 对 atom_interaction_graph 中的边进行三分类打分
        edge_weights, recon_stats, edge_logits = self._run_edge_reconstruction(
            atom_interaction_graph, ligand_intra, protein_intra
        )
        
        # 计算对应的物理能量 (用于辅助 Loss，未聚合，对应每一条边)
        flat_edge_energies = self._calc_inter_energy_for_loss(
            atom_interaction_graph, ligand_intra, protein_intra
        )

        # === 6. 交互编码 (应用重构权重) ===
        # 将 edge_weights 传入 encoder
        inter_lig, inter_prot = self.inter_atom_encoder(
            atom_interaction_graph, 
            ligand_intra, 
            protein_intra, 
            edge_weights=edge_weights # [关键] 传入权重
        )

        # === 7. HIL & Substructure (保持不变) ===
        ligand_group = self._get_batch_offset_group_ids(ligand_atom_graph, sample["ligand_fragment_graph"], "group")
        protein_group = self._get_batch_offset_group_ids(protein_atom_graph, sample["protein_residue_graph"], "group")
        
        updated_inter_lig, updated_intra_lig = self.ligand_atom_interactive(ligand_intra, inter_lig, ligand_group)
        updated_inter_prot, updated_intra_prot = self.protein_atom_interactive(protein_intra, inter_prot, protein_group)
        
        H_lig_atom_final = updated_intra_lig
        H_prot_atom_final = updated_intra_prot

        # Substructure Aggregation
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

        # Inter Substructure Interaction
        # [修改] 使用预构建的子结构交互图
        inter_lig_sub, inter_prot_sub = self.inter_sub_encoder(
            substructure_interaction_graph, 
            ligand_intra_sub, 
            protein_intra_sub
        )

        ligand_sub_updated_inter, ligand_sub_updated_intra = self.ligand_sub_interactive(ligand_intra_sub, inter_lig_sub)
        protein_sub_updated_inter, protein_sub_updated_intra = self.protein_sub_interactive(protein_intra_sub, inter_prot_sub)

        # Readout (Safe Mode)
        def safe_mean_nodes(g, feat):
            with g.local_scope():
                g.ndata['tmp_readout'] = feat
                return dgl.readout_nodes(g, 'tmp_readout', op='mean')

        ligand_pool_intra = safe_mean_nodes(ligand_frag_graph, ligand_sub_updated_intra)
        protein_pool_intra = safe_mean_nodes(protein_res_graph, protein_sub_updated_intra)
        ligand_pool_inter = safe_mean_nodes(ligand_frag_graph, ligand_sub_updated_inter)
        protein_pool_inter = safe_mean_nodes(protein_res_graph, protein_sub_updated_inter)

        # Fusion
        H_gnn = torch.cat([ligand_pool_intra, protein_pool_intra, ligand_pool_inter, protein_pool_inter], dim=1)
        
        # [注意] 使用 L_E_bond_agg 和 P_E_bond_agg 进行融合
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
        
        # 返回 4 个值：亲和力预测, 统计信息, 边分类Logits, 边物理能量
        return y_pred, recon_stats, edge_logits, flat_edge_energies