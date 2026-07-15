import torch
import torch.nn as nn
import torch.nn.functional as F

# 从 channels.py 导入各类 Intra/Inter-type 编码模块
from .channels import (
    LigandAtomChannel, ProteinAtomChannel, InterAtomChannel,
    LigandFragmentChannel, ProteinResidueChannel, InterSubstructureChannel
)
# 从 hil.py 导入 Atom-level 与 Substructure-level Interactive Learning 模块
from .hil import (
    AtomLevelInteractiveLigand, AtomLevelInteractiveProtein,
    SubstructureLevelInteractiveLigand, SubstructureLevelInteractiveProtein
)


class MMDCGDTAModel(nn.Module):
    """
    MMDCG-DTA 模型：
      - 输入：六种图数据（配体原子图、蛋白原子图、原子交互图、配体碎片图、蛋白残基图、子结构交互图）
      - 经过 Atom-level 与 Substructure-level 编码、交互与信息融合，预测 binding affinity.
    """

    def __init__(self, config):
        super(MMDCGDTAModel, self).__init__()
        # 从配置中严格获取各项超参数
        self.l_intra = config["l_intra"]  # Intra-type 编码迭代次数
        self.l_inter = config["l_inter"]  # Inter-type 编码迭代次数
        self.l_atom = config["l_atom"]  # Atom-level interactive learning 迭代次数
        self.l_sub = config["l_sub"]  # Substructure-level interactive learning 迭代次数
        self.d = config["embedding_dim"]  # 嵌入维度 (通常是 128)
        self.neg_slope = config["inter_negative_slope"]  # LeakyReLU 负斜率
        self.d_atom = config["d_atom"]  # 原子交互距离阈值
        self.d_sub = config["d_sub"]  # 子结构交互距离阈值
        # 假设物理化学描述符维度（子结构初始特征 x）为 config["sub_x_dim"]
        self.sub_x_dim = config["sub_x_dim"]

        # ============================================================
        # 【修改 1】特征投影层 (解决维度不匹配的核心)
        # ============================================================
        # 你的报错显示配体原子特征是 5 维 (73x5)，所以这里输入设为 5
        self.ligand_atom_projector = nn.Linear(5, self.d)
        
        # 蛋白原子特征通常是 41 维 (DGL默认)，如果后续蛋白报错，请修改这里的 41 为报错信息中的维度
        self.protein_atom_projector = nn.Linear(5, self.d)

        # ---------------- Atom-level 模块 ----------------
        self.ligand_atom_intra_encoder = LigandAtomChannel(self.l_intra, self.d, self.d, self.neg_slope)
        self.protein_atom_intra_encoder = ProteinAtomChannel(self.l_intra, self.d, self.d, self.neg_slope)
        self.inter_atom_encoder = InterAtomChannel(self.l_inter, self.d_atom, self.d, self.neg_slope)
        self.ligand_atom_interactive = AtomLevelInteractiveLigand(self.l_atom, self.d)
        self.protein_atom_interactive = AtomLevelInteractiveProtein(self.l_atom, self.d)

        # ---------------- Substructure-level 模块 ----------------
        self.ligand_frag_intra_encoder = LigandFragmentChannel(self.l_intra, self.d + self.sub_x_dim,
                                                               self.d + self.sub_x_dim, self.neg_slope)
        self.protein_res_intra_encoder = ProteinResidueChannel(self.l_intra, self.d + self.sub_x_dim,
                                                               self.d + self.sub_x_dim, self.neg_slope)
        self.inter_sub_encoder = InterSubstructureChannel(self.l_inter, self.d_sub, self.d + self.sub_x_dim,
                                                          self.neg_slope)
        self.ligand_sub_interactive = SubstructureLevelInteractiveLigand(self.l_sub, self.d + self.sub_x_dim,
                                                                         self.neg_slope)
        self.protein_sub_interactive = SubstructureLevelInteractiveProtein(self.l_sub, self.d + self.sub_x_dim,
                                                                           self.neg_slope)

        # ---------------- 融合与亲和力预测模块 ----------------
        # 【修改 2】修正融合维度计算
        # H_final 是 [Ligand_Intra, Protein_Intra] -> 2倍
        # Z_final 是 [Ligand_Inter, Protein_Inter] -> 2倍
        # F_final 是 [H_final, Z_final] -> 总共 4 倍
        fusion_dim = 4 * (self.d + self.sub_x_dim)
        
        # GRU 层用于进一步融合序列信息
        self.gru = nn.GRU(input_size=fusion_dim, hidden_size=fusion_dim, batch_first=True)
        # 最后一个全连接层将 GRU 的输出映射到标量预测
        self.pred_fc = nn.Linear(fusion_dim, 1)

    def forward(self, sample):
        """
        sample: 字典，包含图数据
        """
        # -------- Atom-level 编码与交互 --------
        ligand_atom_graph = sample["ligand_atom_graph"]
        protein_atom_graph = sample["protein_atom_graph"]

        # ============================================================
        # 【修改 3】应用投影层，将原始特征 (5维/41维) 映射到 hidden_dim (128维)
        # ============================================================
        h_lig_raw = ligand_atom_graph.ndata["h"]
        h_prot_raw = protein_atom_graph.ndata["h"]
        
        # 映射
        h_lig = self.ligand_atom_projector(h_lig_raw)
        h_prot = self.protein_atom_projector(h_prot_raw)

        # Intra-type 编码：传入映射后的特征 h_lig 和 h_prot
        ligand_intra = self.ligand_atom_intra_encoder(ligand_atom_graph, h_lig)  # [N_lig, d]
        protein_intra = self.protein_atom_intra_encoder(protein_atom_graph, h_prot)  # [N_prot, d]

        # Inter-type 编码：利用原子坐标更新 atom_interaction_graph
        ligand_coords = ligand_atom_graph.ndata["coord"]  # [N_lig, 3]
        protein_coords = protein_atom_graph.ndata["coord"]  # [N_prot, 3]
        inter_lig, inter_prot = self.inter_atom_encoder(ligand_intra, protein_intra, ligand_coords, protein_coords)

        # Atom-level Interactive Learning：分别更新 ligand 和 protein 原子
        ligand_group = ligand_atom_graph.ndata["group"]  # [N_lig]
        protein_group = protein_atom_graph.ndata["group"]  # [N_prot]
        updated_inter_lig, updated_intra_lig = self.ligand_atom_interactive(ligand_intra, inter_lig, ligand_group)
        updated_inter_prot, updated_intra_prot = self.protein_atom_interactive(protein_intra, inter_prot, protein_group)
        
        # 保存 Atom-level 的最终 intra 表示
        H_lig_atom_final = updated_intra_lig  # [N_lig, d]
        H_prot_atom_final = updated_intra_prot  # [N_prot, d]

        # -------- Substructure-level 编码与交互 --------
        # 先更新子结构初始表示：对于每个配体碎片节点
        ligand_frag_graph = sample["ligand_fragment_graph"]
        protein_res_graph = sample["protein_residue_graph"]
        
        # 配体碎片特征更新
        new_ligand_feats = []
        # 注意：如果是 batch 训练，这里循环 num_nodes 可能会慢，但为了保持逻辑一致暂且保留
        # 如果 atom_indices 是 tensor 列表，需要确保数据在同一 device
        for i in range(ligand_frag_graph.num_nodes()):
            x_i = ligand_frag_graph.ndata["x"][i]  # [sub_x_dim]
            indices = ligand_frag_graph.ndata["atom_indices"][i]  # list or tensor
            
            # 兼容 indices 可能为空或 tensor 的情况
            if len(indices) == 0:
                atom_sum = torch.zeros(self.d, device=x_i.device) # 注意维度是 self.d
            else:
                # 这里的 indices 必须是 tensor 类型才能用于索引
                if not isinstance(indices, torch.Tensor):
                    indices = torch.tensor(indices, device=x_i.device)
                atom_sum = H_lig_atom_final[indices].sum(dim=0)  # [d]
            
            new_feat = torch.cat([x_i, atom_sum], dim=0)  # [d + sub_x_dim]
            new_ligand_feats.append(new_feat)
            
        new_ligand_feats = torch.stack(new_ligand_feats, dim=0)
        ligand_frag_graph.ndata["h"] = new_ligand_feats

        # 蛋白残基特征更新
        new_protein_feats = []
        for j in range(protein_res_graph.num_nodes()):
            x_j = protein_res_graph.ndata["x"][j]
            indices = protein_res_graph.ndata["atom_indices"][j]
            
            if len(indices) == 0:
                atom_sum = torch.zeros(self.d, device=x_j.device)
            else:
                if not isinstance(indices, torch.Tensor):
                    indices = torch.tensor(indices, device=x_j.device)
                atom_sum = H_prot_atom_final[indices].sum(dim=0)
                
            new_feat = torch.cat([x_j, atom_sum], dim=0)
            new_protein_feats.append(new_feat)
            
        new_protein_feats = torch.stack(new_protein_feats, dim=0)
        protein_res_graph.ndata["h"] = new_protein_feats

        # Intra-type 编码：对子结构图分别编码
        ligand_intra_sub = self.ligand_frag_intra_encoder(ligand_frag_graph, ligand_frag_graph.ndata["h"])
        protein_intra_sub = self.protein_res_intra_encoder(protein_res_graph, protein_res_graph.ndata["h"])

        # Inter-type 编码：利用子结构节点坐标更新子结构交互图
        inter_sub_graph = sample["substructure_interaction_graph"]
        inter_sub_input = torch.cat([ligand_intra_sub, protein_intra_sub], dim=0)
        inter_sub_graph.ndata["h"] = inter_sub_input
        
        ligand_sub_coords = ligand_frag_graph.ndata["coord"]
        protein_sub_coords = protein_res_graph.ndata["coord"]
        inter_lig_sub, inter_prot_sub = self.inter_sub_encoder(
            ligand_intra_sub, protein_intra_sub, ligand_sub_coords, protein_sub_coords
        )

        # Substructure-level Interactive Learning
        ligand_sub_updated_inter, ligand_sub_updated_intra = self.ligand_sub_interactive(ligand_intra_sub, inter_lig_sub)
        protein_sub_updated_inter, protein_sub_updated_intra = self.protein_sub_interactive(protein_intra_sub, inter_prot_sub)

        # -------- 融合与预测 --------
        # 池化
        ligand_pool_intra = ligand_sub_updated_intra.mean(dim=0)  # [d + sub_x_dim]
        protein_pool_intra = protein_sub_updated_intra.mean(dim=0)  # [d + sub_x_dim]
        # 拼接 Intra 部分
        H_final = torch.cat([ligand_pool_intra, protein_pool_intra], dim=0)  # [2*(d+sub_x_dim)]

        ligand_pool_inter = ligand_sub_updated_inter.mean(dim=0)
        protein_pool_inter = protein_sub_updated_inter.mean(dim=0)
        # 拼接 Inter 部分
        Z_final = torch.cat([ligand_pool_inter, protein_pool_inter], dim=0)  # [2*(d+sub_x_dim)]

        # 最终融合：将 H_final 和 Z_final 拼接
        # 结果维度：2*(...) + 2*(...) = 4*(d+sub_x_dim)
        F_final = torch.cat([H_final, Z_final], dim=0)  

        # 增加 batch 和 sequence 维度 [1, 1, fusion_dim]
        F_final = F_final.unsqueeze(0).unsqueeze(0) 

        # 利用 GRU 进行进一步融合
        gru_out, _ = self.gru(F_final)
        fusion_rep = gru_out.squeeze(0).squeeze(0)

        # 最后预测亲和力
        y_pred = self.pred_fc(fusion_rep)  # [1]

        return y_pred
