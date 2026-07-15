import torch
import torch.nn as nn
import torch.nn.functional as F

# 从 channels.py 导入各类 Intra/Inter-type 编码模块
from channels import (
    LigandAtomChannel, ProteinAtomChannel, InterAtomChannel,
    LigandFragmentChannel, ProteinResidueChannel, InterSubstructureChannel
)
# 从 hil.py 导入 Atom-level 与 Substructure-level Interactive Learning 模块
from hil import (
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
        self.l_atom = config["l_atom"]    # Atom-level interactive learning 迭代次数
        self.l_sub = config["l_sub"]      # Substructure-level interactive learning 迭代次数
        self.d = config["embedding_dim"]  # 嵌入维度
        self.neg_slope = config["inter_negative_slope"]  # LeakyReLU 负斜率
        self.d_atom = config["d_atom"]    # 原子交互距离阈值
        self.d_sub = config["d_sub"]      # 子结构交互距离阈值
        
        # 原始特征维度 (根据 graphs.py 生成的数据)
        self.raw_atom_dim = config.get("raw_atom_dim", 5) 
        self.sub_x_dim = config.get("sub_x_dim", 5)       # 配体碎片特征维度
        self.prot_res_dim = config.get("prot_res_dim", 1) # 蛋白残基特征维度

        # ---------------- Atom-level 模块 ----------------
        # 使用 raw_atom_dim 作为输入维度
        self.ligand_atom_intra_encoder = LigandAtomChannel(self.l_intra, self.raw_atom_dim, self.d, self.neg_slope)
        self.protein_atom_intra_encoder = ProteinAtomChannel(self.l_intra, self.raw_atom_dim, self.d, self.neg_slope)
        self.inter_atom_encoder = InterAtomChannel(self.l_inter, self.d_atom, self.d, self.neg_slope)
        self.ligand_atom_interactive = AtomLevelInteractiveLigand(self.l_atom, self.d)
        self.protein_atom_interactive = AtomLevelInteractiveProtein(self.l_atom, self.d)

        # ---------------- Substructure-level 模块 ----------------
        # 1. 计算原始输入维度 (Raw Dimensions)
        # 碎片: sub_x_dim (5) + 聚合来的原子 embedding (d)
        raw_frag_in_dim = self.sub_x_dim + self.d
        # 残基: prot_res_dim (1) + 聚合来的原子 embedding (d)
        raw_res_in_dim = self.prot_res_dim + self.d

        # 2. 定义统一的目标维度 (Unified Dimension)
        # 我们将两者都投影到 (d + sub_x_dim) = 133 维，以对齐 InterSubstructureChannel
        self.unified_sub_dim = self.d + self.sub_x_dim

        # 3. 定义投影层 (Projection Layers)
        # 将原始维度投影到统一维度
        self.frag_proj = nn.Linear(raw_frag_in_dim, self.unified_sub_dim) 
        self.res_proj = nn.Linear(raw_res_in_dim, self.unified_sub_dim)   

        # 4. 定义 Intra Encoders (关键修改：使用 unified_sub_dim 作为输入)
        # 之前这里使用 raw_frag_in_dim/raw_res_in_dim 导致了维度冲突
        self.ligand_frag_intra_encoder = LigandFragmentChannel(self.l_intra, self.unified_sub_dim, self.unified_sub_dim, self.neg_slope)
        self.protein_res_intra_encoder = ProteinResidueChannel(self.l_intra, self.unified_sub_dim, self.unified_sub_dim, self.neg_slope)
        
        # 5. 定义 Inter Encoder
        # 输入维度为 unified_sub_dim
        self.inter_sub_encoder = InterSubstructureChannel(self.l_inter, self.d_sub, self.unified_sub_dim, self.neg_slope)

        # 6. 定义 Interactive Layers
        self.ligand_sub_interactive = SubstructureLevelInteractiveLigand(self.l_sub, self.unified_sub_dim, self.neg_slope)
        self.protein_sub_interactive = SubstructureLevelInteractiveProtein(self.l_sub, self.unified_sub_dim, self.neg_slope)

        # ---------------- 融合与预测 ----------------
        # 融合后的维度
        fusion_dim = 4 * self.unified_sub_dim
        self.gru = nn.GRU(input_size=fusion_dim, hidden_size=fusion_dim, batch_first=True)
        self.pred_fc = nn.Linear(fusion_dim, 1)

    def _infer_group_assignment(self, atom_pos, num_groups, device):
        """
        [新增工具函数] 当数据中缺少原子-碎片映射时，
        使用简单的 K-Means 聚类推断原子归属和碎片中心。
        """
        if num_groups <= 0: # 保护
            return torch.zeros(atom_pos.size(0), dtype=torch.long, device=device), atom_pos.mean(0, keepdim=True)
            
        # 简单初始化：均匀切分（比随机更稳定）
        step = max(1, atom_pos.size(0) // num_groups)
        centers = atom_pos[::step][:num_groups]
        if centers.size(0) < num_groups: # 补齐
             centers = torch.cat([centers, atom_pos[:num_groups-centers.size(0)]], dim=0)

        # 简单 K-Means (迭代 5 次)
        for _ in range(5):
            # 计算距离 [N_atom, N_group]
            dists = torch.cdist(atom_pos, centers)
            # E-step: 分配
            group_assign = torch.argmin(dists, dim=1)
            # M-step: 更新中心
            new_centers = []
            for i in range(num_groups):
                mask = (group_assign == i)
                if mask.sum() > 0:
                    new_centers.append(atom_pos[mask].mean(dim=0))
                else:
                    new_centers.append(centers[i]) # 保持原样
            centers = torch.stack(new_centers, dim=0)
            
        return group_assign, centers

    def forward(self, sample):
        device = sample["ligand_atom_graph"].device
        
        # 1. 提取图和特征 (注意: graphs.py 使用 keys: 'h', 'pos', 'dist')
        ligand_atom_graph = sample["ligand_atom_graph"]
        protein_atom_graph = sample["protein_atom_graph"]
        
        # Intra 编码
        # 读取 'h' 并传入
        ligand_intra = self.ligand_atom_intra_encoder(ligand_atom_graph, ligand_atom_graph.ndata["h"])
        protein_intra = self.protein_atom_intra_encoder(protein_atom_graph, protein_atom_graph.ndata["h"])

        # 2. 获取坐标 (graphs.py 使用 'pos')
        ligand_coords = ligand_atom_graph.ndata["pos"]
        protein_coords = protein_atom_graph.ndata["pos"]
        
        # Inter 编码
        inter_lig, inter_prot = self.inter_atom_encoder(ligand_intra, protein_intra, ligand_coords, protein_coords)

        # 3. 处理 Group Assignment (Atom -> Frag/Residue)
        # 检查是否已有 mapping，如果没有则动态计算
        if "group" in ligand_atom_graph.ndata:
            ligand_group = ligand_atom_graph.ndata["group"]
            ligand_frag_coords = sample["ligand_fragment_graph"].ndata.get("pos", None) # 尝试获取碎片坐标
        else:
            # 动态推断: 使用 K-Means 将原子聚类到 Fragment 数量的组中
            num_frags = sample["ligand_fragment_graph"].num_nodes()
            ligand_group, ligand_frag_coords = self._infer_group_assignment(ligand_coords, num_frags, device)
            # 将推断出的坐标赋予图，供后续使用
            sample["ligand_fragment_graph"].ndata["pos"] = ligand_frag_coords

        if "group" in protein_atom_graph.ndata:
            protein_group = protein_atom_graph.ndata["group"]
            protein_res_coords = sample["protein_residue_graph"].ndata["pos"]
        else:
            # 动态推断: Protein 类似 (虽然通常 loader 会处理，但为了鲁棒性)
            num_res = sample["protein_residue_graph"].num_nodes()
            protein_group, protein_res_coords = self._infer_group_assignment(protein_coords, num_res, device)
            # 注意: protein_residue_graph 在 graphs.py 中实际上是有 pos 的，这里主要是为了生成 group 映射

        # Atom Interactive
        updated_inter_lig, updated_intra_lig = self.ligand_atom_interactive(ligand_intra, inter_lig, ligand_group)
        updated_inter_prot, updated_intra_prot = self.protein_atom_interactive(protein_intra, inter_prot, protein_group)
        
        H_lig_atom_final = updated_intra_lig
        H_prot_atom_final = updated_intra_prot

        # 4. Substructure Level 聚合
        ligand_frag_graph = sample["ligand_fragment_graph"]
        protein_res_graph = sample["protein_residue_graph"]

        # 聚合 Ligand Atom -> Frag
        new_ligand_feats = []
        for i in range(ligand_frag_graph.num_nodes()):
            x_i = ligand_frag_graph.ndata["h"][i] # 使用 'h'
            # 找出属于该组的原子索引
            indices = (ligand_group == i).nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                atom_sum = torch.zeros(self.d, device=device)
            else:
                atom_sum = H_lig_atom_final[indices].sum(dim=0)
            new_feat = torch.cat([x_i, atom_sum], dim=0)
            new_ligand_feats.append(new_feat)
        new_ligand_feats = torch.stack(new_ligand_feats, dim=0)
        
        # 聚合 Protein Atom -> Residue
        new_protein_feats = []
        for j in range(protein_res_graph.num_nodes()):
            x_j = protein_res_graph.ndata["h"][j]
            indices = (protein_group == j).nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                atom_sum = torch.zeros(self.d, device=device)
            else:
                atom_sum = H_prot_atom_final[indices].sum(dim=0)
            new_feat = torch.cat([x_j, atom_sum], dim=0)
            new_protein_feats.append(new_feat)
        new_protein_feats = torch.stack(new_protein_feats, dim=0)

        # [重要] 投影到统一维度 (Unified Dimension)
        ligand_sub_input = self.frag_proj(new_ligand_feats)   # Shape: [N_frags, unified_sub_dim]
        protein_sub_input = self.res_proj(new_protein_feats)  # Shape: [N_res, unified_sub_dim]

        # Intra Substructure
        # 传递 graph 和 feature (现在 feature 维度已正确对齐)
        ligand_intra_sub = self.ligand_frag_intra_encoder(ligand_frag_graph, ligand_sub_input)
        
        # 传递 edge features (dist) 给 protein residue encoder
        prot_edge_feats = protein_res_graph.edata['dist'] if 'dist' in protein_res_graph.edata else None
        protein_intra_sub = self.protein_res_intra_encoder(protein_res_graph, protein_sub_input, prot_edge_feats)

        # Inter Substructure
        # 使用动态推断或读取的坐标
        inter_lig_sub, inter_prot_sub = self.inter_sub_encoder(
            ligand_intra_sub, protein_intra_sub, ligand_frag_coords, protein_res_coords
        )

        # Sub Interactive
        ligand_sub_updated_inter, ligand_sub_updated_intra = self.ligand_sub_interactive(ligand_intra_sub, inter_lig_sub)
        protein_sub_updated_inter, protein_sub_updated_intra = self.protein_sub_interactive(protein_intra_sub, inter_prot_sub)

        # 融合
        ligand_pool_intra = ligand_sub_updated_intra.mean(dim=0)
        protein_pool_intra = protein_sub_updated_intra.mean(dim=0)
        H_final = torch.cat([ligand_pool_intra, protein_pool_intra], dim=0)

        ligand_pool_inter = ligand_sub_updated_inter.mean(dim=0)
        protein_pool_inter = protein_sub_updated_inter.mean(dim=0)
        Z_final = torch.cat([ligand_pool_inter, protein_pool_inter], dim=0)

        F_final = torch.cat([H_final, Z_final], dim=0)
        F_final = F_final.unsqueeze(0).unsqueeze(0) # [1, 1, dim]

        gru_out, _ = self.gru(F_final)
        fusion_rep = gru_out.squeeze(0).squeeze(0)

        y_pred = self.pred_fc(fusion_rep)
        return y_pred
