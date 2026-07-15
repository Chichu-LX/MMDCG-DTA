import torch
import torch.nn as nn
import torch.nn.functional as F


class WarpGate(nn.Module):
    """
    实现 warp gate 机制：
      WG(B, u) = (1 - g) ⊙ u + g ⊙ B,
    其中 g = sigmoid(Linear_B(B) + Linear_u(u))
    """

    def __init__(self, d):
        super(WarpGate, self).__init__()
        self.linear_B = nn.Linear(d, d, bias=True)
        self.linear_u = nn.Linear(d, d, bias=True)

    def forward(self, B, u):
        g = torch.sigmoid(self.linear_B(B) + self.linear_u(u))
        return (1 - g) * u + g * B


class AtomLevelInteractiveLigand(nn.Module):
    """
    Atom-level Interactive Learning for ligand atoms.
    输入：
      - H_intra: [N, d]，来自 intra-type 通道的配体原子表示；
      - Z_inter: [N, d]，来自 inter-type 通道的配体原子表示；
      - group_assign: [N]，每个原子所属的组（例如每个配体原子所属的配体碎片ID）。

    更新过程分两步：
      1. intra → inter：利用 intra 表示更新 inter 表示。
      2. inter → intra：利用更新后的 inter 表示反向更新 intra 表示。

    注意：在 inter → intra 阶段，直接使用已更新的 inter 表示作为当前状态。
    输出：
      - 更新后的 inter 表示和 intra 表示，均形状为 [N, d].
    """

    def __init__(self, l_atom, d):
        super(AtomLevelInteractiveLigand, self).__init__()
        self.l_atom = l_atom
        self.d = d
        self.linear_msg = nn.Linear(d, d, bias=True)
        self.warp_gate = WarpGate(d)
        self.gru_bridge = nn.GRUCell(d, d)
        self.gru_atom = nn.GRUCell(d, d)

    def forward(self, H_intra, Z_inter, group_assign):
        # H_intra: [N, d] (intra-type ligand atom representation)
        # Z_inter: [N, d] (inter-type ligand atom representation)
        # group_assign: [N] (每个原子所属组ID)
        unique_groups = torch.unique(group_assign)
        # 初始化局部桥节点：每个组的桥节点初始为该组 inter 表示之和
        bridge = {}
        for g in unique_groups:
            idx = (group_assign == g).nonzero(as_tuple=True)[0]
            bridge[g.item()] = Z_inter[idx].sum(dim=0)

        # Step 1: intra → inter
        for _ in range(self.l_atom):
            updated_bridge = {}
            updated_Z = Z_inter.clone()
            for g in unique_groups:
                g_val = g.item()
                idx = (group_assign == g).nonzero(as_tuple=True)[0]
                B = bridge[g_val]  # 当前组的桥节点表示 [d]
                H_group = H_intra[idx]  # 组内 intra 表示 [n_g, d]
                # 计算余弦相似度作为权重
                cos_sim = F.cosine_similarity(H_group, B.unsqueeze(0), dim=1)  # [n_g]
                msg = self.linear_msg(H_group)  # [n_g, d]
                weight = F.softmax(cos_sim, dim=0).unsqueeze(1)  # [n_g, 1]
                u_a2b = (weight * msg).sum(dim=0)  # [d]
                u_a2b = F.leaky_relu(u_a2b)
                # 更新桥节点
                B_new = self.warp_gate(B, u_a2b)
                B_new = self.gru_bridge(u_a2b.unsqueeze(0), B_new.unsqueeze(0)).squeeze(0)
                updated_bridge[g_val] = B_new
                # 从桥节点传递信息到每个 inter 原子
                u_b2a = F.leaky_relu(self.linear_msg(B_new))
                for i in idx:
                    z = Z_inter[i]
                    msg_atom = self.warp_gate(z, u_b2a)
                    z_new = self.gru_atom(msg_atom.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
                    updated_Z[i] = z_new
            bridge = updated_bridge
            Z_inter = updated_Z
        Z_inter_updated = Z_inter

        # Step 2: inter → intra
        updated_H = H_intra.clone()
        for _ in range(self.l_atom):
            for g in unique_groups:
                g_val = g.item()
                idx = (group_assign == g).nonzero(as_tuple=True)[0]
                # 利用已更新的 inter 表示（当前状态）作为桥节点信息
                B = Z_inter_updated[idx].sum(dim=0)
                u_b2h = F.leaky_relu(self.linear_msg(B))
                for i in idx:
                    h = H_intra[i]
                    h_new = self.gru_atom(u_b2h.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
                    updated_H[i] = h_new
            H_intra = updated_H
        H_intra_updated = H_intra

        return Z_inter_updated, H_intra_updated


class AtomLevelInteractiveProtein(nn.Module):
    """
    Atom-level Interactive Learning for protein atoms.
    输入参数与 Ligand 版本类似，group_assign 表示每个蛋白原子所属的残基ID。
    输出更新后的 inter 和 intra 表示。
    """

    def __init__(self, l_atom, d):
        super(AtomLevelInteractiveProtein, self).__init__()
        self.l_atom = l_atom
        self.d = d
        self.linear_msg = nn.Linear(d, d, bias=True)
        self.warp_gate = WarpGate(d)
        self.gru_bridge = nn.GRUCell(d, d)
        self.gru_atom = nn.GRUCell(d, d)

    def forward(self, H_intra, Z_inter, group_assign):
        # H_intra: [N, d] (intra-type protein atom representation)
        # Z_inter: [N, d] (inter-type protein atom representation)
        # group_assign: [N] (每个蛋白原子所属残基ID)
        unique_groups = torch.unique(group_assign)
        bridge = {}
        for g in unique_groups:
            idx = (group_assign == g).nonzero(as_tuple=True)[0]
            bridge[g.item()] = Z_inter[idx].sum(dim=0)

        # Step 1: intra → inter
        for _ in range(self.l_atom):
            updated_bridge = {}
            updated_Z = Z_inter.clone()
            for g in unique_groups:
                g_val = g.item()
                idx = (group_assign == g).nonzero(as_tuple=True)[0]
                B = bridge[g_val]
                H_group = H_intra[idx]
                cos_sim = F.cosine_similarity(H_group, B.unsqueeze(0), dim=1)
                msg = self.linear_msg(H_group)
                weight = F.softmax(cos_sim, dim=0).unsqueeze(1)
                u_a2b = (weight * msg).sum(dim=0)
                u_a2b = F.leaky_relu(u_a2b)
                B_new = self.warp_gate(B, u_a2b)
                B_new = self.gru_bridge(u_a2b.unsqueeze(0), B_new.unsqueeze(0)).squeeze(0)
                updated_bridge[g_val] = B_new
                u_b2a = F.leaky_relu(self.linear_msg(B_new))
                for i in idx:
                    z = Z_inter[i]
                    msg_atom = self.warp_gate(z, u_b2a)
                    z_new = self.gru_atom(msg_atom.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
                    updated_Z[i] = z_new
            bridge = updated_bridge
            Z_inter = updated_Z
        Z_inter_updated = Z_inter

        # Step 2: inter → intra
        updated_H = H_intra.clone()
        for _ in range(self.l_atom):
            for g in unique_groups:
                g_val = g.item()
                idx = (group_assign == g).nonzero(as_tuple=True)[0]
                B = Z_inter_updated[idx].sum(dim=0)
                u_b2h = F.leaky_relu(self.linear_msg(B))
                for i in idx:
                    h = H_intra[i]
                    h_new = self.gru_atom(u_b2h.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
                    updated_H[i] = h_new
            H_intra = updated_H
        H_intra_updated = H_intra

        return Z_inter_updated, H_intra_updated


class SubstructureLevelInteractiveLigand(nn.Module):
    """
    Substructure-level Interactive Learning for ligand fragments.

    输入：
      - H_intra: [N, d]，来自 intra-type 通道的 ligand fragment 表示
      - Z_inter: [N, d]，来自 inter-type 通道的 ligand fragment 表示
    更新流程分两步：
      1. intra → inter：利用 intra 表示更新 inter 表示，过程依赖全局桥节点（对所有 ligand fragments 求和初始化）。
      2. inter → intra：利用更新后的 inter 表示反向更新 intra 表示（直接基于当前 inter 表示）。

    输出：
      - 更新后的 inter 表示和 intra 表示，均形状为 [N, d].
    """

    def __init__(self, l_sub, d, negative_slope=0.2):
        super(SubstructureLevelInteractiveLigand, self).__init__()
        self.l_sub = l_sub
        self.d = d
        self.activation = nn.LeakyReLU(negative_slope)
        # 用于消息传递（用于桥节点和子结构节点更新）的线性变换
        self.linear_msg = nn.Linear(d, d, bias=True)
        # Warp gate 机制
        self.warp_gate = WarpGate(d)
        # GRUCell 用于更新全局桥节点
        self.gru_bridge = nn.GRUCell(d, d)
        # GRUCell 用于更新子结构节点
        self.gru_node = nn.GRUCell(d, d)

    def forward(self, H_intra, Z_inter):
        """
        参数：
          H_intra: [N, d]，intra-type 子结构（ligand fragment）表示
          Z_inter: [N, d]，inter-type 子结构（ligand fragment）表示
        """
        N, d = H_intra.size()
        # 初始化全局桥节点：对所有 inter 表示求和
        global_bridge = Z_inter.sum(dim=0)  # [d]

        # Step 1: intra → inter
        for _ in range(self.l_sub):
            # 聚合所有子结构的 intra 信息
            cos_sim = F.cosine_similarity(H_intra, global_bridge.unsqueeze(0), dim=1)  # [N]
            msg = self.linear_msg(H_intra)  # [N, d]
            weight = F.softmax(cos_sim, dim=0).unsqueeze(1)  # [N, 1]
            u_a2b = (weight * msg).sum(dim=0)  # [d]
            u_a2b = F.leaky_relu(u_a2b)
            # 更新全局桥节点
            global_bridge_new = self.warp_gate(global_bridge, u_a2b)
            global_bridge_new = self.gru_bridge(u_a2b.unsqueeze(0), global_bridge_new.unsqueeze(0)).squeeze(0)
            global_bridge = global_bridge_new
            # 将全局桥节点信息传递给每个 inter 子结构节点
            u_b2a = F.leaky_relu(self.linear_msg(global_bridge))
            updated_Z = []
            for i in range(N):
                z = Z_inter[i]
                msg_node = self.warp_gate(z, u_b2a)
                z_new = self.gru_node(msg_node.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
                updated_Z.append(z_new)
            Z_inter = torch.stack(updated_Z, dim=0)
        Z_inter_updated = Z_inter

        # Step 2: inter → intra
        updated_H = H_intra.clone()
        for _ in range(self.l_sub):
            u_b2h = F.leaky_relu(self.linear_msg(global_bridge))
            updated_nodes = []
            for i in range(N):
                h = H_intra[i]
                h_new = self.gru_node(u_b2h.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
                updated_nodes.append(h_new)
            H_intra = torch.stack(updated_nodes, dim=0)
        H_intra_updated = H_intra

        return Z_inter_updated, H_intra_updated


class SubstructureLevelInteractiveProtein(nn.Module):
    """
    Substructure-level Interactive Learning for protein residues.

    输入：
      - H_intra: [N, d]，来自 intra-type 通道的 protein residue 表示
      - Z_inter: [N, d]，来自 inter-type 通道的 protein residue 表示
    过程与 ligand 版本类似，使用全局桥节点来实现信息聚合和双向传递。

    输出：
      - 更新后的 inter 表示和 intra 表示，均形状为 [N, d].
    """

    def __init__(self, l_sub, d, negative_slope=0.2):
        super(SubstructureLevelInteractiveProtein, self).__init__()
        self.l_sub = l_sub
        self.d = d
        self.activation = nn.LeakyReLU(negative_slope)
        self.linear_msg = nn.Linear(d, d, bias=True)
        self.warp_gate = WarpGate(d)
        self.gru_bridge = nn.GRUCell(d, d)
        self.gru_node = nn.GRUCell(d, d)

    def forward(self, H_intra, Z_inter):
        N, d = H_intra.size()
        global_bridge = Z_inter.sum(dim=0)  # 初始化全局桥节点 [d]

        # Step 1: intra → inter
        for _ in range(self.l_sub):
            cos_sim = F.cosine_similarity(H_intra, global_bridge.unsqueeze(0), dim=1)  # [N]
            msg = self.linear_msg(H_intra)  # [N, d]
            weight = F.softmax(cos_sim, dim=0).unsqueeze(1)  # [N, 1]
            u_a2b = (weight * msg).sum(dim=0)  # [d]
            u_a2b = F.leaky_relu(u_a2b)
            global_bridge_new = self.warp_gate(global_bridge, u_a2b)
            global_bridge_new = self.gru_bridge(u_a2b.unsqueeze(0), global_bridge_new.unsqueeze(0)).squeeze(0)
            global_bridge = global_bridge_new
            u_b2a = F.leaky_relu(self.linear_msg(global_bridge))
            updated_Z = []
            for i in range(N):
                z = Z_inter[i]
                msg_node = self.warp_gate(z, u_b2a)
                z_new = self.gru_node(msg_node.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
                updated_Z.append(z_new)
            Z_inter = torch.stack(updated_Z, dim=0)
        Z_inter_updated = Z_inter

        # Step 2: inter → intra
        updated_H = H_intra.clone()
        for _ in range(self.l_sub):
            u_b2h = F.leaky_relu(self.linear_msg(global_bridge))
            updated_nodes = []
            for i in range(N):
                h = H_intra[i]
                h_new = self.gru_node(u_b2h.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
                updated_nodes.append(h_new)
            H_intra = torch.stack(updated_nodes, dim=0)
        H_intra_updated = H_intra

        return Z_inter_updated, H_intra_updated

