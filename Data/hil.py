import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint  # [新增]

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
        # B: [Batch, d], u: [Batch, d]
        g = torch.sigmoid(self.linear_B(B) + self.linear_u(u))
        return (1 - g) * u + g * B


class AtomLevelInteractiveLigand(nn.Module):
    def __init__(self, l_atom, d, use_checkpoint=True): # [新增参数]
        super(AtomLevelInteractiveLigand, self).__init__()
        self.l_atom = l_atom
        self.d = d
        self.use_checkpoint = use_checkpoint # [新增]
        self.linear_msg = nn.Linear(d, d, bias=True)
        self.warp_gate = WarpGate(d)
        self.gru_bridge = nn.GRUCell(d, d)
        self.gru_atom = nn.GRUCell(d, d)

    # 将核心逻辑封装，供 checkpoint 调用
    def _forward_impl(self, H_intra, Z_inter, group_assign):
        # 确保 group_assign 在正确设备
        # 注意：checkpoint 过程中，group_assign 不需要梯度，这符合要求
        unique_groups = torch.unique(group_assign)
        
        bridge = {}
        for g in unique_groups:
            idx = (group_assign == g).nonzero(as_tuple=True)[0]
            bridge[g.item()] = Z_inter[idx].sum(dim=0)

        # Step 1: intra -> inter
        for _ in range(self.l_atom):
            updated_bridge = {}
            all_z_new = []
            all_indices = []

            for g in unique_groups:
                g_val = g.item()
                idx = (group_assign == g).nonzero(as_tuple=True)[0]
                
                # 1. Update Bridge
                B = bridge[g_val] 
                H_group = H_intra[idx]
                
                cos_sim = F.cosine_similarity(H_group, B.unsqueeze(0), dim=1) 
                msg = self.linear_msg(H_group)
                weight = F.softmax(cos_sim, dim=0).unsqueeze(1)
                
                u_a2b = (weight * msg).sum(dim=0)
                u_a2b = F.leaky_relu(u_a2b)
                
                B_new = self.warp_gate(B.unsqueeze(0), u_a2b.unsqueeze(0)).squeeze(0)
                B_new = self.gru_bridge(u_a2b.unsqueeze(0), B_new.unsqueeze(0)).squeeze(0)
                updated_bridge[g_val] = B_new
                
                # 2. Update Atoms
                u_b2a = F.leaky_relu(self.linear_msg(B_new))
                z_group = Z_inter[idx]
                u_b2a_expanded = u_b2a.unsqueeze(0).expand(z_group.size(0), -1)
                
                msg_atom = self.warp_gate(z_group, u_b2a_expanded)
                z_new_group = self.gru_atom(msg_atom, z_group)
                
                all_z_new.append(z_new_group)
                all_indices.append(idx)
            
            bridge = updated_bridge
            
            Z_next = torch.zeros_like(Z_inter)
            if len(all_z_new) > 0:
                flat_z = torch.cat(all_z_new, dim=0)
                flat_idx = torch.cat(all_indices, dim=0)
                Z_next.index_copy_(0, flat_idx, flat_z)
            Z_inter = Z_next

        Z_inter_updated = Z_inter

        # Step 2: inter -> intra
        for _ in range(self.l_atom):
            all_h_new = []
            all_indices = []
            
            for g in unique_groups:
                g_val = g.item()
                idx = (group_assign == g).nonzero(as_tuple=True)[0]
                
                B = Z_inter_updated[idx].sum(dim=0)
                u_b2h = F.leaky_relu(self.linear_msg(B))
                
                h_group = H_intra[idx]
                u_b2h_expanded = u_b2h.unsqueeze(0).expand(h_group.size(0), -1)
                
                h_new_group = self.gru_atom(u_b2h_expanded, h_group)
                
                all_h_new.append(h_new_group)
                all_indices.append(idx)
            
            H_next = torch.zeros_like(H_intra)
            if len(all_h_new) > 0:
                flat_h = torch.cat(all_h_new, dim=0)
                flat_idx = torch.cat(all_indices, dim=0)
                H_next.index_copy_(0, flat_idx, flat_h)
            H_intra = H_next

        return Z_inter_updated, H_intra

    def forward(self, H_intra, Z_inter, group_assign):
        # [修改] 启用 Checkpointing
        if self.use_checkpoint and self.training and H_intra.requires_grad:
            # group_assign 是整数索引，不需要梯度，直接传入即可
            return checkpoint(self._forward_impl, H_intra, Z_inter, group_assign)
        else:
            return self._forward_impl(H_intra, Z_inter, group_assign)


class AtomLevelInteractiveProtein(nn.Module):
    def __init__(self, l_atom, d, use_checkpoint=True):
        super(AtomLevelInteractiveProtein, self).__init__()
        self.l_atom = l_atom
        self.d = d
        self.use_checkpoint = use_checkpoint
        self.linear_msg = nn.Linear(d, d, bias=True)
        self.warp_gate = WarpGate(d)
        self.gru_bridge = nn.GRUCell(d, d)
        self.gru_atom = nn.GRUCell(d, d)

    def _forward_impl(self, H_intra, Z_inter, group_assign):
        # ... (内部逻辑与原代码一致，略去重复以节省空间，结构同 Ligand 类) ...
        # 请确保这里的逻辑与上面 Ligand 类中的 _forward_impl 完全一致（变量名除外）
        # 为完整性，建议直接复制 Ligand 的 _forward_impl 逻辑
        unique_groups = torch.unique(group_assign)
        bridge = {}
        for g in unique_groups:
            idx = (group_assign == g).nonzero(as_tuple=True)[0]
            bridge[g.item()] = Z_inter[idx].sum(dim=0)

        # Step 1
        for _ in range(self.l_atom):
            updated_bridge = {}
            all_z_new = []
            all_indices = []
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
                B_new = self.warp_gate(B.unsqueeze(0), u_a2b.unsqueeze(0)).squeeze(0)
                B_new = self.gru_bridge(u_a2b.unsqueeze(0), B_new.unsqueeze(0)).squeeze(0)
                updated_bridge[g_val] = B_new
                u_b2a = F.leaky_relu(self.linear_msg(B_new))
                z_group = Z_inter[idx]
                u_b2a_expanded = u_b2a.unsqueeze(0).expand(z_group.size(0), -1)
                msg_atom = self.warp_gate(z_group, u_b2a_expanded)
                z_new_group = self.gru_atom(msg_atom, z_group)
                all_z_new.append(z_new_group)
                all_indices.append(idx)
            bridge = updated_bridge
            Z_next = torch.zeros_like(Z_inter)
            if len(all_z_new) > 0:
                flat_z = torch.cat(all_z_new, dim=0)
                flat_idx = torch.cat(all_indices, dim=0)
                Z_next.index_copy_(0, flat_idx, flat_z)
            Z_inter = Z_next
        Z_inter_updated = Z_inter

        # Step 2
        for _ in range(self.l_atom):
            all_h_new = []
            all_indices = []
            for g in unique_groups:
                g_val = g.item()
                idx = (group_assign == g).nonzero(as_tuple=True)[0]
                B = Z_inter_updated[idx].sum(dim=0)
                u_b2h = F.leaky_relu(self.linear_msg(B))
                h_group = H_intra[idx]
                u_b2h_expanded = u_b2h.unsqueeze(0).expand(h_group.size(0), -1)
                h_new_group = self.gru_atom(u_b2h_expanded, h_group)
                all_h_new.append(h_new_group)
                all_indices.append(idx)
            H_next = torch.zeros_like(H_intra)
            if len(all_h_new) > 0:
                flat_h = torch.cat(all_h_new, dim=0)
                flat_idx = torch.cat(all_indices, dim=0)
                H_next.index_copy_(0, flat_idx, flat_h)
            H_intra = H_next
        return Z_inter_updated, H_intra

    def forward(self, H_intra, Z_inter, group_assign):
        if self.use_checkpoint and self.training and H_intra.requires_grad:
            return checkpoint(self._forward_impl, H_intra, Z_inter, group_assign)
        else:
            return self._forward_impl(H_intra, Z_inter, group_assign)


class SubstructureLevelInteractiveLigand(nn.Module):
    def __init__(self, l_sub, d, negative_slope=0.2, use_checkpoint=True):
        super(SubstructureLevelInteractiveLigand, self).__init__()
        self.l_sub = l_sub
        self.d = d
        self.use_checkpoint = use_checkpoint
        self.activation = nn.LeakyReLU(negative_slope)
        self.linear_msg = nn.Linear(d, d, bias=True)
        self.warp_gate = WarpGate(d)
        self.gru_bridge = nn.GRUCell(d, d)
        self.gru_node = nn.GRUCell(d, d)

    def _forward_impl(self, H_intra, Z_inter):
        # Global bridge logic
        N, d = H_intra.size()
        global_bridge = Z_inter.sum(dim=0) 

        # Step 1: intra -> inter
        for _ in range(self.l_sub):
            cos_sim = F.cosine_similarity(H_intra, global_bridge.unsqueeze(0), dim=1)
            msg = self.linear_msg(H_intra)
            weight = F.softmax(cos_sim, dim=0).unsqueeze(1)
            u_a2b = (weight * msg).sum(dim=0)
            u_a2b = F.leaky_relu(u_a2b)
            
            B_new = self.warp_gate(global_bridge.unsqueeze(0), u_a2b.unsqueeze(0)).squeeze(0)
            global_bridge = self.gru_bridge(u_a2b.unsqueeze(0), B_new.unsqueeze(0)).squeeze(0)
            
            u_b2a = F.leaky_relu(self.linear_msg(global_bridge))
            u_b2a_expanded = u_b2a.unsqueeze(0).expand(N, -1)
            
            msg_node = self.warp_gate(Z_inter, u_b2a_expanded)
            Z_inter = self.gru_node(msg_node, Z_inter)

        Z_inter_updated = Z_inter

        # Step 2: inter -> intra
        for _ in range(self.l_sub):
            u_b2h = F.leaky_relu(self.linear_msg(global_bridge))
            u_b2h_expanded = u_b2h.unsqueeze(0).expand(N, -1)
            H_intra = self.gru_node(u_b2h_expanded, H_intra)

        return Z_inter_updated, H_intra

    def forward(self, H_intra, Z_inter):
        if self.use_checkpoint and self.training and H_intra.requires_grad:
            return checkpoint(self._forward_impl, H_intra, Z_inter)
        else:
            return self._forward_impl(H_intra, Z_inter)


class SubstructureLevelInteractiveProtein(nn.Module):
    def __init__(self, l_sub, d, negative_slope=0.2, use_checkpoint=True):
        super(SubstructureLevelInteractiveProtein, self).__init__()
        self.l_sub = l_sub
        self.d = d
        self.use_checkpoint = use_checkpoint
        self.activation = nn.LeakyReLU(negative_slope)
        self.linear_msg = nn.Linear(d, d, bias=True)
        self.warp_gate = WarpGate(d)
        self.gru_bridge = nn.GRUCell(d, d)
        self.gru_node = nn.GRUCell(d, d)

    def _forward_impl(self, H_intra, Z_inter):
        N, d = H_intra.size()
        global_bridge = Z_inter.sum(dim=0)

        # Step 1
        for _ in range(self.l_sub):
            cos_sim = F.cosine_similarity(H_intra, global_bridge.unsqueeze(0), dim=1)
            msg = self.linear_msg(H_intra)
            weight = F.softmax(cos_sim, dim=0).unsqueeze(1)
            u_a2b = (weight * msg).sum(dim=0)
            u_a2b = F.leaky_relu(u_a2b)
            B_new = self.warp_gate(global_bridge.unsqueeze(0), u_a2b.unsqueeze(0)).squeeze(0)
            global_bridge = self.gru_bridge(u_a2b.unsqueeze(0), B_new.unsqueeze(0)).squeeze(0)
            u_b2a = F.leaky_relu(self.linear_msg(global_bridge))
            u_b2a_expanded = u_b2a.unsqueeze(0).expand(N, -1)
            msg_node = self.warp_gate(Z_inter, u_b2a_expanded)
            Z_inter = self.gru_node(msg_node, Z_inter)
        Z_inter_updated = Z_inter

        # Step 2
        for _ in range(self.l_sub):
            u_b2h = F.leaky_relu(self.linear_msg(global_bridge))
            u_b2h_expanded = u_b2h.unsqueeze(0).expand(N, -1)
            H_intra = self.gru_node(u_b2h_expanded, H_intra)

        return Z_inter_updated, H_intra

    def forward(self, H_intra, Z_inter):
        if self.use_checkpoint and self.training and H_intra.requires_grad:
            return checkpoint(self._forward_impl, H_intra, Z_inter)
        else:
            return self._forward_impl(H_intra, Z_inter)