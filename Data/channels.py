import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from mechanics import InteractionForceMLP 

# =============================================================================
# 1. GAT Layer (修改：支持 edge_weights)
# =============================================================================
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=0, negative_slope=0.2):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        
        self.W = nn.Linear(in_dim, out_dim, bias=False) 
        
        if edge_dim > 0:
            self.W_e = nn.Linear(edge_dim, out_dim, bias=False)
            self.attn_dim = 3 * out_dim
        else:
            self.W_e = None
            self.attn_dim = 2 * out_dim

        self.attn_l = nn.Parameter(torch.FloatTensor(1, out_dim))
        self.attn_r = nn.Parameter(torch.FloatTensor(1, out_dim))
        if edge_dim > 0:
            self.attn_e = nn.Parameter(torch.FloatTensor(1, out_dim))

        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W.weight)
        if self.W_e is not None:
            nn.init.xavier_normal_(self.W_e.weight)
            nn.init.xavier_normal_(self.attn_e)
        nn.init.xavier_normal_(self.attn_l)
        nn.init.xavier_normal_(self.attn_r)
        nn.init.zeros_(self.bias)

    def forward(self, g, h, edge_feats=None, edge_weights=None):
        """
        [修改] 增加 edge_weights 参数，用于接收内部分子力学权重
        """
        feat = self.W(h)
        el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
        
        g.ndata.update({'ft': feat, 'el': el, 'er': er})
        
        if self.edge_dim > 0 and edge_feats is not None:
            feat_e = self.W_e(edge_feats)
            ee = (feat_e * self.attn_e).sum(dim=-1).unsqueeze(-1)
            g.edata['ee'] = ee
            g.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leakyrelu(g.edata.pop('e') + g.edata.pop('ee'))
        else:
            g.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leakyrelu(g.edata.pop('e'))
            
        g.edata['a'] = edge_softmax(g, e)
        
        # [关键修改] 如果传入了物理权重 (Bond Energy)，则对 Attention 进行加权
        if edge_weights is not None:
            g.edata['a'] = g.edata['a'] * edge_weights
            
        g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
        
        rst = g.ndata.pop('ft') + self.bias
        g.ndata.pop('el')
        g.ndata.pop('er')
        return self.leakyrelu(rst)


class IntraChannel(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, negative_slope=0.2, edge_dim=0):
        super(IntraChannel, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(in_dim, hidden_dim, edge_dim, negative_slope))
        for _ in range(num_layers - 1):
            self.layers.append(GATLayer(hidden_dim, hidden_dim, edge_dim, negative_slope))

    def forward(self, g, h, edge_feats=None, edge_weights=None):
        # 将权重传递给每一层 GAT
        for layer in self.layers:
            h = layer(g, h, edge_feats, edge_weights)
        return h


# =============================================================================
# 2. Wrapper Channels (修改：传递 edge_weights)
# =============================================================================

class LigandAtomChannel(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, negative_slope):
        super(LigandAtomChannel, self).__init__()
        self.encoder = IntraChannel(num_layers, in_dim, hidden_dim, negative_slope, edge_dim=0)
    def forward(self, g, h, edge_weights=None): 
        return self.encoder(g, h, edge_weights=edge_weights)

class ProteinAtomChannel(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, negative_slope):
        super(ProteinAtomChannel, self).__init__()
        self.encoder = IntraChannel(num_layers, in_dim, hidden_dim, negative_slope, edge_dim=0)
    def forward(self, g, h, edge_weights=None): 
        return self.encoder(g, h, edge_weights=edge_weights)

class LigandFragmentChannel(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, negative_slope):
        super(LigandFragmentChannel, self).__init__()
        self.encoder = IntraChannel(num_layers, in_dim, hidden_dim, negative_slope, edge_dim=0)
    def forward(self, g, h, edge_weights=None): 
        return self.encoder(g, h, edge_weights=edge_weights)

class ProteinResidueChannel(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, negative_slope):
        super(ProteinResidueChannel, self).__init__()
        self.encoder = IntraChannel(num_layers, in_dim, hidden_dim, negative_slope, edge_dim=1)
    def forward(self, g, h, edge_feats, edge_weights=None): 
        return self.encoder(g, h, edge_feats, edge_weights=edge_weights)


# =============================================================================
# 3. Inter Channels (保持之前的分子力学版本)
# =============================================================================

class InterAtomChannel(nn.Module):
    def __init__(self, l_inter, d, negative_slope=0.2):
        super(InterAtomChannel, self).__init__()
        self.l_inter = l_inter
        self.d = d
        self.activation = nn.LeakyReLU(negative_slope)
        self.ligand_linear = nn.Linear(d, d, bias=True)
        self.protein_linear = nn.Linear(d, d, bias=True)
        self.attn_l = nn.Parameter(torch.FloatTensor(1, d))
        self.attn_p = nn.Parameter(torch.FloatTensor(1, d))
        self.attn_e = nn.Parameter(torch.FloatTensor(1, d)) 
        self.dist_linear = nn.Linear(1, d) 
        self.mechanics_mlp = InteractionForceMLP(atom_dim=d, hidden_dim=64)
        self.energy_fusion = nn.Linear(3, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.ligand_linear.weight)
        nn.init.xavier_normal_(self.protein_linear.weight)
        nn.init.xavier_normal_(self.attn_l)
        nn.init.xavier_normal_(self.attn_p)
        nn.init.xavier_normal_(self.attn_e)
        nn.init.xavier_normal_(self.dist_linear.weight)
        nn.init.constant_(self.energy_fusion.bias, 2.0) 

    def forward(self, g, h_l, h_p, edge_weights=None):
        h_all = torch.cat([h_l, h_p], dim=0)
        
        with g.local_scope():
            g.ndata['h'] = h_all
            raw_dists = g.edata['dist'] # [E, 1]
            feat_e = self.dist_linear(raw_dists) # [E, d]
            
            def compute_mechanics_weights(edges):
                vdw, elec, hbond = self.mechanics_mlp(edges.src['h'], edges.dst['h'], edges.data['dist'])
                energies = torch.cat([vdw, elec, hbond], dim=-1)
                raw_w = self.energy_fusion(energies)
                w = torch.sigmoid(raw_w)
                return {'phys_w': w}

            if edge_weights is None:
                g.apply_edges(compute_mechanics_weights)
                final_weights = g.edata.pop('phys_w')
            else:
                if edge_weights.dim() == 1: edge_weights = edge_weights.unsqueeze(-1)
                final_weights = edge_weights

            for _ in range(self.l_inter):
                h_curr = g.ndata['h']
                el = (h_curr * self.attn_l).sum(dim=-1).unsqueeze(-1)
                ep = (h_curr * self.attn_p).sum(dim=-1).unsqueeze(-1)
                ee = (feat_e * self.attn_e).sum(dim=-1).unsqueeze(-1)
                g.ndata.update({'el': el, 'ep': ep})
                g.edata['ee'] = ee
                g.apply_edges(fn.u_add_v('el', 'ep', 'e_nodes'))
                e = self.activation(g.edata.pop('e_nodes') + g.edata['ee'])
                
                a = edge_softmax(g, e)
                a = a * final_weights
                
                # 显式赋值给 edata，防止 update_all 找不到
                g.edata['a'] = a 
                
                g.update_all(fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'h_new'))
                g.ndata['h'] = self.activation(g.ndata['h_new'])
            
            h_final = g.ndata['h']
            n_l = h_l.shape[0]
            
            return h_final[:n_l], h_final[n_l:]


class InterSubstructureChannel(nn.Module):
    def __init__(self, l_inter, d, negative_slope=0.2):
        super(InterSubstructureChannel, self).__init__()
        self.l_inter = l_inter
        self.d = d
        self.activation = nn.LeakyReLU(negative_slope)
        self.ligand_linear = nn.Linear(d, d, bias=True)
        self.protein_linear = nn.Linear(d, d, bias=True)
        self.attn_l = nn.Parameter(torch.FloatTensor(1, d))
        self.attn_p = nn.Parameter(torch.FloatTensor(1, d))
        self.attn_e = nn.Parameter(torch.FloatTensor(1, d)) 
        self.dist_linear = nn.Linear(1, d) 
        self.mechanics_mlp = InteractionForceMLP(atom_dim=d, hidden_dim=64)
        self.energy_fusion = nn.Linear(3, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.ligand_linear.weight)
        nn.init.xavier_normal_(self.protein_linear.weight)
        nn.init.xavier_normal_(self.attn_l)
        nn.init.xavier_normal_(self.attn_p)
        nn.init.xavier_normal_(self.attn_e)
        nn.init.xavier_normal_(self.dist_linear.weight)
        nn.init.constant_(self.energy_fusion.bias, 2.0)

    def forward(self, g, h_l, h_p, edge_weights=None):
        h_all = torch.cat([h_l, h_p], dim=0)
        
        with g.local_scope():
            g.ndata['h'] = h_all
            raw_dists = g.edata['dist']
            feat_e = self.dist_linear(raw_dists)
            
            def compute_mechanics_weights(edges):
                vdw, elec, hbond = self.mechanics_mlp(edges.src['h'], edges.dst['h'], edges.data['dist'])
                energies = torch.cat([vdw, elec, hbond], dim=-1)
                raw_w = self.energy_fusion(energies)
                w = torch.sigmoid(raw_w)
                return {'phys_w': w}

            if edge_weights is None:
                g.apply_edges(compute_mechanics_weights)
                final_weights = g.edata.pop('phys_w')
            else:
                if edge_weights.dim() == 1: edge_weights = edge_weights.unsqueeze(-1)
                final_weights = edge_weights

            for _ in range(self.l_inter):
                h_curr = g.ndata['h']
                el = (h_curr * self.attn_l).sum(dim=-1).unsqueeze(-1)
                ep = (h_curr * self.attn_p).sum(dim=-1).unsqueeze(-1)
                ee = (feat_e * self.attn_e).sum(dim=-1).unsqueeze(-1)
                g.ndata.update({'el': el, 'ep': ep})
                g.edata['ee'] = ee
                g.apply_edges(fn.u_add_v('el', 'ep', 'e_nodes'))
                e = self.activation(g.edata.pop('e_nodes') + g.edata['ee'])
                a = edge_softmax(g, e)
                a = a * final_weights
                
                g.edata['a'] = a
                
                g.update_all(fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'h_new'))
                g.ndata['h'] = self.activation(g.ndata['h_new'])
            
            h_final = g.ndata['h']
            n_l = h_l.shape[0]
            
            return h_final[:n_l], h_final[n_l:]