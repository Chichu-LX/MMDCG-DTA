import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import math

class GATLayer(nn.Module):
    """
    单层 GAT 层的实现，公式如下：

      H_i^(l+1) = σ( ∑_{j∈N(i)} a_{ij} * W1 * H_j^(l) + b )

    注意力系数 a_{ij} 的计算采用：

      e_{ij} = LeakyReLU( a^T [W2 * H_i^(l) || W3 * H_j^(l)] )
      a_{ij} = Softmax_j( e_{ij} )

    其中 || 表示向量拼接，σ 通常使用 LeakyReLU 激活。
    """

    def __init__(self, in_dim, out_dim, negative_slope=0.2):
        super(GATLayer, self).__init__()
        self.W1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W2 = nn.Linear(in_dim, out_dim, bias=False)
        self.W3 = nn.Linear(in_dim, out_dim, bias=False)
        # 注意力参数，维度为 2*out_dim
        self.attn = nn.Parameter(torch.FloatTensor(2 * out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W1.weight)
        nn.init.xavier_normal_(self.W2.weight)
        nn.init.xavier_normal_(self.W3.weight)
        nn.init.xavier_normal_(self.attn.unsqueeze(0))
        nn.init.zeros_(self.bias)

    def edge_attention(self, edges):
        # edges.src['h_att']来自 W2(H_i), edges.dst['h_att']来自 W3(H_j)
        cat = torch.cat([edges.src['h_att'], edges.dst['h_att']], dim=1)
        # 计算 e_ij = LeakyReLU( a^T cat )
        e = self.leakyrelu((cat * self.attn).sum(dim=1))
        return {'e': e}

    def message_func(self, edges):
        # 将变换后的源节点特征乘以注意力系数
        return {'m': edges.src['h_trans'] * edges.data['a'].unsqueeze(1)}

    def forward(self, g, h):
        # h: [N, in_dim]
        # 1. 对源节点做线性变换用于消息传递
        h_trans = self.W1(h)  # [N, out_dim]
        # 2. 分别对源和目标节点计算注意力用的表示
        h_att_src = self.W2(h)  # [N, out_dim]
        h_att_dst = self.W3(h)  # [N, out_dim]
        g.ndata['h_att'] = h_att_src
        g.ndata['h_trans'] = h_trans
        g.ndata['h_att_dst'] = h_att_dst  # 将目标节点的表示存储下来

        # 3. 计算边注意力分数
        g.apply_edges(self.edge_attention)
        # 4. 对每个目标节点做 softmax
        e = g.edata.pop('e')
        a = edge_softmax(g, e)
        g.edata['a'] = a

        # 5. 消息传递聚合
        g.update_all(self.message_func, fn.sum(msg='m', out='h_new'))
        h_new = g.ndata.pop('h_new') + self.bias
        # 6. 激活函数
        return self.leakyrelu(h_new)


class IntraChannel(nn.Module):
    """
    Intra-type Channel 模块，用于堆叠多层 GATLayer。

    参数:
      num_layers: 层数（l_intra）
      in_dim: 输入特征维度
      hidden_dim: 每层输出维度（最终输出维度为 hidden_dim）
      negative_slope: LeakyReLU 的负斜率
    """

    def __init__(self, num_layers, in_dim, hidden_dim, negative_slope=0.2):
        super(IntraChannel, self).__init__()
        self.layers = nn.ModuleList()
        # 第一层从 in_dim 到 hidden_dim
        self.layers.append(GATLayer(in_dim, hidden_dim, negative_slope))
        # 后续层保持 hidden_dim
        for _ in range(num_layers - 1):
            self.layers.append(GATLayer(hidden_dim, hidden_dim, negative_slope))

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h)
        return h


# 分别为配体原子图、蛋白原子图、配体碎片图、蛋白残基图构建独立的编码器模块
class LigandAtomChannel(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, negative_slope):
        super(LigandAtomChannel, self).__init__()
        self.encoder = IntraChannel(num_layers, in_dim, hidden_dim, negative_slope)

    def forward(self, g, h):
        # g: Ligand Atom Graph, h: 初始原子特征
        return self.encoder(g, h)


class ProteinAtomChannel(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, negative_slope):
        super(ProteinAtomChannel, self).__init__()
        self.encoder = IntraChannel(num_layers, in_dim, hidden_dim, negative_slope)

    def forward(self, g, h):
        # g: Protein Atom Graph, h: 初始原子特征
        return self.encoder(g, h)


class LigandFragmentChannel(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, negative_slope):
        super(LigandFragmentChannel, self).__init__()
        self.encoder = IntraChannel(num_layers, in_dim, hidden_dim, negative_slope)

    def forward(self, g, h):
        # g: Ligand Fragment Graph, h: 初始子结构特征（已融合了物理化学描述符与原子级信息）
        return self.encoder(g, h)


class ProteinResidueChannel(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, negative_slope):
        super(ProteinResidueChannel, self).__init__()
        self.encoder = IntraChannel(num_layers, in_dim, hidden_dim, negative_slope)

    def forward(self, g, h):
        # g: Protein Residue Graph, h: 初始残基特征（已融合了物理化学描述符与原子级信息）
        return self.encoder(g, h)


class InterAtomChannel(nn.Module):
    """
    Inter-atom Channel 模块：对原子交互图进行迭代更新。
    输入：
      - ligand_feats: [N_L, d]，配体原子初始表示（从 intra-channel 得到）
      - protein_feats: [N_P, d]，蛋白原子初始表示
      - ligand_coords: [N_L, 3]，配体原子的空间坐标
      - protein_coords: [N_P, 3]，蛋白原子的空间坐标
    超参数：
      - d_th: 距离阈值（d_atom），严格从 yaml 中读取
      - l_inter: 迭代层数
      - d: 表示维度
      - activation: 激活函数（例如 LeakyReLU）
    输出：
      - 更新后的 ligand_feats 和 protein_feats（分别形状 [N_L, d] 和 [N_P, d]）
    """

    def __init__(self, l_inter, d_th, d, negative_slope=0.2):
        super(InterAtomChannel, self).__init__()
        self.l_inter = l_inter
        self.d_th = d_th
        self.d = d
        self.activation = nn.LeakyReLU(negative_slope)
        # 定义更新权重（分别用于 ligand 和 protein 更新），均为线性层（无偏置，可加偏置也可）
        self.ligand_linear = nn.Linear(d, d, bias=True)
        self.protein_linear = nn.Linear(d, d, bias=True)
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.ligand_linear.weight)
        nn.init.xavier_normal_(self.protein_linear.weight)
        if self.ligand_linear.bias is not None:
            nn.init.zeros_(self.ligand_linear.bias)
        if self.protein_linear.bias is not None:
            nn.init.zeros_(self.protein_linear.bias)

    def forward(self, ligand_feats, protein_feats, ligand_coords, protein_coords):
        """
        ligand_feats: [N_L, d]
        protein_feats: [N_P, d]
        ligand_coords: [N_L, 3]
        protein_coords: [N_P, 3]
        """
        # 计算配体与蛋白之间的距离矩阵 [N_L, N_P]
        # 利用广播计算欧氏距离
        diff = ligand_coords.unsqueeze(1) - protein_coords.unsqueeze(0)  # [N_L, N_P, 3]
        dist_matrix = torch.norm(diff, dim=-1)  # [N_L, N_P]

        # 计算权重矩阵 D^{atom}：
        # 对每个 ligand 原子 i 和蛋白原子 j，如果 dist <= d_th，则 weight = exp(d_th - dist)
        # 否则设为 0
        mask = (dist_matrix <= self.d_th).float()  # [N_L, N_P]
        exp_term = torch.exp(self.d_th - dist_matrix) * mask  # [N_L, N_P]
        # 对每个 ligand 原子做 softmax over protein atoms
        weight_matrix = exp_term / (exp_term.sum(dim=1, keepdim=True) + 1e-8)  # [N_L, N_P]

        # 迭代更新
        Z_L = ligand_feats  # [N_L, d]
        Z_P = protein_feats  # [N_P, d]

        for _ in range(self.l_inter):
            # 对于 ligand：每个 ligand 节点 i 收集所有蛋白节点 j 信息
            # 计算 element-wise 乘积：需要扩展维度
            # 计算 shape: [N_L, N_P, d]
            prod_L = Z_L.unsqueeze(1) * Z_P.unsqueeze(0)
            # 线性变换：reshape为 [N_L*N_P, d] -> apply linear -> reshape回 [N_L, N_P, d]
            prod_L_flat = prod_L.view(-1, self.d)
            updated_L_flat = self.ligand_linear(prod_L_flat)
            updated_L = updated_L_flat.view(Z_L.size(0), Z_P.size(0), self.d)
            updated_L = self.activation(updated_L)
            # 加权求和 over protein nodes (dim=1)
            Z_L_new = torch.sum(weight_matrix.unsqueeze(-1) * updated_L, dim=1)  # [N_L, d]

            # 对于 protein：每个蛋白节点 j 收集所有 ligand节点 i 信息
            prod_P = Z_P.unsqueeze(0) * Z_L.unsqueeze(1)  # [N_L, N_P, d]
            prod_P_flat = prod_P.view(-1, self.d)
            updated_P_flat = self.protein_linear(prod_P_flat)
            updated_P = updated_P_flat.view(Z_L.size(0), Z_P.size(0), self.d)
            updated_P = self.activation(updated_P)
            # 这里需要对 protein 节点 j 做 softmax over ligand nodes：即转置 weight_matrix
            weight_matrix_T = weight_matrix.transpose(0, 1)  # [N_P, N_L]
            Z_P_new = torch.sum(weight_matrix_T.unsqueeze(-1) * updated_P.transpose(0, 1), dim=1)  # [N_P, d]

            # 更新
            Z_L, Z_P = Z_L_new, Z_P_new

        return Z_L, Z_P


class InterSubstructureChannel(nn.Module):
    """
    Inter-substructure Channel 模块：用于子结构交互图（配体碎片和蛋白残基）的交互更新。
    输入：
      - ligand_sub_feats: [N_L_sub, d]，配体碎片节点表示
      - protein_sub_feats: [N_P_sub, d]，蛋白残基节点表示
      - ligand_sub_coords: [N_L_sub, 3]，配体碎片的代表坐标（可由各碎片内原子最小距离计算获得）
      - protein_sub_coords: [N_P_sub, 3]，蛋白残基的代表坐标
    超参数：
      - d_th: 距离阈值 (d_sub)
      - l_inter: 迭代层数
      - d: 表示维度
    输出：
      - 更新后的 ligand_sub_feats 和 protein_sub_feats
    """

    def __init__(self, l_inter, d_th, d, negative_slope=0.2):
        super(InterSubstructureChannel, self).__init__()
        self.l_inter = l_inter
        self.d_th = d_th
        self.d = d
        self.activation = nn.LeakyReLU(negative_slope)
        self.ligand_linear = nn.Linear(d, d, bias=True)
        self.protein_linear = nn.Linear(d, d, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.ligand_linear.weight)
        nn.init.xavier_normal_(self.protein_linear.weight)
        if self.ligand_linear.bias is not None:
            nn.init.zeros_(self.ligand_linear.bias)
        if self.protein_linear.bias is not None:
            nn.init.zeros_(self.protein_linear.bias)

    def forward(self, ligand_sub_feats, protein_sub_feats, ligand_sub_coords, protein_sub_coords):
        """
        输入：
          ligand_sub_feats: [N_L_sub, d]
          protein_sub_feats: [N_P_sub, d]
          ligand_sub_coords: [N_L_sub, 3]
          protein_sub_coords: [N_P_sub, 3]
        """
        # 计算距离矩阵 [N_L_sub, N_P_sub]
        diff = ligand_sub_coords.unsqueeze(1) - protein_sub_coords.unsqueeze(0)
        dist_matrix = torch.norm(diff, dim=-1)

        # 计算权重矩阵 D^{sub}
        mask = (dist_matrix <= self.d_th).float()
        exp_term = torch.exp(self.d_th - dist_matrix) * mask
        weight_matrix = exp_term / (exp_term.sum(dim=1, keepdim=True) + 1e-8)

        Z_L = ligand_sub_feats
        Z_P = protein_sub_feats

        for _ in range(self.l_inter):
            # 更新 ligand 子结构节点表示
            prod_L = Z_L.unsqueeze(1) * Z_P.unsqueeze(0)  # [N_L_sub, N_P_sub, d]
            prod_L_flat = prod_L.view(-1, self.d)
            updated_L_flat = self.ligand_linear(prod_L_flat)
            updated_L = updated_L_flat.view(Z_L.size(0), Z_P.size(0), self.d)
            updated_L = self.activation(updated_L)
            Z_L_new = torch.sum(weight_matrix.unsqueeze(-1) * updated_L, dim=1)  # [N_L_sub, d]

            # 更新 protein 子结构节点表示
            prod_P = Z_P.unsqueeze(0) * Z_L.unsqueeze(1)  # [N_L_sub, N_P_sub, d]
            prod_P_flat = prod_P.view(-1, self.d)
            updated_P_flat = self.protein_linear(prod_P_flat)
            updated_P = updated_P_flat.view(Z_L.size(0), Z_P.size(0), self.d)
            updated_P = self.activation(updated_P)
            weight_matrix_T = weight_matrix.transpose(0, 1)
            Z_P_new = torch.sum(weight_matrix_T.unsqueeze(-1) * updated_P.transpose(0, 1), dim=1)  # [N_P_sub, d]

            Z_L, Z_P = Z_L_new, Z_P_new

        return Z_L, Z_P

