import torch
import torch.nn as nn
import torch.nn.functional as F

class BondEnergyMLP(nn.Module):
    """
    模拟键伸长能 (Bond Stretching Energy)
    输入: 边长 (距离)
    输出: 能量标量
    物理直觉: E = k * (r - r0)^2，MLP将拟合这个曲线
    """
    def __init__(self, hidden_dim=32):
        super(BondEnergyMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(), # SiLU (Swish) 平滑激活函数更适合拟合物理势能
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # 输出能量标量
        )

    def forward(self, dist):
        # dist shape: [E, 1]
        return self.mlp(dist)

class AngleDihedralEnergyMLP(nn.Module):
    """
    模拟键弯折能 (Angle Bending) 和 二面角势能 (Torsion)
    由于原始图数据没有显式的角度/二面角索引，利用GNN聚合后的节点特征来近似。
    节点特征经过图卷积后包含了邻居的几何信息。
    输入: 节点特征
    输出: 该节点相关的局部应变能
    """
    def __init__(self, in_dim, hidden_dim=64):
        super(AngleDihedralEnergyMLP, self).__init__()
        # 分别模拟两个势能
        self.angle_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.torsion_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_h):
        E_angle = self.angle_mlp(node_h)
        E_torsion = self.torsion_mlp(node_h)
        return E_angle, E_torsion

class InteractionForceMLP(nn.Module):
    """
    模拟非共价相互作用: 范德华力 (VDW), 静电 (Electrostatics), 氢键 (Hydrogen Bond)
    输入: 配体节点特征, 蛋白节点特征, 距离
    输出: 三种力的能量分量
    """
    def __init__(self, atom_dim, hidden_dim=64):
        super(InteractionForceMLP, self).__init__()
        self.input_dim = atom_dim * 2 + 1 # h_l + h_p + dist
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 三个独立的头 (Heads)
        self.vdw_head = nn.Linear(hidden_dim, 1)
        self.elec_head = nn.Linear(hidden_dim, 1)
        self.hbond_head = nn.Linear(hidden_dim, 1)

    def forward(self, h_l, h_p, dist):
        # 拼接特征
        cat_feat = torch.cat([h_l, h_p, dist], dim=-1)
        latent = self.shared(cat_feat)
        
        E_vdw = self.vdw_head(latent)
        E_elec = self.elec_head(latent)
        E_hbond = self.hbond_head(latent)
        
        return E_vdw, E_elec, E_hbond