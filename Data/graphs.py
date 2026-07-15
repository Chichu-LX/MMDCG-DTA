import io
import numpy as np
import torch
import dgl
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from Bio.PDB import PDBParser
from featurize import featurize_atom, featurize_substructure

# =================================================================================
# 辅助函数
# =================================================================================
def check_and_fix_features(features, feature_name="", sample_name=""):
    if isinstance(features, np.ndarray):
        nan_mask = np.isnan(features)
        inf_mask = np.isinf(features)
        if np.any(nan_mask) or np.any(inf_mask):
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    elif torch.is_tensor(features):
        nan_mask = torch.isnan(features)
        inf_mask = torch.isinf(features)
        if torch.any(nan_mask) or torch.any(inf_mask):
            features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    return features

def safe_read_sdf(sdf_content, sample_name=""):
    mol = Chem.MolFromMolBlock(sdf_content, removeHs=False)
    if mol is None:
        mol = Chem.MolFromMolBlock(sdf_content, removeHs=False, sanitize=False)
        if mol is not None:
            try:
                mol.UpdatePropertyCache(strict=False)
                fast_sanitize_ops = (Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES ^ Chem.SANITIZE_CLEANUP)
                Chem.SanitizeMol(mol, sanitizeOps=fast_sanitize_ops)
                Chem.GetSymmSSSR(mol)
                try: AllChem.ComputeGasteigerCharges(mol)
                except: pass
            except: return None
    return mol

# =================================================================================
# 1. 配体原子图 (Ligand Atom Graph) - [保持原逻辑，确保生成 pos]
# =================================================================================
def build_ligand_atom_graph(ligand_sdf_content, sample_name=""):
    mol = safe_read_sdf(ligand_sdf_content, sample_name)
    if mol is None: return dgl.graph(([], []), num_nodes=0)

    try: AllChem.ComputeGasteigerCharges(mol)
    except: pass
    
    if mol.GetNumConformers() == 0:
        try: AllChem.EmbedMolecule(mol, randomSeed=42)
        except: return dgl.graph(([], []), num_nodes=0)

    conf = mol.GetConformer()
    atom_features = []
    atom_positions = []

    for i, atom in enumerate(mol.GetAtoms()):
        atom_features.append(featurize_atom(atom))
        try:
            pos = conf.GetAtomPosition(i)
            atom_positions.append([pos.x, pos.y, pos.z])
        except:
            atom_positions.append([0.0, 0.0, 0.0])

    atom_features = np.array(atom_features, dtype=np.float32)
    atom_positions = np.array(atom_positions, dtype=np.float32)
    
    atom_features = check_and_fix_features(atom_features, "ligand_atom_feat", sample_name)
    atom_positions = check_and_fix_features(atom_positions, "ligand_atom_pos", sample_name)

    src_list, dst_list = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src_list.extend([i, j])
        dst_list.extend([j, i])
    
    if len(src_list) == 0:
        src_list, dst_list = range(mol.GetNumAtoms()), range(mol.GetNumAtoms())

    g = dgl.graph((src_list, dst_list), num_nodes=mol.GetNumAtoms())
    g.ndata['h'] = torch.tensor(atom_features)
    g.ndata['pos'] = torch.tensor(atom_positions) # 关键：存储坐标
    return g

# =================================================================================
# 2. 蛋白原子图 (Protein Atom Graph) - [保持原逻辑，确保生成 pos]
# =================================================================================
def build_protein_atom_graph(protein_pdb_content, sample_name=""):
    mol = Chem.MolFromPDBBlock(protein_pdb_content, removeHs=False, sanitize=False)
    if mol is None: return dgl.graph(([], []), num_nodes=0)
    
    try: mol.UpdatePropertyCache(strict=False)
    except: pass

    conf = mol.GetConformer()
    atom_features = []
    atom_positions = []

    for i, atom in enumerate(mol.GetAtoms()):
        atom_features.append(featurize_atom(atom))
        try:
            pos = conf.GetAtomPosition(i)
            atom_positions.append([pos.x, pos.y, pos.z])
        except:
            atom_positions.append([0.0, 0.0, 0.0])

    atom_features = np.array(atom_features, dtype=np.float32)
    atom_positions = np.array(atom_positions, dtype=np.float32)
    
    atom_features = check_and_fix_features(atom_features)
    atom_positions = check_and_fix_features(atom_positions)

    src_list, dst_list = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src_list.extend([i, j])
        dst_list.extend([j, i])
        
    if len(src_list) == 0 and mol.GetNumAtoms() > 0:
        src_list, dst_list = range(mol.GetNumAtoms()), range(mol.GetNumAtoms())

    g = dgl.graph((src_list, dst_list), num_nodes=mol.GetNumAtoms())
    g.ndata['h'] = torch.tensor(atom_features)
    g.ndata['pos'] = torch.tensor(atom_positions) # 关键：存储坐标
    return g

# =================================================================================
# 3. 原子交互图 (Atom Interaction Graph) - [修改：基于预构建图构建]
# =================================================================================
def build_atom_interaction_graph(ligand_graph, protein_graph, d_atom, sample_name=""):
    """
    修改策略：不再读取 SDF/PDB 字符串，而是接收已经构建好的 Atom Graph。
    利用图中已有的 ndata['pos'] 计算距离，避免二次解析失败。
    """
    # 1. 检查输入图是否有效
    if ligand_graph.num_nodes() == 0 or protein_graph.num_nodes() == 0:
        return dgl.graph(([], []), num_nodes=0)

    # 2. 获取坐标 (直接从图特征中提取)
    l_pos = ligand_graph.ndata['pos'].numpy()
    p_pos = protein_graph.ndata['pos'].numpy()
    
    n_ligand = l_pos.shape[0]
    n_protein = p_pos.shape[0]

    # 3. 计算距离矩阵 (N_L, N_P)
    # 利用广播机制：(N_L, 1, 3) - (1, N_P, 3)
    diff = l_pos[:, np.newaxis, :] - p_pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)

    # 4. 筛选满足阈值的边
    ligand_indices, protein_indices = np.where(dists <= d_atom)
    valid_dists = dists[ligand_indices, protein_indices]

    # 5. 构建二部图
    # 节点索引：前 N_L 个是配体，后 N_P 个是蛋白
    src_list = []
    dst_list = []
    edge_dists = []

    for k, (l_idx, p_idx) in enumerate(zip(ligand_indices, protein_indices)):
        d_val = valid_dists[k]
        
        # 配体 -> 蛋白 (双向边以便消息传递)
        src_list.append(l_idx)
        dst_list.append(n_ligand + p_idx)
        edge_dists.append(d_val)
        
        # 蛋白 -> 配体
        src_list.append(n_ligand + p_idx)
        dst_list.append(l_idx)
        edge_dists.append(d_val)

    total_nodes = n_ligand + n_protein
    g = dgl.graph((src_list, dst_list), num_nodes=total_nodes)
    
    # 存储边距离特征
    if len(edge_dists) > 0:
        edge_dists = np.array(edge_dists, dtype=np.float32)
        g.edata['dist'] = torch.tensor(edge_dists).unsqueeze(-1)
    else:
        g.edata['dist'] = torch.tensor([], dtype=torch.float32).view(0, 1)

    return g

# =================================================================================
# 辅助逻辑：碎片坐标映射 (不简化逻辑，确保 BRICS 碎片能找到对应的空间位置)
# =================================================================================
def _get_sorted_fragments(mol):
    try:
        frag_smiles_set = BRICS.BRICSDecompose(mol)
        frag_smiles_list = sorted(list(frag_smiles_set))
        fragment_mols = []
        for smi in frag_smiles_list:
            m = Chem.MolFromSmiles(smi)
            if m: fragment_mols.append(m)
        return fragment_mols, frag_smiles_list
    except:
        try: smi = Chem.MolToSmiles(mol, canonical=True)
        except: smi = "C"
        return [mol], [smi]

def _get_fragment_center(mol, fragment_mol):
    """
    计算碎片在原分子中的几何中心。
    通过子结构匹配找到碎片在原分子中的原子索引，然后求平均坐标。
    """
    try:
        conf = mol.GetConformer()
        match = mol.GetSubstructMatch(fragment_mol)
        if match:
            coords = np.array([conf.GetAtomPosition(i) for i in match])
            return np.mean(coords, axis=0)
        else:
            # 如果严格匹配失败（可能是因为 dummy atoms），尝试模糊匹配或返回分子中心
            # 这里为了保证流程，返回整个分子的中心作为妥协，但不中断流程
            all_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            return np.mean(all_coords, axis=0)
    except:
        return np.zeros(3, dtype=np.float32)

# =================================================================================
# 4. 配体碎片图 (Ligand Fragment Graph) - [修改：增加坐标计算]
# =================================================================================
def build_ligand_fragment_graph(ligand_sdf_content, sample_name=""):
    mol = safe_read_sdf(ligand_sdf_content, sample_name)
    if mol is None: return dgl.graph(([], []), num_nodes=0)

    # 1. 切割碎片
    fragments, _ = _get_sorted_fragments(mol)
    
    node_features = []
    node_positions = [] # 新增：存储碎片中心

    for frag in fragments:
        # 特征
        try:
            feat = featurize_substructure(frag, node_type="ligand")
            node_features.append(feat)
        except:
            node_features.append(np.zeros(5, dtype=np.float32))
        
        # 坐标 (新增逻辑)
        center = _get_fragment_center(mol, frag)
        node_positions.append(center)

    if not node_features:
        return dgl.graph(([], []), num_nodes=0)

    node_features = np.array(node_features, dtype=np.float32)
    node_features = check_and_fix_features(node_features)
    node_positions = np.array(node_positions, dtype=np.float32)
    node_positions = check_and_fix_features(node_positions)

    num_fragments = len(fragments)
    # 构建完全图
    src_list, dst_list = [], []
    for i in range(num_fragments):
        for j in range(num_fragments):
            if i != j:
                src_list.extend([i])
                dst_list.extend([j])
    if len(src_list) == 0:
        src_list, dst_list = range(num_fragments), range(num_fragments)

    g = dgl.graph((src_list, dst_list), num_nodes=num_fragments)
    g.ndata['h'] = torch.tensor(node_features)
    g.ndata['pos'] = torch.tensor(node_positions) # 保存坐标供交互图使用

    return g

# =================================================================================
# 5. 蛋白残基图 (Protein Residue Graph) - [修改：确保保存 pos]
# =================================================================================
def build_protein_residue_graph(protein_pdb_content, d_res, sample_name=""):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("x", io.StringIO(protein_pdb_content))
        model = structure[0]
    except: return dgl.graph(([], []), num_nodes=0)

    residues = []
    ca_coords = []
    
    # 提取残基和CA坐标
    for chain in model:
        for res in chain:
            if res.id[0] != " ": continue
            if 'CA' in res:
                try:
                    coord = res['CA'].get_coord()
                    residues.append(res)
                    ca_coords.append(coord)
                except: pass
    
    if not residues: return dgl.graph(([], []), num_nodes=0)

    ca_coords = np.array(ca_coords, dtype=np.float32)
    ca_coords = check_and_fix_features(ca_coords)
    
    # 提取特征
    node_features = []
    for res in residues:
        try:
            feat = featurize_substructure(res, node_type="protein")
            node_features.append(feat)
        except:
            node_features.append(np.zeros(1, dtype=np.float32))
    
    node_features = np.array(node_features, dtype=np.float32)
    node_features = check_and_fix_features(node_features)

    # 距离连边 (残基内部接触图)
    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    i_idxs, j_idxs = np.where((dists <= d_res) & (dists > 0))
    edge_dists = dists[i_idxs, j_idxs]

    g = dgl.graph((i_idxs, j_idxs), num_nodes=len(residues))
    g.ndata['h'] = torch.tensor(node_features)
    g.ndata['pos'] = torch.tensor(ca_coords) # 保存坐标
    
    if len(edge_dists) > 0:
        g.edata['dist'] = torch.tensor(edge_dists, dtype=torch.float32).unsqueeze(-1)
    else:
        g.edata['dist'] = torch.tensor([], dtype=torch.float32).view(0, 1)
        
    return g

# =================================================================================
# 6. 子结构交互图 (Substructure Interaction Graph) - [修改：基于预构建图构建]
# =================================================================================
def build_substructure_interaction_graph(ligand_frag_g, protein_res_g, d_sub, sample_name=""):
    """
    基于已有的配体碎片图和蛋白残基图构建交互图。
    利用图中 ndata['pos'] 计算距离。
    """
    if ligand_frag_g.num_nodes() == 0 or protein_res_g.num_nodes() == 0:
        return dgl.graph(([], []), num_nodes=0)
    
    # 直接提取坐标
    l_pos = ligand_frag_g.ndata['pos'].numpy()
    p_pos = protein_res_g.ndata['pos'].numpy()
    
    n_lig = l_pos.shape[0]
    n_prot = p_pos.shape[0]

    # 计算距离
    diff = l_pos[:, np.newaxis, :] - p_pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    # 筛选
    l_indices, p_indices = np.where(dists <= d_sub)
    valid_dists = dists[l_indices, p_indices]

    src_list = []
    dst_list = []
    edge_dists = []

    for k, (l_idx, p_idx) in enumerate(zip(l_indices, p_indices)):
        d_val = valid_dists[k]
        
        # Frag -> Res
        src_list.append(l_idx)
        dst_list.append(n_lig + p_idx)
        edge_dists.append(d_val)
        
        # Res -> Frag
        src_list.append(n_lig + p_idx)
        dst_list.append(l_idx)
        edge_dists.append(d_val)

    total_nodes = n_lig + n_prot
    g = dgl.graph((src_list, dst_list), num_nodes=total_nodes)
    
    if len(edge_dists) > 0:
        edge_dists = np.array(edge_dists, dtype=np.float32)
        g.edata['dist'] = torch.tensor(edge_dists).unsqueeze(-1)
    else:
        g.edata['dist'] = torch.tensor([], dtype=torch.float32).view(0, 1)

    return g