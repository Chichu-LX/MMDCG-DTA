import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


def featurize_atom(atom):
    """
    构建原子节点特征。
    输入: RDKit 的 atom 对象。
    提取的特征包括：
      - 原子序数
      - 形式电荷
      - 是否芳香（1 表示芳香，0 表示非芳香）
      - 杂化类型（SP:0, SP2:1, SP3:2, SP3D:3, SP3D2:4，其它为 -1）
      - Gasteiger 部分电荷（若未计算则为 0.0）
    返回: numpy 数组，dtype 为 float32。
    """
    atomic_num = atom.GetAtomicNum()
    formal_charge = atom.GetFormalCharge()
    aromatic = 1 if atom.GetIsAromatic() else 0

    hyb = atom.GetHybridization()
    if hyb == Chem.rdchem.HybridizationType.SP:
        hyb_val = 0
    elif hyb == Chem.rdchem.HybridizationType.SP2:
        hyb_val = 1
    elif hyb == Chem.rdchem.HybridizationType.SP3:
        hyb_val = 2
    elif hyb == Chem.rdchem.HybridizationType.SP3D:
        hyb_val = 3
    elif hyb == Chem.rdchem.HybridizationType.SP3D2:
        hyb_val = 4
    else:
        hyb_val = -1

    try:
        gasteiger = float(atom.GetProp('_GasteigerCharge'))
        # 检查是否为NaN或Inf
        if np.isnan(gasteiger) or np.isinf(gasteiger):
            gasteiger = 0.0
    except Exception:
        gasteiger = 0.0

    return np.array([atomic_num, formal_charge, aromatic, hyb_val, gasteiger], dtype=np.float32)


def safe_descriptor(descriptor_func, mol, default=0.0):
    """安全计算分子描述符，避免异常"""
    try:
        value = descriptor_func(mol)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def featurize_substructure(substructure, node_type="ligand"):
    """
    构建子结构节点特征。

    对于 ligand（配体碎片）：
      - substructure 为 RDKit Mol 对象
      - 计算分子描述符，如分子量、LogP、TPSA、可旋转键数量和总形式电荷

    对于 protein（蛋白残基）：
      - substructure 为 Biopython 的 Residue 对象
      - 这里简单地计算残基内所有原子的平均原子质量（作为示例，实际中可扩展为更多特征，
        如二面角、极性、疏水性、溶剂可及性、氢键信息等）

    参数:
      substructure: 子结构对象，根据 node_type 决定类型。
      node_type: 字符串，"ligand" 或 "protein"。

    返回: numpy 数组，dtype 为 float32，表示子结构特征向量。
    """
    if node_type == "ligand":
        # 处理配体碎片，substructure 为 RDKit Mol 对象
        mw = safe_descriptor(Descriptors.MolWt, substructure, 0.0)
        logp = safe_descriptor(Descriptors.MolLogP, substructure, 0.0)
        tpsa = safe_descriptor(Descriptors.TPSA, substructure, 0.0)
        rot_bonds = safe_descriptor(Descriptors.NumRotatableBonds, substructure, 0.0)
        
        # 总形式电荷
        try:
            formal_charge = sum([atom.GetFormalCharge() for atom in substructure.GetAtoms()])
        except Exception:
            formal_charge = 0.0
            
        features = np.array([mw, logp, tpsa, rot_bonds, formal_charge], dtype=np.float32)
        return features
    elif node_type == "protein":
        # 处理蛋白残基，substructure 为 Biopython 的 Residue 对象
        # 示例中使用残基中所有原子的平均原子质量作为特征
        # 构建一个简单的原子质量字典
        atomic_weights = {
            "H": 1.008,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "S": 32.06,
            "P": 30.974,
            "F": 18.998,
            "Cl": 35.45,
            "Br": 79.904,
            "I": 126.904,
            "NA": 22.990,  # Sodium
            "MG": 24.305,  # Magnesium
            "CA": 40.078,  # Calcium
            "ZN": 65.38,   # Zinc
            "FE": 55.845,  # Iron
            "CU": 63.546,  # Copper
        }
        weights = []
        for atom in substructure.get_atoms():
            elem = atom.element.strip().upper()
            wt = atomic_weights.get(elem, 0.0)
            weights.append(wt)
        if weights:
            avg_weight = np.mean(weights)
            if np.isnan(avg_weight) or np.isinf(avg_weight):
                avg_weight = 0.0
        else:
            avg_weight = 0.0
        return np.array([avg_weight], dtype=np.float32)
    else:
        raise ValueError("Unknown node_type: should be 'ligand' or 'protein'")


if __name__ == "__main__":
    # 示例用法：
    # 对于原子特征，需要先用 RDKit 解析一个分子对象，然后对每个原子调用 featurize_atom。
    sdf_content = """  
  Mrv1810 02101915262D          

  12 12  0  0  0  0            999 V2000
    1.2990   -0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2990   -0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2990   -2.2500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -3.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2990   -2.2500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5981   -0.7500    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.5981    0.7500    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.2990    2.2500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    3.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2990    2.2500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5981    0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  1  0
  4  5  1  0
  5  6  1  0
  6  1  1  0
  1  7  1  0
  7  8  1  0
  8  9  1  0
  9 10  1  0
 10 11  1  0
 11 12  1  0
M  END
    """
    mol = Chem.MolFromMolBlock(sdf_content, removeHs=False)
    if mol is not None:
        # 计算 Gasteiger 部分电荷
        from rdkit.Chem import AllChem
        AllChem.ComputeGasteigerCharges(mol)
        # 对第一个原子进行特征提取
        atom0 = mol.GetAtomWithIdx(0)
        atom_feat = featurize_atom(atom0)
        print("原子特征:", atom_feat)

        # 对 ligand 子结构示例，直接对整个分子提取描述符
        frag_feat = featurize_substructure(mol, node_type="ligand")
        print("配体子结构特征:", frag_feat)
    else:
        print("无法解析SDF内容")