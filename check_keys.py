import pickle
import dgl
import torch

def check_keys():
    print("正在加载 refined_set_graphs.pkl ...")
    if not os.path.exists("refined_set_graphs.pkl"):
        print("错误: 文件不存在！")
        return

    with open("refined_set_graphs.pkl", "rb") as f:
        data = pickle.load(f)
    
    # 获取第一个样本
    first_key = list(data.keys())[0]
    sample = data[first_key]
    
    print(f"\n样本 ID: {first_key}")
    print("-" * 30)
    
    # 检查 Ligand 图的键
    if "ligand_atom_graph" in sample:
        g = sample["ligand_atom_graph"]
        print("【配体图 (Ligand Graph) 节点特征键名】:")
        print(g.ndata.keys())
        if "h" in g.ndata:
             print(f"  -> 'h' shape: {g.ndata['h'].shape}")
    
    print("-" * 30)

    # 检查 Protein 图的键
    if "protein_atom_graph" in sample:
        g = sample["protein_atom_graph"]
        print("【蛋白图 (Protein Graph) 节点特征键名】:")
        print(g.ndata.keys())

import os
if __name__ == "__main__":
    check_keys()
