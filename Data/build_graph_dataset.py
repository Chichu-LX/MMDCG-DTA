import os
import pickle
import yaml
import io
import re
import numpy as np
import torch
import traceback

# 从 graphs.py 中导入函数
from graphs import (
    build_ligand_atom_graph,
    build_protein_atom_graph,
    build_atom_interaction_graph,
    build_ligand_fragment_graph,
    build_protein_residue_graph,
    build_substructure_interaction_graph
)

def load_affinity_labels(index_file):
    affinity_dict = {}
    if not os.path.exists(index_file): return {}
    with open(index_file, "r") as f:
        lines = f.readlines()
    for line in lines[6:]:
        parts = line.strip().split()
        if len(parts) < 4: continue
        try: affinity_dict[parts[0]] = float(parts[3])
        except: pass
    return affinity_dict

def clean_sdf_content(sdf_content):
    lines = sdf_content.split('\n')
    return '\n'.join([l for l in lines if 'nan' not in l.lower() and 'inf' not in l.lower()])

def clean_pdb_content(pdb_content):
    lines = pdb_content.split('\n')
    cleaned = []
    for line in lines:
        if 'nan' in line.lower() or 'inf' in line.lower(): continue
        if line.startswith('ATOM') or line.startswith('HETATM'):
            if len(line) < 54: continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def print_graph_info(graph, graph_name, sample_name):
    # 简化的打印函数，避免日志过大
    print(f"    {graph_name}: Nodes={graph.num_nodes()}, Edges={graph.num_edges()}")

def process_single_complex(comp_id, data, affinity_labels, d_atom, d_res, d_sub, stats, verbose=True):
    """
    [修改] 线性处理流程：
    1. 构建基础图 (Atom, Fragment, Residue)
    2. 基于基础图构建交互图 (Interaction)
    """
    try:
        ligand_content = clean_sdf_content(data['ligand'])
        pocket_content = clean_pdb_content(data['pocket'])
        
        # --- 第一步：构建基础图 ---
        
        # 1. Ligand Atom Graph
        ligand_atom_g = build_ligand_atom_graph(ligand_content, comp_id)
        if ligand_atom_g.num_nodes() == 0: raise ValueError("Empty Ligand Atom Graph")

        # 2. Protein Atom Graph
        protein_atom_g = build_protein_atom_graph(pocket_content, comp_id)
        if protein_atom_g.num_nodes() == 0: raise ValueError("Empty Protein Atom Graph")

        # 3. Ligand Fragment Graph
        ligand_fragment_g = build_ligand_fragment_graph(ligand_content, comp_id)
        # 允许碎片图为空吗？如果不允许，可以在这里抛出异常
        
        # 4. Protein Residue Graph
        protein_residue_g = build_protein_residue_graph(pocket_content, d_res, comp_id)
        
        # --- 第二步：构建交互图 (使用已构建的图对象) ---
        
        # 5. Atom Interaction Graph
        # 直接传入图对象，graphs.py 会利用 ndata['pos']
        atom_interaction_g = build_atom_interaction_graph(
            ligand_atom_g, protein_atom_g, d_atom, comp_id
        )
        
        # 6. Substructure Interaction Graph
        substructure_interaction_g = build_substructure_interaction_graph(
            ligand_fragment_g, protein_residue_g, d_sub, comp_id
        )
        
        # --- 第三步：打包 ---
        label = affinity_labels.get(comp_id, None)
        
        # 统计空图 (仅供参考)
        if atom_interaction_g.num_edges() == 0: stats['empty_graphs'] += 1
        
        stats['success'] += 1
        
        if verbose:
            print(f"\n[{comp_id}] Success. Label={label}")
            print_graph_info(ligand_atom_g, "L-Atom", comp_id)
            print_graph_info(protein_atom_g, "P-Atom", comp_id)
            print_graph_info(atom_interaction_g, "Inter-Atom", comp_id)
            print_graph_info(ligand_fragment_g, "L-Frag", comp_id)
            print_graph_info(protein_residue_g, "P-Res", comp_id)
            print_graph_info(substructure_interaction_g, "Inter-Sub", comp_id)
        
        return {
            'ligand_atom_graph': ligand_atom_g,
            'protein_atom_graph': protein_atom_g,
            'atom_interaction_graph': atom_interaction_g, # 包含真实边
            'ligand_fragment_graph': ligand_fragment_g,
            'protein_residue_graph': protein_residue_g,
            'substructure_interaction_graph': substructure_interaction_g, # 包含真实边
            'label': label
        }
        
    except Exception as e:
        stats['failed'] += 1
        if verbose: print(f"  Failed {comp_id}: {e}")
        return None

def build_sample_graphs(dataset, affinity_labels, d_atom, d_res, d_sub, verbose=True):
    samples = {}
    stats = {'success': 0, 'failed': 0, 'empty_graphs': 0}
    total = len(dataset)
    count = 0
    
    for comp_id, data in dataset.items():
        count += 1
        # 只在前5个样本打印详细日志，后面只打印进度
        is_verbose = (count <= 5)
        if count % 100 == 0: print(f"Processing {count}/{total}...")
        
        sample = process_single_complex(comp_id, data, affinity_labels, d_atom, d_res, d_sub, stats, verbose=is_verbose)
        if sample:
            samples[comp_id] = sample

    print(f"\nFinished processing. Success: {stats['success']}, Failed: {stats['failed']}")
    return samples

def main():
    if not os.path.exists("core_set.pkl"):
        print("Data not found (core_set.pkl). Please run loader.py first.")
        return

    with open("core_set.pkl", "rb") as f: core_set = pickle.load(f)
    with open("refined_set.pkl", "rb") as f: refined_set = pickle.load(f)
    
    d_atom = 4.0
    d_res = 8.0
    d_sub = 8.0
    
    if os.path.exists("INDEX_data.2016"):
        affinity_labels = load_affinity_labels("INDEX_data.2016")
    else:
        affinity_labels = {}

    print("Building Core Set...")
    core_graphs = build_sample_graphs(core_set, affinity_labels, d_atom, d_res, d_sub)
    with open("core_set_graphs.pkl", "wb") as f: pickle.dump(core_graphs, f)
    
    print("Building Refined Set...")
    ref_graphs = build_sample_graphs(refined_set, affinity_labels, d_atom, d_res, d_sub)
    with open("refined_set_graphs.pkl", "wb") as f: pickle.dump(ref_graphs, f)

if __name__ == "__main__":
    main()