import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
import pickle
import os
import time
import numpy as np

# 导入 Stage 2 模型
from MMDCG_DTA_Stage2 import MMDCGDTAModel_Stage2

# =============================================================================
# 损失函数
# =============================================================================
class EdgeClassificationLoss(nn.Module):
    def __init__(self):
        super(EdgeClassificationLoss, self).__init__()
        # 三分类交叉熵
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.ce(logits, labels)

# =============================================================================
# K-Means 和 数据补丁
# =============================================================================
def simple_kmeans(x, k, max_iters=10):
    if k <= 0: return torch.zeros(x.size(0), dtype=torch.long, device=x.device), torch.zeros(0, x.size(1), device=x.device)
    if k >= x.size(0): return torch.arange(x.size(0), device=x.device), x
    indices = torch.randperm(x.size(0), device=x.device)[:k]
    centers = x[indices]
    labels = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    for _ in range(max_iters):
        dists = torch.cdist(x, centers)
        new_labels = torch.argmin(dists, dim=1)
        if torch.equal(labels, new_labels): break
        labels = new_labels
        new_centers = []
        for i in range(k):
            mask = (labels == i)
            if mask.sum() > 0: new_centers.append(x[mask].mean(dim=0))
            else: new_centers.append(centers[i])
        centers = torch.stack(new_centers, dim=0)
    return labels, centers

def patch_add_group_ids(dataset, device="cpu"):
    print("Running data patch...")
    count = 0
    for sample in dataset:
        if sample is None: continue
        try:
            l_atom_g = sample['ligand_atom_graph']
            l_frag_g = sample['ligand_fragment_graph']
            if 'pos' in l_atom_g.ndata:
                atom_pos = l_atom_g.ndata['pos']
                num_frags = l_frag_g.num_nodes()
                if num_frags > 0:
                    labels, centers = simple_kmeans(atom_pos, num_frags)
                    l_atom_g.ndata['group'] = labels.to(torch.int32)
                    if 'pos' not in l_frag_g.ndata: l_frag_g.ndata['pos'] = centers.to(torch.float32)
            
            p_atom_g = sample['protein_atom_graph']
            p_res_g = sample['protein_residue_graph']
            if 'pos' in p_atom_g.ndata:
                atom_pos_p = p_atom_g.ndata['pos']
                num_res = p_res_g.num_nodes()
                if num_res > 0:
                    labels_p, centers_p = simple_kmeans(atom_pos_p, num_res)
                    p_atom_g.ndata['group'] = labels_p.to(torch.int32)
                    if 'pos' not in p_res_g.ndata: p_res_g.ndata['pos'] = centers_p.to(torch.float32)
            count += 1
        except: pass
    print(f"Patched {count} samples.")

def collate_fn(batch):
    valid_batch = []
    for b in batch:
        if b is None: continue
        try:
            if 'pos' not in b['ligand_atom_graph'].ndata or 'group' not in b['ligand_atom_graph'].ndata: continue
            if 'pos' not in b['protein_atom_graph'].ndata or 'group' not in b['protein_atom_graph'].ndata: continue
            if b['label'] is None: continue
            valid_batch.append(b)
        except: continue
    if len(valid_batch) == 0: return None
    
    batch_lig_atom = dgl.batch([b['ligand_atom_graph'] for b in valid_batch])
    batch_prot_atom = dgl.batch([b['protein_atom_graph'] for b in valid_batch])
    batch_lig_frag = dgl.batch([b['ligand_fragment_graph'] for b in valid_batch])
    batch_prot_res = dgl.batch([b['protein_residue_graph'] for b in valid_batch])
    batch_atom_inter = dgl.batch([b['atom_interaction_graph'] for b in valid_batch])
    batch_sub_inter = dgl.batch([b['substructure_interaction_graph'] for b in valid_batch])
    
    batch_labels = torch.tensor([b['label'] for b in valid_batch], dtype=torch.float32).view(-1, 1)
    
    return {
        'ligand_atom_graph': batch_lig_atom,
        'protein_atom_graph': batch_prot_atom,
        'ligand_fragment_graph': batch_lig_frag,
        'protein_residue_graph': batch_prot_res,
        'atom_interaction_graph': batch_atom_inter,
        'substructure_interaction_graph': batch_sub_inter,
        'label': batch_labels
    }

def generate_edge_labels(batch_graph, device):
    """
    基于距离生成三分类标签: 0:Remove, 1:Keep, 2:Add
    """
    dists = batch_graph.edata['dist'].squeeze(-1).to(device)
    labels = torch.zeros_like(dists, dtype=torch.long)
    
    mask_add = dists < 3.5
    mask_keep = (dists >= 3.5) & (dists < 6.0)
    
    labels[mask_add] = 2
    labels[mask_keep] = 1
    
    return labels

# =============================================================================
# Main Training (引入交替训练的内循环)
# =============================================================================
def train_stage2():
    config = {
        "l_intra": 2, "l_inter": 2, "l_atom": 2, "l_sub": 2,
        "embedding_dim": 64, "inter_negative_slope": 0.2,
        "d_atom": 8.0, 
        "d_sub": 8.0,
        "raw_atom_dim": 5, "sub_x_dim": 5, "prot_res_dim": 1,
        "use_checkpoint": True,
        "lr": 5e-5, 
        "epochs": 50, 
        "batch_size": 16, 
        "inner_max_iters": 5,   # [新增] 边重构内循环最大次数
        "inner_tolerance": 0.01,# [新增] 图稳定判定阈值：边类别改变比例 < 1%
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    print(f"Using device: {config['device']}")
    
    if not os.path.exists("core_set_graphs.pkl"): return
    with open("core_set_graphs.pkl", "rb") as f: core_data = pickle.load(f)
    dataset = list(core_data.values())
    patch_add_group_ids(dataset, device="cpu")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    
    # 1. 初始化模型
    model = MMDCGDTAModel_Stage2(config).to(config['device'])
    
    # 2. 加载 Stage 1 权重 (过滤掉 edge_classifier)
    stage1_path = "stage1_model_final.pth"
    if os.path.exists(stage1_path):
        print(f"Loading Stage 1 weights from {stage1_path}...")
        stage1_state = torch.load(stage1_path, map_location=config['device'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in stage1_state.items() 
                           if k in model_dict and v.shape == model_dict[k].shape and 'edge_classifier' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    
    # ==========================================================
    # [核心修改] 分离优化器：实现交替训练 (Alternating Training)
    # ==========================================================
    # 优化器 1: 专门负责边重构器
    edge_params = list(model.edge_classifier.parameters())
    optimizer_edge = optim.Adam(edge_params, lr=config['lr'])
    
    # 优化器 2: 专门负责主体网络 (GNN, 物理特征融合, 预测层)
    main_params = [p for n, p in model.named_parameters() if 'edge_classifier' not in n]
    optimizer_main = optim.Adam(main_params, lr=config['lr'])
    
    criterion_affinity = nn.MSELoss()
    criterion_edge = EdgeClassificationLoss() 
    
    print("Start Training Stage 2 (Alternating Optimization with Inner Loop)...")
    model.train()
    
    for epoch in range(config['epochs']):
        total_loss = 0
        
        for batch_idx, sample in enumerate(dataloader):
            if sample is None: continue
            for k in sample:
                if isinstance(sample[k], (torch.Tensor, dgl.DGLGraph)):
                    sample[k] = sample[k].to(config['device'])
            
            # -------------------------------------------------------------
            # Phase 1: 内循环 (Inner Loop) - 彻底训练边重构器，直到图结构稳定
            # -------------------------------------------------------------
            prev_classes = None
            actual_inner_iters = 0
            loss_edge = torch.tensor(0.0)
            
            for inner_step in range(config['inner_max_iters']):
                optimizer_edge.zero_grad()
                
                # 前向传播 (仅为了获取 logits)
                _, _, logits, _ = model(sample)
                
                if logits.numel() > 0:
                    edge_labels = generate_edge_labels(sample['atom_interaction_graph'], config['device'])
                    loss_edge = criterion_edge(logits, edge_labels)
                    
                    # 仅更新 EdgeReconstructor
                    loss_edge.backward()
                    optimizer_edge.step()
                    
                    # 判断图是否稳定
                    curr_classes = torch.argmax(logits, dim=1).detach()
                    if prev_classes is not None:
                        change_ratio = (curr_classes != prev_classes).float().mean().item()
                        if change_ratio <= config['inner_tolerance']:
                            # 图不再改变，终止内循环
                            actual_inner_iters = inner_step + 1
                            break
                    
                    prev_classes = curr_classes
                    actual_inner_iters = inner_step + 1
                else:
                    break # 没有边，直接跳出内循环
                    
            # -------------------------------------------------------------
            # Phase 2: 外更新 (Outer Update) - 基于稳定后的图结构更新结果
            # -------------------------------------------------------------
            optimizer_main.zero_grad()
            
            # 使用最新最优的边权重重新计算特征和亲和力
            preds, stats, _, _ = model(sample)
            preds = preds.view(-1, 1)
            
            # 仅更新主网络参数
            loss_main = criterion_affinity(preds, sample['label'])
            loss_main.backward()
            optimizer_main.step()
            
            total_loss += loss_main.item()
            
            # --- 打印日志 ---
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Main Loss: {loss_main.item():.4f} | "
                      f"Edge Loss: {loss_edge.item():.4f} (Inner Iterations: {actual_inner_iters}) | "
                      f"R/K/A: {stats['ratio_remove']:.2f}/{stats['ratio_keep']:.2f}/{stats['ratio_add']:.2f}")
        
        print(f"Epoch {epoch} Finished | Avg Main Loss: {total_loss / len(dataloader):.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"stage2_model_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "stage2_model_final.pth")
    print("Stage 2 Complete.")

if __name__ == "__main__":
    train_stage2()