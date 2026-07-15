import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
import pickle
import os
import time
import numpy as np

# 导入 Stage 3 模型
from MMDCG_DTA_Stage3 import MMDCGDTAModel_Stage3

# =============================================================================
# K-Means 和 数据补丁 (与 Stage 1/2 严格对齐)
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

def patch_add_group_ids(dataset):
    print("Running data patch: Generating 'group' IDs and syncing physics coordinates...")
    count = 0
    for sample in dataset:
        if sample is None: continue
        try:
            # 配体补丁
            l_atom_g = sample['ligand_atom_graph']
            l_frag_g = sample['ligand_fragment_graph']
            if 'pos' in l_atom_g.ndata:
                atom_pos = l_atom_g.ndata['pos']
                num_frags = l_frag_g.num_nodes()
                if num_frags > 0:
                    labels, centers = simple_kmeans(atom_pos, num_frags)
                    l_atom_g.ndata['group'] = labels.to(torch.int32)
                    l_frag_g.ndata['pos'] = centers.to(torch.float32)
            # 蛋白补丁
            p_atom_g = sample['protein_atom_graph']
            p_res_g = sample['protein_residue_graph']
            if 'pos' in p_atom_g.ndata:
                atom_pos_p = p_atom_g.ndata['pos']
                num_res = p_res_g.num_nodes()
                if num_res > 0:
                    labels_p, centers_p = simple_kmeans(atom_pos_p, num_res)
                    p_atom_g.ndata['group'] = labels_p.to(torch.int32)
                    p_res_g.ndata['pos'] = centers_p.to(torch.float32)
            count += 1
        except: continue
    print(f"Patched {count} samples.")

# =============================================================================
# [修复版] Collate Function: 确保 Schema 完整并打包交互图
# =============================================================================
def collate_fn(batch):
    final_batch = []
    for b in batch:
        if b is None: continue
        try:
            # 必须通过 Schema 检查才能打包
            if 'group' not in b['ligand_atom_graph'].ndata: continue
            if 'group' not in b['protein_atom_graph'].ndata: continue
            if b['label'] is None: continue
            final_batch.append(b)
        except: continue
            
    if len(final_batch) == 0: return None
    
    # 执行打包
    return {
        'ligand_atom_graph': dgl.batch([b['ligand_atom_graph'] for b in final_batch]),
        'protein_atom_graph': dgl.batch([b['protein_atom_graph'] for b in final_batch]),
        'ligand_fragment_graph': dgl.batch([b['ligand_fragment_graph'] for b in final_batch]),
        'protein_residue_graph': dgl.batch([b['protein_residue_graph'] for b in final_batch]),
        'atom_interaction_graph': dgl.batch([b['atom_interaction_graph'] for b in final_batch]),
        'substructure_interaction_graph': dgl.batch([b['substructure_interaction_graph'] for b in final_batch]),
        'label': torch.tensor([b['label'] for b in final_batch], dtype=torch.float32).view(-1, 1)
    }

# =============================================================================
# Main Training Function
# =============================================================================
def train_stage3():
    config = {
        "l_intra": 2, "l_inter": 2, "l_atom": 2, "l_sub": 2,
        "embedding_dim": 64, "inter_negative_slope": 0.2,
        "d_atom": 8.0, "d_sub": 8.0,
        "raw_atom_dim": 5, "sub_x_dim": 5, "prot_res_dim": 1,
        "use_checkpoint": True,
        "lr": 1e-4, "epochs": 50, "batch_size": 16, 
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print(f"Using device: {config['device']}")
    
    # 1. 加载数据
    if not os.path.exists("refined_set_graphs.pkl"):
        print("Error: core_set_graphs.pkl not found.")
        return
    with open("refined_set_graphs.pkl", "rb") as f:
        dataset = list(pickle.load(f).values())
    
    patch_add_group_ids(dataset) 
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    
    # 2. 初始化模型并加载 Stage 2 权重
    model = MMDCGDTAModel_Stage3(config).to(config['device'])
    stage2_path = "stage2_model_final.pth"
    if os.path.exists(stage2_path):
        print(f"Loading Stage 2 weights from {stage2_path}...")
        model.load_state_dict(torch.load(stage2_path, map_location=config['device']), strict=False)
    else:
        print("Warning: Stage 2 model not found. Starting with initial weights.")

    # 3. [关键步骤] 冻结边重构器
    model.freeze_reconstructor()
    
    # 4. 优化器 (仅更新 requires_grad=True 的参数，如 Encoders 和 Physics MLP)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
    criterion = nn.MSELoss()
    
    # 5. 训练循环
    print("Start Training Stage 3 (Final Fine-tuning)...")
    model.train()
    
    for epoch in range(500):
        total_loss = 0
        for batch_idx, sample in enumerate(dataloader):
            if sample is None: continue
            for k in sample:
                if isinstance(sample[k], (torch.Tensor, dgl.DGLGraph)):
                    sample[k] = sample[k].to(config['device'])
            
            optimizer.zero_grad()
            
            # 解包 4 个值，忽略后两个（Stage 3 仅优化回归 Loss）
            preds, stats, _, _ = model(sample)
            preds = preds.view(-1, 1)
            
            loss = criterion(preds, sample['label'])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | MSE Loss: {loss.item():.4f} | "
                      f"R/K/A Ratio: {stats['ratio_remove']:.2f}/{stats['ratio_keep']:.2f}/{stats['ratio_add']:.2f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Finished | Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f"stage3_model_epoch_{epoch+1}.pth")

    final_path = "stage3_model_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Stage 3 Complete. Final Model saved to {final_path}")

if __name__ == "__main__":
    train_stage3()
