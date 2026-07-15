import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
import pickle
import os
import time
import numpy as np

# 导入 Stage 1 模型
from MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1

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

def patch_add_group_ids(dataset, name="Dataset"):
    print(f"Running data patch on {name}: Generating 'group' IDs...")
    count = 0
    for sample in dataset:
        if sample is None: continue
        try:
            # 配体
            l_atom_g = sample['ligand_atom_graph']
            l_frag_g = sample['ligand_fragment_graph']
            if 'pos' in l_atom_g.ndata:
                atom_pos = l_atom_g.ndata['pos']
                num_frags = l_frag_g.num_nodes()
                if num_frags > 0:
                    labels, centers = simple_kmeans(atom_pos, num_frags)
                    l_atom_g.ndata['group'] = labels.to(torch.int32)
                    if 'pos' not in l_frag_g.ndata: l_frag_g.ndata['pos'] = centers.to(torch.float32)
            # 蛋白
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
        except: continue
    print(f"Patched {count} samples in {name}.")

# =============================================================================
# Collate Function
# =============================================================================
def collate_fn(batch):
    final_batch = []
    for b in batch:
        if b is None: continue
        try:
            if 'group' not in b['ligand_atom_graph'].ndata: continue
            if 'group' not in b['protein_atom_graph'].ndata: continue
            if b['label'] is None: continue
            final_batch.append(b)
        except: continue
    if len(final_batch) == 0: return None
    
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
# Evaluation Function (Stage 1 specific)
# =============================================================================
def evaluate(model, dataloader, device):
    model.eval()
    total_mae = 0
    count = 0
    with torch.no_grad():
        for sample in dataloader:
            if sample is None: continue
            for k in sample:
                if isinstance(sample[k], (torch.Tensor, dgl.DGLGraph)):
                    sample[k] = sample[k].to(device)
            
            # Stage 1 模型只返回 preds 1个值
            preds = model(sample)
            preds = preds.view(-1, 1)
            
            # 计算 MAE
            mae = torch.abs(preds - sample['label']).sum().item()
            total_mae += mae
            count += sample['label'].size(0)
            
    return total_mae / (count + 1e-8)

# =============================================================================
# Main Training Function
# =============================================================================
def train_stage1():
    config = {
        "l_intra": 2,
        "l_inter": 2,
        "l_atom": 2,
        "l_sub": 2,
        "embedding_dim": 64,
        "inter_negative_slope": 0.2,
        "d_atom": 4.0,  # Stage 1 保持为 4.0 和 8.0 阈值
        "d_sub": 8.0,
        "raw_atom_dim": 5,   
        "sub_x_dim": 5,      
        "prot_res_dim": 1,
        "use_checkpoint": True,
        "lr": 1e-4,
        "epochs": 1000, # 改为 1000 轮
        "batch_size": 16,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print(f"Using device: {config['device']}")
    
    # 1. 加载训练集和测试集
    if not os.path.exists("refined_set_graphs.pkl") or not os.path.exists("core_set_graphs.pkl"):
        print("Error: Dataset files missing.")
        return

    with open("refined_set_graphs.pkl", "rb") as f:
        train_data = list(pickle.load(f).values())
    with open("core_set_graphs.pkl", "rb") as f:
        test_data = list(pickle.load(f).values())
    
    # 数据补丁
    patch_add_group_ids(train_data, name="TrainSet")
    patch_add_group_ids(test_data, name="TestSet")
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # 2. 初始化模型
    model = MMDCGDTAModel_Stage1(config).to(config['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    
    # 3. 训练循环
    print(f"Start Training Stage 1 (Max Epochs: {config['epochs']}, No Early Stopping)...")
    
    best_test_mae = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        total_train_loss = 0
        
        for batch_idx, sample in enumerate(train_loader):
            if sample is None: continue
            
            for k in sample:
                if isinstance(sample[k], (torch.Tensor, dgl.DGLGraph)):
                    sample[k] = sample[k].to(config['device'])
            
            optimizer.zero_grad()
            
            # Stage 1 返回单值 preds
            preds = model(sample)
            preds = preds.view(-1, 1)
            
            loss = criterion(preds, sample['label'])
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- 每轮评估 ---
        test_mae = evaluate(model, test_loader, config['device'])
        
        print(f"Epoch {epoch} | Train Loss (MSE): {avg_train_loss:.4f} | Test MAE: {test_mae:.4f}")
        
        # --- 保存最佳模型 ---
        if test_mae < best_test_mae:
            best_test_mae = test_mae
            torch.save(model.state_dict(), "stage1_model_best.pth")
            print(f"  >>> New Best Model Saved (MAE: {best_test_mae:.4f})")
            
        # --- 每 50 轮固定保存一次 ---
        if (epoch + 1) % 50 == 0:
            save_path = f"stage1_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Periodic Checkpoint Saved: {save_path}")

    # 4. 最终保存
    final_path = "stage1_model_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Stage 1 Complete. Final Model saved to {final_path}")
    print(f"Best Test MAE achieved: {best_test_mae:.4f}")

if __name__ == "__main__":
    train_stage1()