import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
import pickle
import os
import time
import numpy as np
import random
import copy

from MMDCG_DTA_Stage2 import MMDCGDTAModel_Stage2

# =============================================================================
# Loss & Optimizer
# =============================================================================
class EdgeClassificationLoss(nn.Module):
    def __init__(self):
        super(EdgeClassificationLoss, self).__init__()
        # 三分类交叉熵
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.ce(logits, labels)

class PCGradOptimizer:
    def __init__(self, optimizer):
        self._optim = optimizer
    def zero_grad(self):
        self._optim.zero_grad()
    def step(self):
        self._optim.step()
    def pc_backward(self, objectives):
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads)
        self._unflatten_grad(pc_grad, shapes, has_grads)
    def _project_conflicting(self, grads):
        pc_grad = copy.deepcopy(grads)
        random.shuffle(pc_grad)
        for g_i in pc_grad:
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    norm_g_j = g_j.norm()**2
                    if norm_g_j > 1e-8: g_i -= (g_i_g_j / norm_g_j) * g_j
        final_grad = torch.zeros_like(grads[0])
        for g in pc_grad: final_grad += g
        return final_grad
    def _pack_grad(self, objectives):
        grads = []
        shapes = []
        has_grads = []
        for loss in objectives:
            self._optim.zero_grad()
            loss.backward(retain_graph=True)
            grad_list = []
            shape_list = []
            has_grad_list = []
            for param in self._optim.param_groups[0]['params']:
                if param.grad is not None:
                    grad_list.append(param.grad.view(-1))
                    shape_list.append(param.shape)
                    has_grad_list.append(True)
                else:
                    shape_list.append(param.shape)
                    has_grad_list.append(False)
            if len(grad_list) > 0: grads.append(torch.cat(grad_list))
            else: grads.append(torch.zeros(0))
            shapes.append(shape_list)
            has_grads.append(has_grad_list)
        return grads, shapes[0], has_grads[0]
    def _unflatten_grad(self, pc_grad, shapes, has_grads):
        idx = 0
        for i, param in enumerate(self._optim.param_groups[0]['params']):
            if has_grads[i]:
                numel = param.numel()
                g = pc_grad[idx : idx + numel]
                param.grad = g.view(shapes[i])
                idx += numel
            else: param.grad = None

# =============================================================================
# K-Means & Patching
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
    
    # 距离阈值设定
    mask_add = dists < 3.5
    mask_keep = (dists >= 3.5) & (dists < 6.0)
    # mask_remove = dists >= 6.0 (Label 0 by default)
    
    labels[mask_add] = 2
    labels[mask_keep] = 1
    
    return labels

# =============================================================================
# Main Training
# =============================================================================
def train_stage2():
    config = {
        "l_intra": 2, "l_inter": 2, "l_atom": 2, "l_sub": 2,
        "embedding_dim": 64, "inter_negative_slope": 0.2,
        "d_atom": 8.0, 
        "d_sub": 8.0,
        "raw_atom_dim": 5, "sub_x_dim": 5, "prot_res_dim": 1,
        "use_checkpoint": True,
        "lr": 5e-5, "epochs": 50, "batch_size": 16, 
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    print(f"Using device: {config['device']}")
    
    if not os.path.exists("refined_set_graphs.pkl"): return
    with open("refined_set_graphs.pkl", "rb") as f: core_data = pickle.load(f)
    dataset = list(core_data.values())
    patch_add_group_ids(dataset, device="cpu")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    
    # 1. 初始化全新的 Stage 2 模型 (此时 edge_classifier 是全新的 3 分类层)
    model = MMDCGDTAModel_Stage2(config).to(config['device'])
    
    # 2. 加载 Stage 1 权重
    stage1_path = "stage1_model_final.pth"
    if os.path.exists(stage1_path):
        print(f"Loading Stage 1 weights from {stage1_path}...")
        stage1_state = torch.load(stage1_path, map_location=config['device'])
        
        model_dict = model.state_dict()
        
        # [关键修复] 过滤掉 edge_classifier 的权重
        # 即使 Stage 1 里有这个名字的层，维度也不对，所以必须过滤
        pretrained_dict = {k: v for k, v in stage1_state.items() 
                           if k in model_dict and v.shape == model_dict[k].shape and 'edge_classifier' not in k}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)} layers from Stage 1 (Edge Classifier re-initialized).")
    
    base_optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    optimizer = PCGradOptimizer(base_optimizer)
    
    criterion_affinity = nn.MSELoss()
    criterion_edge = EdgeClassificationLoss() 
    
    print("Start Training Stage 2 (3-Class Edge Reconstruction)...")
    model.train()
    
    for epoch in range(config['epochs']):
        total_loss = 0
        
        for batch_idx, sample in enumerate(dataloader):
            if sample is None: continue
            for k in sample:
                if isinstance(sample[k], torch.Tensor) or isinstance(sample[k], dgl.DGLGraph):
                    sample[k] = sample[k].to(config['device'])
            
            optimizer.zero_grad()
            
            # Forward
            preds, stats, logits, energies = model(sample)
            preds = preds.view(-1, 1)
            
            # Loss 1: Affinity
            loss_main = criterion_affinity(preds, sample['label'])
            
            # Loss 2: Edge Classification
            if logits.numel() > 0:
                edge_labels = generate_edge_labels(sample['atom_interaction_graph'], config['device'])
                loss_aux = criterion_edge(logits, edge_labels)
            else:
                loss_aux = torch.tensor(0.0, device=config['device'], requires_grad=True)
            
            # Pareto Update
            optimizer.pc_backward([loss_main, loss_aux])
            optimizer.step()
            
            total_loss += loss_main.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Main: {loss_main.item():.4f} | Aux: {loss_aux.item():.4f} | "
                      f"R/K/A: {stats['ratio_remove']:.2f}/{stats['ratio_keep']:.2f}/{stats['ratio_add']:.2f}")
        
        print(f"Epoch {epoch} Finished | Avg Main Loss: {total_loss / len(dataloader):.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"stage2_model_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "stage2_model_final.pth")
    print("Stage 2 Complete.")

if __name__ == "__main__":
    train_stage2()
