import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
import pickle
import os
import time
import numpy as np

# Import your model
from MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1

# =============================================================================
# K-Means & Patching Logic (Keep existing logic, ensure robust error handling)
# =============================================================================
def simple_kmeans(x, k, max_iters=10):
    if k <= 0:
        return torch.zeros(x.size(0), dtype=torch.long, device=x.device), torch.zeros(0, x.size(1), device=x.device)
    if k >= x.size(0):
        return torch.arange(x.size(0), device=x.device), x

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
            if mask.sum() > 0:
                new_centers.append(x[mask].mean(dim=0))
            else:
                new_centers.append(centers[i])
        centers = torch.stack(new_centers, dim=0)
    return labels, centers

def patch_add_group_ids(dataset, device="cpu"):
    """
    Attempts to add 'group' IDs to all samples.
    """
    print("Running data patch: Generating 'group' IDs via K-Means...")
    count = 0
    for i, sample in enumerate(dataset):
        if sample is None: continue
        
        try:
            # Ligand
            l_atom_g = sample['ligand_atom_graph']
            l_frag_g = sample['ligand_fragment_graph']
            
            if 'pos' in l_atom_g.ndata:
                atom_pos = l_atom_g.ndata['pos']
                num_frags = l_frag_g.num_nodes()
                if num_frags > 0:
                    labels, centers = simple_kmeans(atom_pos, num_frags)
                    l_atom_g.ndata['group'] = labels.to(torch.int32)
                    # Sync fragment positions if missing
                    if 'pos' not in l_frag_g.ndata:
                        l_frag_g.ndata['pos'] = centers.to(torch.float32)
                else:
                    # Edge case: 0 fragments (shouldn't happen in valid data)
                    pass

            # Protein
            p_atom_g = sample['protein_atom_graph']
            p_res_g = sample['protein_residue_graph']
            
            if 'pos' in p_atom_g.ndata:
                atom_pos_p = p_atom_g.ndata['pos']
                num_res = p_res_g.num_nodes()
                if num_res > 0:
                    labels_p, centers_p = simple_kmeans(atom_pos_p, num_res)
                    p_atom_g.ndata['group'] = labels_p.to(torch.int32)
                    if 'pos' not in p_res_g.ndata:
                        p_res_g.ndata['pos'] = centers_p.to(torch.float32)
            
            count += 1
            if count % 1000 == 0: print(f"  Patched {count} samples...")
            
        except Exception as e:
            print(f"Error patching sample {i}: {e}")
            continue

    print("Data patching complete.")

# =============================================================================
# [FIXED] Collate Function
# =============================================================================
def collate_fn(batch):
    # 1. Basic Filter: Remove None
    valid_batch = [b for b in batch if b is not None]
    if len(valid_batch) == 0: return None
    
    # 2. Schema Filter: Remove samples missing 'group' or 'pos'
    # This is crucial to prevent the DGL Schema Error
    final_batch = []
    for sample in valid_batch:
        try:
            l_g = sample['ligand_atom_graph']
            p_g = sample['protein_atom_graph']
            
            # Check Ligand features
            if 'pos' not in l_g.ndata or 'group' not in l_g.ndata:
                continue # Skip this sample
            
            # Check Protein features
            if 'pos' not in p_g.ndata or 'group' not in p_g.ndata:
                continue # Skip this sample
            
            # Check Label
            if sample['label'] is None:
                continue

            final_batch.append(sample)
        except:
            continue
            
    if len(final_batch) == 0: return None

    # 3. Extract and Batch
    ligand_atom_graphs = [b['ligand_atom_graph'] for b in final_batch]
    protein_atom_graphs = [b['protein_atom_graph'] for b in final_batch]
    ligand_fragment_graphs = [b['ligand_fragment_graph'] for b in final_batch]
    protein_residue_graphs = [b['protein_residue_graph'] for b in final_batch]
    
    # Interaction graphs (Pre-built)
    atom_interaction_graphs = [b['atom_interaction_graph'] for b in final_batch]
    substructure_interaction_graphs = [b['substructure_interaction_graph'] for b in final_batch]
    
    labels = [b['label'] for b in final_batch]
    
    # Batching with DGL
    batch_lig_atom = dgl.batch(ligand_atom_graphs)
    batch_prot_atom = dgl.batch(protein_atom_graphs)
    batch_lig_frag = dgl.batch(ligand_fragment_graphs)
    batch_prot_res = dgl.batch(protein_residue_graphs)
    batch_atom_inter = dgl.batch(atom_interaction_graphs)
    batch_sub_inter = dgl.batch(substructure_interaction_graphs)
    
    batch_labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    
    return {
        'ligand_atom_graph': batch_lig_atom,
        'protein_atom_graph': batch_prot_atom,
        'ligand_fragment_graph': batch_lig_frag,
        'protein_residue_graph': batch_prot_res,
        'atom_interaction_graph': batch_atom_inter,
        'substructure_interaction_graph': batch_sub_inter,
        'label': batch_labels
    }

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
        "d_atom": 4.0,
        "d_sub": 8.0,
        "raw_atom_dim": 5,   
        "sub_x_dim": 5,      
        "prot_res_dim": 1,
        "use_checkpoint": True,
        "lr": 1e-4,
        "epochs": 50,
        "batch_size": 16,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    print(f"Using device: {config['device']}")
    
    if not os.path.exists("refined_set_graphs.pkl"):
        print("Data not found. Please run build_graph_dataset.py first.")
        return

    print("Loading datasets...")
    with open("refined_set_graphs.pkl", "rb") as f:
        core_data = pickle.load(f)
    dataset = list(core_data.values())
    
    # Run the patch
    patch_add_group_ids(dataset, device="cpu") 
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn # Use the robust collate_fn
    )
    
    model = MMDCGDTAModel_Stage1(config).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    
    print("Start Training Stage 1 (Physics Augmented MMDCG-DTA - Pre-built Graphs)...")
    model.train()
    
    for epoch in range(config['epochs']):
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, sample in enumerate(dataloader):
            if sample is None: continue
            
            # Move data to GPU
            for k in sample:
                if isinstance(sample[k], torch.Tensor) or isinstance(sample[k], dgl.DGLGraph):
                    sample[k] = sample[k].to(config['device'])
            
            optimizer.zero_grad()
            preds = model(sample)
            loss = criterion(preds, sample['label'])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"Epoch {epoch} Finished | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.1f}s")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"stage1_model_epoch_{epoch+1}.pth")

    final_path = "stage1_model_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Stage 1 Training Complete. Model saved to {final_path}")

if __name__ == "__main__":
    train_stage1()
