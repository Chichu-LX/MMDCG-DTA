#!/usr/bin/env python3
"""Precompute MMDCG-DTA fusion features, then train MLP head on them. Much faster."""

import os, sys, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Data'))
from Data.MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1
import dgl

CASE_DIR = "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study"


def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    vx = y_true - np.mean(y_true)
    vy = y_pred - np.mean(y_pred)
    pearson = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + 1e-8)
    sd = np.std(y_true - y_pred)
    return {"RMSE": rmse, "MAE": mae, "Pearson": pearson, "SD": sd}


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def extract_fusion_features(encoder, graph_data, device):
    """Extract fusion features for all complexes."""
    encoder.eval()
    features = {}
    skipped = 0

    with torch.no_grad():
        for i, (cid, sample) in enumerate(graph_data.items()):
            try:
                if sample.get('label') is None:
                    skipped += 1
                    continue

                sample_dev = {}
                for k, v in sample.items():
                    if hasattr(v, 'to') and k != 'label':
                        sample_dev[k] = v.to(device)
                    elif k != 'label':
                        sample_dev[k] = v

                # Run encoder forward to get fusion features
                ligand_atom_graph = sample_dev["ligand_atom_graph"]
                protein_atom_graph = sample_dev["protein_atom_graph"]
                atom_interaction_graph = sample_dev["atom_interaction_graph"]
                substructure_interaction_graph = sample_dev.get("substructure_interaction_graph")

                # Bond energy
                L_E_bond_agg, L_bond_weights = encoder._calc_bond_energy_weights(
                    ligand_atom_graph, encoder.ligand_bond_sim)
                P_E_bond_agg, P_bond_weights = encoder._calc_bond_energy_weights(
                    protein_atom_graph, encoder.protein_bond_sim)

                # Intra encoding
                ligand_intra = encoder.ligand_atom_intra_encoder(
                    ligand_atom_graph, ligand_atom_graph.ndata["h"], edge_weights=L_bond_weights)
                protein_intra = encoder.protein_atom_intra_encoder(
                    protein_atom_graph, protein_atom_graph.ndata["h"], edge_weights=P_bond_weights)

                # Angle/torsion
                L_E_angle, L_E_torsion = encoder._calc_angle_energy(
                    ligand_atom_graph, encoder.ligand_angle_sim, ligand_intra)
                P_E_angle, P_E_torsion = encoder._calc_angle_energy(
                    protein_atom_graph, encoder.protein_angle_sim, protein_intra)

                # Inter energy
                I_E_vdw, I_E_elec, I_E_hbond = encoder._calc_inter_energy(
                    atom_interaction_graph, ligand_intra, protein_intra)

                # Inter encoding
                inter_lig, inter_prot = encoder.inter_atom_encoder(
                    atom_interaction_graph, ligand_intra, protein_intra)

                # HIL atom
                ligand_group = encoder._get_batch_offset_group_ids(
                    ligand_atom_graph, sample_dev["ligand_fragment_graph"], "group")
                protein_group = encoder._get_batch_offset_group_ids(
                    protein_atom_graph, sample_dev["protein_residue_graph"], "group")

                updated_inter_lig, updated_intra_lig = encoder.ligand_atom_interactive(
                    ligand_intra, inter_lig, ligand_group)
                updated_inter_prot, updated_intra_prot = encoder.protein_atom_interactive(
                    protein_intra, inter_prot, protein_group)

                # Aggregate
                ligand_frag_graph = sample_dev["ligand_fragment_graph"]
                protein_res_graph = sample_dev["protein_residue_graph"]

                atom_sum_lig = torch.zeros(ligand_frag_graph.num_nodes(), encoder.d,
                                           device=ligand_atom_graph.device)
                atom_sum_lig.index_add_(0, ligand_group, updated_intra_lig)
                new_ligand_feats = torch.cat([ligand_frag_graph.ndata["h"], atom_sum_lig], dim=1)

                atom_sum_prot = torch.zeros(protein_res_graph.num_nodes(), encoder.d,
                                           device=protein_atom_graph.device)
                atom_sum_prot.index_add_(0, protein_group, updated_intra_prot)
                new_protein_feats = torch.cat([protein_res_graph.ndata["h"], atom_sum_prot], dim=1)

                ligand_sub_input = encoder.frag_proj(new_ligand_feats)
                protein_sub_input = encoder.res_proj(new_protein_feats)

                # Substructure
                ligand_intra_sub = encoder.ligand_frag_intra_encoder(ligand_frag_graph, ligand_sub_input)
                prot_edge_feats = protein_res_graph.edata.get('dist') if 'dist' in protein_res_graph.edata else None
                protein_intra_sub = encoder.protein_res_intra_encoder(protein_res_graph, protein_sub_input, prot_edge_feats)

                inter_lig_sub, inter_prot_sub = encoder.inter_sub_encoder(
                    substructure_interaction_graph, ligand_intra_sub, protein_intra_sub)

                ligand_sub_updated_inter, ligand_sub_updated_intra = encoder.ligand_sub_interactive(
                    ligand_intra_sub, inter_lig_sub)
                protein_sub_updated_inter, protein_sub_updated_intra = encoder.protein_sub_interactive(
                    protein_intra_sub, inter_prot_sub)

                # Readout
                def safe_mean(g, feat):
                    with g.local_scope():
                        g.ndata['tmp_r'] = feat
                        return dgl.readout_nodes(g, 'tmp_r', op='mean')

                lig_pool_intra = safe_mean(ligand_frag_graph, ligand_sub_updated_intra)
                prot_pool_intra = safe_mean(protein_res_graph, protein_sub_updated_intra)
                lig_pool_inter = safe_mean(ligand_frag_graph, ligand_sub_updated_inter)
                prot_pool_inter = safe_mean(protein_res_graph, protein_sub_updated_inter)

                H_gnn = torch.cat([lig_pool_intra, prot_pool_intra,
                                  lig_pool_inter, prot_pool_inter], dim=1)
                H_physics = torch.cat([
                    L_E_bond_agg, L_E_angle, L_E_torsion,
                    P_E_bond_agg, P_E_angle, P_E_torsion,
                    I_E_vdw, I_E_elec, I_E_hbond
                ], dim=1)

                F_final = torch.cat([H_gnn, H_physics], dim=1)
                features[cid] = {
                    'fusion': F_final.cpu(),
                    'label': sample['label'],
                }

            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  Error {cid}: {type(e).__name__}: {str(e)[:80]}")

            if (i + 1) % 50 == 0:
                print(f"  Extracted {i+1}/{len(graph_data)}...", flush=True)

    print(f"Extracted features for {len(features)} complexes ({skipped} skipped)")
    return features


def main():
    print("=" * 60)
    print("MMDCG-DTA Feature Extraction + MLP Head Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    for graph_path in ["hiv_protease_graphs_full.pkl", "hiv_protease_graphs.pkl"]:
        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                graph_data = pickle.load(f)
            print(f"Loaded {len(graph_data)} graphs from {graph_path}")
            break

    labels = [g['label'] for g in graph_data.values() if g.get('label') is not None]
    target_mean = np.mean(labels)
    target_std = np.std(labels)
    print(f"Target: mean={target_mean:.3f}, std={target_std:.3f}, "
          f"range=[{min(labels):.2f}, {max(labels):.2f}], N={len(labels)}")

    # Load pretrained encoder
    config = {
        'embedding_dim': 64,
        'd_atom': 4.0, 'd_res': 8.0, 'd_sub': 8.0,
        'l_intra': 2, 'l_inter': 2, 'l_atom': 2, 'l_sub': 2,
        'inter_negative_slope': 0.2,
        'sub_x_dim': 5, 'raw_atom_dim': 5, 'prot_res_dim': 1,
    }

    encoder = MMDCGDTAModel_Stage1(config).to(device)

    loaded = False
    for pp in ["../Data/stage1_model_final.pth", "../Data/stage1_model_best.pth"]:
        if os.path.exists(pp):
            state = torch.load(pp, map_location=device, weights_only=True)
            encoder.load_state_dict(state, strict=False)
            print(f"Loaded pretrained encoder: {pp}")
            loaded = True
            break
    if not loaded:
        print("ERROR: No pretrained model!")
        return

    # Extract features
    print("\nExtracting MMDCG-DTA fusion features...")
    t0 = time.time()
    features = extract_fusion_features(encoder, graph_data, device)
    print(f"Feature extraction took {time.time() - t0:.0f}s")

    # Prepare training data
    cids = list(features.keys())
    X = torch.cat([features[c]['fusion'] for c in cids], dim=0)
    y = torch.tensor([features[c]['label'] for c in cids], dtype=torch.float32)

    print(f"Feature matrix: {X.shape}, labels: {y.shape}")
    fusion_dim = X.shape[1]
    print(f"Fusion dimension: {fusion_dim}")

    # Standardize targets
    y_std = (y - target_mean) / target_std

    # Train MLP head
    head = MLPHead(fusion_dim, hidden_dims=[256, 128, 64]).to(device)
    X_dev = X.to(device)
    y_dev = y_std.to(device)

    optimizer = optim.Adam(head.parameters(), lr=1e-3, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_epoch = 0
    patience = 0

    print(f"\nTraining MLP head on {len(cids)} samples...")
    for epoch in range(1, 501):
        head.train()
        optimizer.zero_grad()
        y_pred = head(X_dev)
        loss = criterion(y_pred.view(-1), y_dev.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Metrics
        with torch.no_grad():
            y_pred_np = y_pred.view(-1).cpu().numpy() * target_std + target_mean
            y_true_np = y.numpy()
            metrics = evaluate_metrics(y_true_np, y_pred_np)
            loss_val = loss.item()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss_val:.6f} | "
                  f"RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | "
                  f"Pearson: {metrics['Pearson']:.4f}", flush=True)

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
            patience = 0
            torch.save({
                'epoch': epoch,
                'head_state_dict': head.state_dict(),
                'target_mean': target_mean,
                'target_std': target_std,
                'metrics': metrics,
                'fusion_dim': fusion_dim,
            }, f"{CASE_DIR}/hiv_protease_best_model.pth")
        else:
            patience += 1
            if patience >= 100:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation
    print(f"\n{'='*60}")
    with torch.no_grad():
        head.eval()
        y_pred_final = head(X_dev).view(-1).cpu().numpy() * target_std + target_mean
        final_metrics = evaluate_metrics(y.numpy(), y_pred_final)

    print(f"Best epoch: {best_epoch}, Best Loss: {best_loss:.6f}")
    print(f"Final Metrics:")
    print(f"  RMSE:   {final_metrics['RMSE']:.4f}")
    print(f"  MAE:    {final_metrics['MAE']:.4f}")
    print(f"  Pearson: {final_metrics['Pearson']:.4f}")
    print(f"  SD:     {final_metrics['SD']:.4f}")
    print(f"{'='*60}")

    # Save predictions
    pred_dict = {}
    for i, cid in enumerate(cids):
        pred_dict[cid] = {
            'true': y[i].item(),
            'pred': float(y_pred_final[i]),
        }
    with open(f"{CASE_DIR}/pretrained_predictions.pkl", "wb") as f:
        pickle.dump(pred_dict, f)
    print("Predictions saved to pretrained_predictions.pkl")


if __name__ == "__main__":
    main()
