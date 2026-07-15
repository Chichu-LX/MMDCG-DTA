#!/usr/bin/env python3
"""Quick test that VS pipeline can find all required data."""
import os, sys, pickle

_PARENT = "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main"
_DATA_DIR = os.path.join(_PARENT, "Data")
_CASE_DIR = os.path.join(_PARENT, "case_study")

sys.path.insert(0, _DATA_DIR)
os.chdir(os.path.join(_PARENT, "virtual_screening"))

print("Data dir:", _DATA_DIR, "exists:", os.path.isdir(_DATA_DIR))
print("Case dir:", _CASE_DIR, "exists:", os.path.isdir(_CASE_DIR))

# Test graph loading
graph_path = os.path.join(_CASE_DIR, "hiv_protease_graphs.pkl")
print("Graph path:", graph_path, "exists:", os.path.exists(graph_path))

with open(graph_path, "rb") as f:
    data = pickle.load(f)
print("Complexes:", len(data))
print("Has 1HPV:", "1HPV" in data)

s = data["1HPV"]
print("1HPV protein atoms:", s["protein_atom_graph"].num_nodes())
print("1HPV label:", s["label"])

# Test DUD-E data exists
dude_dir = os.path.join(_PARENT, "virtual_screening", "dude_data")
print("DUD-E actives:", os.path.exists(os.path.join(dude_dir, "actives_final.ism")))
print("DUD-E decoys:", os.path.exists(os.path.join(dude_dir, "decoys_final.ism")))

# Test imports work
from graphs import build_ligand_atom_graph
from MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1
print("All imports OK - pipeline should work!")

# Quick small test: extract features for 1HPV
import torch, dgl
config = {
    "embedding_dim": 64, "d_atom": 4.5, "d_res": 8.5, "d_sub": 8.5,
    "l_intra": 2, "l_inter": 2, "l_atom": 2, "l_sub": 2,
    "inter_negative_slope": 0.2, "sub_x_dim": 5, "raw_atom_dim": 5,
    "prot_res_dim": 1, "use_checkpoint": True,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

encoder = MMDCGDTAModel_Stage1(config).to(device)
encoder.eval()

# Load pretrained
ckpt = os.path.join(_DATA_DIR, "stage1_model_best.pth")
if os.path.exists(ckpt):
    state = torch.load(ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(state, strict=False)
    print("Loaded encoder from", ckpt)

    # Quick forward pass test
    from train_stage1_new import patch_add_group_ids
    patch_add_group_ids([s], name="test")

    sample_dev = {}
    for k, v in s.items():
        if hasattr(v, "to") and k != "label":
            sample_dev[k] = v.to(device)
        elif k != "label":
            sample_dev[k] = v

    with torch.no_grad():
        y = encoder(sample_dev)
        print("Forward pass OK, pred:", y.item())
    print("Fusion dim:", encoder.fusion_dim)
    print("Everything works! Ready to run full pipeline.")
else:
    print("No checkpoint at", ckpt)
