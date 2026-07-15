#!/usr/bin/env python3
"""Quick stratified VS with actual metrics. Runs in ~30-60 min for 4.5K compounds."""
import os, sys, pickle, time, json, random
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/Data")
import dgl
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdFMCS
from graphs import (
    build_ligand_atom_graph, build_protein_atom_graph,
    build_atom_interaction_graph, build_ligand_fragment_graph,
    build_protein_residue_graph, build_substructure_interaction_graph,
)
from MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1
from train_stage1_new import patch_add_group_ids

SCRIPT_DIR = "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/virtual_screening"
OUT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(SCRIPT_DIR)

D_ATOM = 4.5
D_SUB = 8.5
D_RES = 8.5
device = torch.device("cuda")

# === Load DUD-E ===
def parse_ism(fp):
    res = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    res.append({"smiles": parts[0], "id": parts[1]})
    return res

actives = parse_ism("dude_data/actives_final.ism")
decoys = parse_ism("dude_data/decoys_final.ism")
print(f"Full dataset: {len(actives)} actives, {len(decoys)} decoys")

# Stratified sample: ALL actives + 4000 random decoys
random.seed(42)
np.random.seed(42)
sample_decoys = random.sample(decoys, min(4000, len(decoys)))
all_compounds = (
    [(f"active_{c['id']}", c["smiles"], 1) for c in actives]
    + [(f"decoy_{c['id']}", c["smiles"], 0) for c in sample_decoys]
)
random.shuffle(all_compounds)
print(f"Stratified sample: {len(all_compounds)} ({len(actives)} actives + {len(sample_decoys)} decoys)")

# === Load reference ===
with open("/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study/hiv_protease_graphs.pkl", "rb") as f:
    gdata = pickle.load(f)
ref_complex = gdata["1HPV"]
patch_add_group_ids([ref_complex], name="Ref")
print(f"Reference: 1HPV, {ref_complex['protein_atom_graph'].num_nodes()} protein atoms")

# === Load crystal ligand template ===
template_mol = None
tp = "dude_data/crystal_ligand.mol2"
if os.path.exists(tp):
    template_mol = Chem.MolFromMol2File(tp, removeHs=False)
    if template_mol is not None:
        print(f"Template ligand: {template_mol.GetNumAtoms()} atoms")

# === Build graphs ===
print("\n[1/4] Building graphs...")
vs_graphs = {}
t0 = time.time()

for i, (cid, smiles, label_val) in enumerate(all_compounds):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)

        try:
            params = AllChem.ETKDGv3()
        except AttributeError:
            params = AllChem.ETKDG()
        params.randomSeed = 42
        params.numThreads = 0
        params.pruneRmsThresh = 0.5
        AllChem.EmbedMultipleConfs(mol, numConfs=10, params=params)
        if mol.GetNumConformers() == 0:
            continue

        # Align to template if available
        if template_mol is not None:
            try:
                mcs = rdFMCS.FindMCS(
                    [Chem.RemoveHs(mol), Chem.RemoveHs(template_mol)],
                    atomCompare=rdFMCS.AtomCompare.CompareElements,
                    timeout=2,
                )
                if mcs.numAtoms >= 4:
                    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
                    if mcs_mol:
                        mm = Chem.RemoveHs(mol).GetSubstructMatch(mcs_mol)
                        mt = Chem.RemoveHs(template_mol).GetSubstructMatch(mcs_mol)
                        if mm and mt:
                            rdMolAlign.AlignMol(mol, template_mol, atomMap=list(zip(mm, mt)))
            except Exception:
                pass

        AllChem.ComputeGasteigerCharges(mol)
        sdf = Chem.MolToMolBlock(mol)

        lig_atom = build_ligand_atom_graph(sdf)
        if lig_atom.num_nodes() == 0:
            continue
        lig_frag = build_ligand_fragment_graph(sdf)
        if lig_frag.num_nodes() == 0:
            continue

        prot_atom = ref_complex["protein_atom_graph"]
        prot_res = ref_complex["protein_residue_graph"]

        atom_inter = build_atom_interaction_graph(lig_atom, prot_atom, D_ATOM)
        sub_inter = build_substructure_interaction_graph(lig_frag, prot_res, D_SUB)

        vs_graphs[cid] = {
            "ligand_atom_graph": lig_atom,
            "protein_atom_graph": prot_atom,
            "atom_interaction_graph": atom_inter,
            "ligand_fragment_graph": lig_frag,
            "protein_residue_graph": prot_res,
            "substructure_interaction_graph": sub_inter,
            "label": label_val,
        }
    except Exception:
        pass

    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{len(all_compounds)} ({len(vs_graphs)} ok), "
              f"{elapsed:.0f}s, ~{(i+1)/max(1,elapsed/60):.0f}/min", flush=True)

print(f"Built {len(vs_graphs)}/{len(all_compounds)} graphs in {time.time()-t0:.0f}s")

# === Patch group IDs ===
print("\n[2/4] Patching group IDs...")
samples = list(vs_graphs.values())
patch_add_group_ids(samples, name="VS")
valid = {}
for cid, s in vs_graphs.items():
    if "group" in s["ligand_atom_graph"].ndata and "group" in s["protein_atom_graph"].ndata:
        valid[cid] = s
print(f"Valid: {len(valid)}/{len(vs_graphs)}")

# === Load model ===
print("\n[3/4] Loading model + extracting features...")
config = {
    "embedding_dim": 64, "d_atom": D_ATOM, "d_res": D_RES, "d_sub": D_SUB,
    "l_intra": 2, "l_inter": 2, "l_atom": 2, "l_sub": 2,
    "inter_negative_slope": 0.2, "sub_x_dim": 5, "raw_atom_dim": 5,
    "prot_res_dim": 1, "use_checkpoint": True,
}

encoder = MMDCGDTAModel_Stage1(config).to(device)
encoder.eval()
ckpt = "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/Data/stage1_model_best.pth"
encoder.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False), strict=False)
print(f"Loaded encoder, fusion_dim={encoder.fusion_dim}")

# Extract features
features = {}
with torch.no_grad():
    for i, (cid, sample) in enumerate(valid.items()):
        try:
            sd = {k: v.to(device) if hasattr(v, "to") and k != "label" else v
                  for k, v in sample.items() if k != "label"}

            lig = sd["ligand_atom_graph"]
            prot = sd["protein_atom_graph"]
            atom_i = sd["atom_interaction_graph"]
            sub_i = sd["substructure_interaction_graph"]
            lfg = sd["ligand_fragment_graph"]
            prg = sd["protein_residue_graph"]

            L_b, L_bw = encoder._calc_bond_energy_weights(lig, encoder.ligand_bond_sim)
            P_b, P_bw = encoder._calc_bond_energy_weights(prot, encoder.protein_bond_sim)

            li = encoder.ligand_atom_intra_encoder(lig, lig.ndata["h"], edge_weights=L_bw)
            pi = encoder.protein_atom_intra_encoder(prot, prot.ndata["h"], edge_weights=P_bw)

            L_ang, L_tor = encoder._calc_angle_energy(lig, encoder.ligand_angle_sim, li)
            P_ang, P_tor = encoder._calc_angle_energy(prot, encoder.protein_angle_sim, pi)
            I_vdw, I_elec, I_hb = encoder._calc_inter_energy(atom_i, li, pi)

            il_, ip_ = encoder.inter_atom_encoder(atom_i, li, pi)

            lg = encoder._get_batch_offset_group_ids(lig, lfg, "group")
            pg = encoder._get_batch_offset_group_ids(prot, prg, "group")

            uil, uhl = encoder.ligand_atom_interactive(li, il_, lg)
            uip, uhp = encoder.protein_atom_interactive(pi, ip_, pg)

            asl = torch.zeros(lfg.num_nodes(), encoder.d, device=lig.device)
            asl.index_add_(0, lg, uhl)
            nlf = torch.cat([lfg.ndata["h"], asl], dim=1)

            asp = torch.zeros(prg.num_nodes(), encoder.d, device=prot.device)
            asp.index_add_(0, pg, uhp)
            npf = torch.cat([prg.ndata["h"], asp], dim=1)

            lsi = encoder.frag_proj(nlf)
            psi = encoder.res_proj(npf)

            lis = encoder.ligand_frag_intra_encoder(lfg, lsi)
            pe = prg.edata.get("dist") if "dist" in prg.edata else None
            pis = encoder.protein_res_intra_encoder(prg, psi, pe)

            ils, ips = encoder.inter_sub_encoder(sub_i, lis, pis)
            lsu_i, lsu_a = encoder.ligand_sub_interactive(lis, ils)
            psu_i, psu_a = encoder.protein_sub_interactive(pis, ips)

            def sm(g, f):
                with g.local_scope():
                    g.ndata["t"] = f
                    return dgl.readout_nodes(g, "t", op="mean")

            lp_i = sm(lfg, lsu_a); pp_i = sm(prg, psu_a)
            lp_e = sm(lfg, lsu_i); pp_e = sm(prg, psu_i)

            H_gnn = torch.cat([lp_i, pp_i, lp_e, pp_e], dim=1)
            H_phys = torch.cat([L_b, L_ang, L_tor, P_b, P_ang, P_tor,
                                I_vdw, I_elec, I_hb], dim=1)
            F = torch.cat([H_gnn, H_phys], dim=1)
            features[cid] = F.cpu()
        except Exception as e:
            if i < 3:
                print(f"  Err {cid}: {type(e).__name__}: {str(e)[:80]}")

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(valid)}...", flush=True)

print(f"Features: {len(features)} compounds")

# === Score and evaluate ===
print("\n[4/4] Scoring and evaluating...")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

cids = list(features.keys())
X = torch.cat([features[c] for c in cids], dim=0)
y_labels = np.array([valid[c]["label"] for c in cids])

X_np = StandardScaler().fit_transform(X.numpy())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=5000, C=10.0, class_weight="balanced", random_state=42)

try:
    y_score = cross_val_predict(clf, X_np, y_labels, cv=cv, method="predict_proba")[:, 1]
except Exception as e:
    print(f"  CV failed ({e}), using single fit")
    split = len(X_np) * 4 // 5
    clf.fit(X_np[:split], y_labels[:split])
    y_score = clf.predict_proba(X_np[split:])[:, 1]
    y_labels = y_labels[split:]

scores = {cids[i]: float(y_score[i]) for i in range(len(y_score))}

# Metrics
valid_m = ~np.isnan(y_score) & ~np.isinf(y_score)
yt, ys = y_labels[valid_m], y_score[valid_m]
n_a = int(np.sum(yt))
n_d = len(yt) - n_a

roc_auc = roc_auc_score(yt, ys)
pr_auc = average_precision_score(yt, ys)

si = np.argsort(ys)[::-1]
sl = yt[si]
efs = {}
for pct in [0.5, 1, 2, 5, 10]:
    tn = max(1, int(len(yt) * pct / 100))
    af = np.sum(sl[:tn])
    efs[f"EF_{pct}%"] = float((af / tn) / (n_a / len(yt)))

# BEDROC
def bedroc(yt, ys, a=20):
    n = len(yt); si = np.argsort(ys)[::-1]; sl = yt[si]
    na = np.sum(yt); ra = na / n
    w = np.exp(a * np.arange(1, n + 1) / n) / n
    w /= np.sum(w)
    riemax = ra * np.sum(w[:int(np.ceil(ra * n))])
    riemin = ra * np.sum(w[-int(np.ceil(ra * n)):])
    rie = np.sum(sl * w)
    return float((rie - riemin) / (riemax - riemin + 1e-10))

b20 = bedroc(yt, ys, 20)

print(f"\n{'='*60}")
print("MMDCG-DTA VIRTUAL SCREENING — DUD-E HIV-1 Protease")
print(f"{'='*60}")
print(f"  Compounds:   {len(yt)} ({n_a} actives / {n_d} decoys)")
print(f"  ROC-AUC:     {roc_auc:.4f}  {'✓' if roc_auc >= 0.97 else '✗ target 0.97'}")
print(f"  PR-AUC:      {pr_auc:.4f}   {'✓' if pr_auc >= 0.88 else '✗ target 0.88'}")
print(f"  BEDROC(20):  {b20:.4f}")
for k, v in efs.items():
    print(f"  {k}:     {v:.2f}×")
print(f"{'='*60}")

# Save
results = {
    "metrics": {
        "ROC_AUC": roc_auc, "PR_AUC": pr_auc, "BEDROC_20": b20, **efs
    },
    "n_actives": n_a, "n_decoys": n_d, "n_total": len(yt),
    "reference": "1HPV", "dataset": "DUD-E hivpr (stratified sample)",
    "targets": {"ROC>0.97": roc_auc >= 0.97, "PR>0.88": pr_auc >= 0.88},
}
with open(os.path.join(OUT_DIR, "vs_final_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Per-compound scores
sc_list = [{"compound_id": c, "predicted_score": float(scores.get(c, 0)),
            "label": int(valid[c]["label"]), "is_active": bool(valid[c]["label"] == 1)}
           for c in sorted(valid.keys())]
with open(os.path.join(OUT_DIR, "vs_compound_scores.json"), "w") as f:
    json.dump(sc_list, f, indent=2)

print(f"\nResults saved to {OUT_DIR}/")
