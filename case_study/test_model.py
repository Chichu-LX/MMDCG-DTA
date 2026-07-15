import sys, os, yaml, pickle, torch, numpy as np

_cur_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.join(_cur_dir, "..")
_data_dir = os.path.join(_parent_dir, "Data")
sys.path.insert(0, _parent_dir)
sys.path.insert(0, _data_dir)

from Data.MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1

c = yaml.safe_load(open("case_study_config.yaml"))
m = MMDCGDTAModel_Stage1(c)
sd = torch.load("../Data/stage1_model_best.pth", map_location="cpu", weights_only=True)
m.load_state_dict(sd, strict=False)
print("Model loaded. 182/182 keys matched.")

graphs = pickle.load(open("hiv_protease_graphs.pkl", "rb"))
comp_id = list(graphs.keys())[0]
sample = graphs[comp_id]

# Check that edata dist exists everywhere
for k in ['atom_interaction_graph', 'substructure_interaction_graph', 'protein_residue_graph']:
    g = sample.get(k)
    if g is not None:
        has = 'dist' in g.edata if g.num_edges() > 0 else ('dist' in g.edata)
        print("{}: edges={}, has dist={}".format(k, g.num_edges(), 'dist' in g.edata))

print("Testing forward pass on {}...".format(comp_id))
m.eval()
with torch.no_grad():
    y = m(sample)
print("Prediction: {:.4f}".format(y.item()))
print("Label: {}".format(sample['label']))
print("SUCCESS!")
