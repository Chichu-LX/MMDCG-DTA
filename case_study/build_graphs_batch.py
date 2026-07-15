#!/usr/bin/env python3
"""Build graphs in batches, saving intermediate results."""

import os, sys, pickle, yaml
sys.path.insert(0, "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study")
from build_hiv_graphs import build_all_graphs

case_dir = "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study"

with open(f"{case_dir}/hiv_protease_raw.pkl", "rb") as f:
    raw_data = pickle.load(f)

config = {"d_atom": 4.0, "d_res": 8.0, "d_sub": 8.0}
all_ids = list(raw_data.keys())
batch_size = 50
all_graphs = {}

print(f"Processing {len(all_ids)} complexes in batches of {batch_size}")

for start in range(0, len(all_ids), batch_size):
    batch_ids = all_ids[start:start + batch_size]
    batch_data = {k: raw_data[k] for k in batch_ids}

    print(f"\nBatch {start//batch_size + 1}: {batch_ids[0]} ... {batch_ids[-1]} ({len(batch_ids)} complexes)")

    try:
        graphs = build_all_graphs(batch_data, config, verbose=True)
        all_graphs.update(graphs)
        print(f"  Batch OK: {len(graphs)} graphs")
    except Exception as e:
        print(f"  Batch ERROR: {e}")

    # Save intermediate results
    tmp_path = f"{case_dir}/hiv_protease_graphs_tmp.pkl"
    with open(tmp_path, "wb") as f:
        pickle.dump(all_graphs, f)
    print(f"  Saved {len(all_graphs)} graphs to tmp file")

# Final save
final_path = f"{case_dir}/hiv_protease_graphs.pkl"
with open(final_path, "wb") as f:
    pickle.dump(all_graphs, f)
print(f"\nDONE: {len(all_graphs)} graphs saved to {final_path}")
