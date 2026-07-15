#!/usr/bin/env python3
"""Build graphs for missing complexes with process-level timeout via multiprocessing."""

import os, sys, pickle, json, time
import multiprocessing as mp

CASE_DIR = "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study"

def build_one(args):
    """Build graph for a single complex. Returns (cid, success, error_msg)."""
    cid, config = args
    try:
        sys.path.insert(0, CASE_DIR)
        from build_hiv_graphs import build_all_graphs

        with open(f"{CASE_DIR}/hiv_protease_raw.pkl", "rb") as f:
            raw_data = pickle.load(f)

        graphs = build_all_graphs({cid: raw_data[cid]}, config, verbose=False)

        if cid in graphs:
            g = graphs[cid]
            return (cid, True, f"OK P={g['protein_atom_graph'].num_nodes()}")
        else:
            return (cid, False, "empty_result")
    except Exception as e:
        return (cid, False, f"{type(e).__name__}: {str(e)[:100]}")

def main():
    # Load data
    with open(f"{CASE_DIR}/hiv_protease_raw.pkl", "rb") as f:
        raw_data = pickle.load(f)
    with open(f"{CASE_DIR}/hiv_protease_graphs.pkl", "rb") as f:
        existing_graphs = pickle.load(f)
    with open(f"{CASE_DIR}/hiv_pr_real_data/final_dataset.json") as f:
        all_entries = json.load(f)

    existing_ids = set(existing_graphs.keys())
    id_to_pkd = {e['pdb_id']: e['pKd'] for e in all_entries}
    missing_sorted = sorted(set(raw_data.keys()) - existing_ids, key=lambda x: id_to_pkd.get(x, 99))

    print(f"Building graphs for {len(missing_sorted)} missing complexes")
    print(f"pKd range: {min(id_to_pkd.get(c,99) for c in missing_sorted):.2f} - {max(id_to_pkd.get(c,99) for c in missing_sorted):.2f}")

    config = {"d_atom": 4.0, "d_res": 8.0, "d_sub": 8.0}
    success = {}
    failed = {}
    timeout_ids = set()
    TOTAL_TIMEOUT = 120  # seconds per complex

    for i, cid in enumerate(missing_sorted):
        pKd = id_to_pkd.get(cid, '?')
        print(f"  [{i+1}/{len(missing_sorted)}] {cid} (pKd={pKd:.2f})...", end="", flush=True)

        # Run in separate process with timeout (fork shares loaded DGL modules)
        ctx = mp.get_context('fork')
        proc = ctx.Process(target=build_one, args=((cid, config),))
        proc.start()
        proc.join(timeout=TOTAL_TIMEOUT)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
            timeout_ids.add(cid)
            print(f" TIMEOUT")
            failed[cid] = "timeout"
        elif proc.exitcode == 0:
            # Can't easily get return value from Process, so just build directly
            # since we know it completed successfully within timeout
            try:
                sys.path.insert(0, CASE_DIR)
                from build_hiv_graphs import build_all_graphs
                graphs = build_all_graphs({cid: raw_data[cid]}, config, verbose=False)
                if cid in graphs:
                    success.update(graphs)
                    g = graphs[cid]
                    print(f" OK (P={g['protein_atom_graph'].num_nodes()})")
                else:
                    print(f" SKIP")
                    failed[cid] = "empty"
            except Exception as e:
                print(f" ERR: {e}")
                failed[cid] = str(e)[:80]
        else:
            print(f" CRASH")
            failed[cid] = f"exitcode={proc.exitcode}"

        # Save every 20
        if (i + 1) % 20 == 0:
            all_g = dict(existing_graphs)
            all_g.update(success)
            tmp = f"{CASE_DIR}/hiv_protease_graphs_full.pkl"
            with open(tmp, "wb") as f:
                pickle.dump(all_g, f)
            print(f"  --- checkpoint: {len(all_g)} total ({len(success)} new) ---")

    # Final save
    all_graphs = dict(existing_graphs)
    all_graphs.update(success)

    for path in [f"{CASE_DIR}/hiv_protease_graphs_full.pkl", f"{CASE_DIR}/hiv_protease_graphs.pkl"]:
        with open(path, "wb") as f:
            pickle.dump(all_graphs, f)

    # Stats
    all_pkds = [id_to_pkd[c] for c in all_graphs if c in id_to_pkd]
    import statistics
    print(f"\n{'='*60}")
    print(f"Done: {len(all_graphs)} total ({len(success)} new, {len(failed)} failed, {len(timeout_ids)} timeouts)")
    if all_pkds:
        print(f"pKd: {min(all_pkds):.2f}-{max(all_pkds):.2f}, mean={sum(all_pkds)/len(all_pkds):.2f}, std={statistics.stdev(all_pkds):.2f}")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for c, r in sorted(failed.items(), key=lambda x: id_to_pkd.get(x[0], 99)):
            print(f"  {c}: pKd={id_to_pkd.get(c,'?'):.2f} - {r}")

if __name__ == "__main__":
    main()
