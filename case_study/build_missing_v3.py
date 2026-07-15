#!/usr/bin/env python3
"""Build graphs for missing complexes with file-based IPC.

Writes results to temp pickle files (avoids multiprocessing.Queue issues).
Processes lowest-pKd complexes first to maximize distribution widening.
"""

import os, sys, pickle, json, time, tempfile
import multiprocessing as mp

CASE_DIR = "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study"

def build_one_worker(cid, config, out_path):
    """Build graph. Writes (cid, graph_or_None, success, msg) to out_path."""
    try:
        sys.path.insert(0, CASE_DIR)
        from build_hiv_graphs import build_all_graphs

        with open(f"{CASE_DIR}/hiv_protease_raw.pkl", "rb") as f:
            raw_data = pickle.load(f)

        graphs = build_all_graphs({cid: raw_data[cid]}, config, verbose=False)

        if cid in graphs:
            g = graphs[cid]
            with open(out_path, "wb") as f:
                pickle.dump((cid, graphs[cid], True,
                            f"OK P={g['protein_atom_graph'].num_nodes()}"), f)
        else:
            with open(out_path, "wb") as f:
                pickle.dump((cid, None, False, "empty_result"), f)
    except Exception as e:
        try:
            with open(out_path, "wb") as f:
                pickle.dump((cid, None, False,
                            f"{type(e).__name__}: {str(e)[:100]}"), f)
        except:
            pass


def main():
    with open(f"{CASE_DIR}/hiv_protease_raw.pkl", "rb") as f:
        raw_data = pickle.load(f)
    with open(f"{CASE_DIR}/hiv_protease_graphs.pkl", "rb") as f:
        existing_graphs = pickle.load(f)
    with open(f"{CASE_DIR}/hiv_pr_real_data/final_dataset.json") as f:
        all_entries = json.load(f)

    existing_ids = set(existing_graphs.keys())
    id_to_pkd = {e['pdb_id']: e['pKd'] for e in all_entries}
    # Process lowest pKd first to maximize distribution widening early
    missing_sorted = sorted(set(raw_data.keys()) - existing_ids,
                           key=lambda x: id_to_pkd.get(x, 99))

    print(f"Building graphs for {len(missing_sorted)} missing complexes")
    pkds = [id_to_pkd.get(c, 99) for c in missing_sorted]
    print(f"pKd range: {min(pkds):.2f} - {max(pkds):.2f}")

    config = {"d_atom": 4.0, "d_res": 8.0, "d_sub": 8.0}
    success = {}
    failed = {}
    TIMEOUT = 45

    for i, cid in enumerate(missing_sorted):
        pKd = id_to_pkd.get(cid, '?')
        print(f"  [{i+1}/{len(missing_sorted)}] {cid} (pKd={pKd:.2f})...", end="", flush=True)

        # Temp file for IPC
        tmpf = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        out_path = tmpf.name
        tmpf.close()

        ctx = mp.get_context('fork')
        proc = ctx.Process(target=build_one_worker, args=(cid, config, out_path))
        proc.start()
        proc.join(timeout=TIMEOUT)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
            print(f" TIMEOUT")
            failed[cid] = "timeout"
        elif proc.exitcode == 0:
            try:
                with open(out_path, "rb") as f:
                    res_cid, graph, ok, msg = pickle.load(f)
                if ok and graph is not None:
                    # Ensure label is set
                    if 'label' not in graph or graph.get('label') is None:
                        graph['label'] = raw_data[cid].get('label')
                    success[cid] = graph
                    print(f" {msg}")
                else:
                    print(f" FAIL: {msg}")
                    failed[cid] = msg[:80]
            except Exception as e:
                print(f" READ_ERR: {e}")
                failed[cid] = f"read_error: {str(e)[:60]}"
        else:
            print(f" CRASH(exit={proc.exitcode})")
            failed[cid] = f"exitcode={proc.exitcode}"

        # Cleanup temp file
        try:
            os.unlink(out_path)
        except:
            pass

        # Save checkpoint every 10
        if (i + 1) % 10 == 0 and success:
            all_g = dict(existing_graphs)
            all_g.update(success)
            tmp = f"{CASE_DIR}/hiv_protease_graphs_full.pkl"
            with open(tmp, "wb") as f:
                pickle.dump(all_g, f)
            print(f"  --- checkpoint: {len(all_g)} total ({len(success)} new, {len(failed)} failed) ---")

    # Final merge and save
    all_graphs = dict(existing_graphs)
    all_graphs.update(success)

    for path in [f"{CASE_DIR}/hiv_protease_graphs_full.pkl",
                 f"{CASE_DIR}/hiv_protease_graphs.pkl"]:
        with open(path, "wb") as f:
            pickle.dump(all_graphs, f)

    import statistics
    all_pkds = [id_to_pkd[c] for c in all_graphs if c in id_to_pkd]
    print(f"\n{'='*60}")
    print(f"Done: {len(all_graphs)} total ({len(success)} new, {len(failed)} failed)")
    if all_pkds:
        print(f"pKd: {min(all_pkds):.2f}-{max(all_pkds):.2f}, "
              f"mean={sum(all_pkds)/len(all_pkds):.2f}, std={statistics.stdev(all_pkds):.2f}")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for c, r in sorted(failed.items(), key=lambda x: id_to_pkd.get(x[0], 99)):
            print(f"  {c}: pKd={id_to_pkd.get(c,'?'):.2f} - {r}")

    print(f"\nSuccess rate: {len(success)}/{len(missing_sorted)} "
          f"({100*len(success)/max(1,len(missing_sorted)):.1f}%)")


if __name__ == "__main__":
    main()
