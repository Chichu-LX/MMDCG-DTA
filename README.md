# MMDCG-DTA

Official implementation of **A Novel Molecular Mechanics-Informed Semantic
Fusion Framework via Dynamic Contact Graph for Predicting Drug-Target Binding
Affinity** (MMDCG-DTA).

The public entry point implements the complete three-stage curriculum described
in the paper:

1. Stage 1 learns the affinity backbone from the static 4 Å atom-contact and
   8 Å fragment–residue graphs. Bond, angle, and torsion surrogates jointly
   gate every intramolecular GAT layer, while van der Waals, electrostatic,
   and hydrogen-bond surrogates jointly gate intermolecular messages.
2. Stage 2 expands atom candidates to 8 Å and alternately trains independent
   atom and substructure Remove/Keep/Add reconstructors and the affinity model.
   Each physical contact is supervised once even though DGL represents it with
   two oppositely directed message-passing edges.
3. Stage 3 freezes both contact-scoring MLPs, recomputes their input-dependent
   soft weights on every forward pass, and fine-tunes the affinity network.

Only the main graph-construction, model, training, and test code is included.
Case-study, virtual-screening, and binding-site-analysis scripts are intentionally
excluded.

## Installation

Python 3.9 was used for the released dependency set.

```bash
python -m venv mmdcg_dta_env
source mmdcg_dta_env/bin/activate
python -m pip install -r requirements.txt
```

For CUDA training, install the PyTorch build appropriate for the local CUDA
driver before installing the remaining requirements. DGL 1.1.2 is used because
it is available from the standard Python package index; CUDA-specific DGL wheels
may instead be installed from DGL's official wheel repository.

## Data preparation

PDBbind can be obtained from the official
[PDBbind+ portal](https://www.pdbbind-plus.org.cn/). The dataset is not bundled
with this repository. Project data are available from the
[MMDCG-DTA OneDrive data folder](https://onedrive.live.com/my?id=%2Fpersonal%2Fdde9bb07b5712251%2FDocuments%2FMMDCG%2DDTA&sortField=LinkFilename&isAscending=true&viewid=7768cdb2%2Dc0d4%2D4926%2Da43e%2D9f5a840826e5).
Please ensure that all use complies with the PDBbind license and access terms.

Place PDBbind data in this layout:

```text
Data/PDBbind_dataset/
├── core-set/
└── refined-set/
```

Place the PDBbind affinity index at `Data/INDEX_data.2016`, then run from the
repository root:

```bash
python -m Data.loader
python -m Data.build_graph_dataset
```

The graph builder writes exact BRICS atom-to-fragment and PDB atom-to-residue
memberships into the cache. It also writes both `atom_interaction_graph` (4 Å)
and `atom_candidate_graph` (8 Å). Caches produced by older releases must be
rebuilt; the training pipeline deliberately does not infer hierarchy memberships
with geometric K-means.

## Training and evaluation

Run all three stages:

```bash
python train.py
```

The v2.1 intramolecular gates and 64-dimensional readout change checkpoint
shapes relative to earlier releases. Retrain from Stage 1 rather than loading
an older Stage-1/2/3 checkpoint.

Or resume an individual stage after its predecessor's checkpoint exists:

```bash
python train.py --stage 2
python train.py --stage 3
```

All paper-level architecture and optimization settings are centralized in
`default.yaml`: batch size 16, 64-dimensional hidden states, two layers/steps,
Stage 1/2/3 learning rates of `1e-4`, `5e-5`, and `1e-4`, 32-dimensional
covalent and 64-dimensional non-covalent MLPs, and at most five Stage-2 inner
iterations with early termination when the topology-change ratio is at or below
0.01.

The Refined cache is split deterministically into training and validation data,
after excluding every complex present in the Core cache. Early stopping and
checkpoint selection use validation RMSE only. The Core set is evaluated once
after the best Stage-3 checkpoint is restored, and the results are written to
`results/training/core_test_metrics.json`.

## Verification

The tests exercise exact atom-to-substructure mappings, 4/8 Å graph nesting,
all three intramolecular gate channels, pair-level reconstruction, group-local
cross-hierarchy broadcast, independent hierarchy/channel parameters, and
Stage-3 reconstructor freezing with upstream gradient flow:

```bash
python -m unittest discover -s tests -v
```

## License

This project is released under the MIT License.
