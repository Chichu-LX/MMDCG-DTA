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
[MMDCG-DTA OneDrive data folder](https://1drv.ms/f/c/dde9bb07b5712251/IgAYH4ssKiugQrHlqcRl0iYXATfNGDOfj7cRSyzFuYDVZbo).
Please ensure that all use complies with the PDBbind license and access terms.
