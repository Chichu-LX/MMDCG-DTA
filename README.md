# MMDCG-DTA

Official implementation of **MMDCG-DTA: Molecular Mechanics-Informed Dynamic
Contact Graph Framework for Predicting Protein-Ligand Binding Affinity**.

MMDCG-DTA combines molecular-mechanics priors with dynamically reconstructed
protein-ligand contact graphs. It learns coupled atom-level and
substructure-level representations through a three-stage training strategy.

## Repository Contents

- `Model/MMDCG_DTA.py`: core MMDCG-DTA model.
- `Data/MMDCG_DTA_Stage1.py`: molecular-mechanics-informed representation learning.
- `Data/MMDCG_DTA_Stage2.py`: dynamic interaction-edge reconstruction.
- `Data/MMDCG_DTA_Stage3.py`: final affinity-prediction fine-tuning.
- `Data/`: graph construction, featurization, staged model code, and training utilities.
- `Utils/`: evaluation metrics.
- `train.py`: training entry point.

Large datasets, graph caches, trained checkpoints, logs, virtual environments,
and local server artifacts are intentionally excluded from this open-source
release.

## Installation

Create a Python environment, then install the dependencies:

```bash
python -m venv mmdcg_dta_env
source mmdcg_dta_env/bin/activate
pip install -r requirements.txt
```

The original experiments used CUDA-enabled PyTorch and DGL versions listed in
`requirements.txt`. Install the matching wheels for your CUDA environment when
needed.

## Data Preparation

Place the PDBbind data under:

```text
Data/PDBbind_dataset/
```

Then build graph datasets:

```bash
cd Data
python build_graph_dataset.py
```

This generates graph cache files such as `refined_set_graphs.pkl` and
`core_set_graphs.pkl`. These files are not committed because they are generated
artifacts.

## Training

From the repository root:

```bash
python train.py
```

Training configuration is stored in `default.yaml`.

## License

This project is released under the MIT License.
