#!/bin/bash
# HIV-1 Protease Case Study Pipeline
# Run on server with mmdcg_dta_env conda environment

set -e

# Activate conda and set up environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmdcg_dta_env

# Set library path for CUDA 11 compatibility
export LD_LIBRARY_PATH=/root/anaconda3/envs/mmdcg_dta_env/lib:/root/anaconda3/envs/mmdcg_dta_env/lib/python3.9/site-packages/torch/lib:/root/anaconda3/envs/mmdcg_dta_env/lib/python3.9/site-packages/dgl:$LD_LIBRARY_PATH

# Set DGL backend
export DGLBACKEND=pytorch

echo "============================================"
echo "MMDCG-DTA Case Study: HIV-1 Protease"
echo "============================================"
echo "Python: $(python --version)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"
python -c "import dgl; print('DGL:', dgl.__version__)"
echo "============================================"

# Change to case study directory
cd /root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study

# Step 1: Extract data (already done, but run for safety)
if [ ! -f "hiv_protease_raw.pkl" ]; then
    echo ""
    echo "[Step 1] Extracting HIV-1 Protease data..."
    python extract_hiv_protease_data.py
else
    echo ""
    echo "[Step 1] HIV-1 Protease data already extracted."
    python -c "import pickle; d=pickle.load(open('hiv_protease_raw.pkl','rb')); print(f'  {len(d)} complexes loaded')"
fi

# Step 2: Build graphs
if [ ! -f "hiv_protease_graphs.pkl" ]; then
    echo ""
    echo "[Step 2] Building graph dataset..."
    python build_hiv_graphs.py
else
    echo ""
    echo "[Step 2] Graphs already built."
    python -c "import pickle; d=pickle.load(open('hiv_protease_graphs.pkl','rb')); print(f'  {len(d)} graph complexes loaded')"
fi

# Step 3 & 4: Run full pipeline (training + inference + interpretability)
echo ""
echo "[Step 3 & 4] Running full pipeline..."
python run_full_pipeline.py

echo ""
echo "============================================"
echo "PIPELINE COMPLETE"
echo "============================================"
echo ""
echo "Output files:"
ls -lh hiv_protease_*.pkl hiv_protease_*.pth hiv_protease_*.json hiv_protease_*.txt 2>/dev/null || echo "(listing files...)"
ls -lh hiv_protease_* 2>/dev/null
