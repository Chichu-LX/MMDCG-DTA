#!/bin/bash
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmdcg_dta_env
export LD_LIBRARY_PATH=/root/anaconda3/envs/mmdcg_dta_env/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "============================================"
echo "MMDCG-DTA Complete Pipeline Run"
echo "Started at: $(date)"
echo "============================================"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
python train.py 2>&1 | tee training_output.log

echo ""
echo "============================================"
echo "Pipeline completed at: $(date)"
echo "============================================"
