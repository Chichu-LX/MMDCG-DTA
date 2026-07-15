#!/bin/bash
set -e
source /root/anaconda3/etc/profile.d/conda.sh
conda activate mmdcg_dta_env
export LD_LIBRARY_PATH=/root/anaconda3/envs/mmdcg_dta_env/lib:/root/anaconda3/envs/mmdcg_dta_env/lib/python3.9/site-packages/torch/lib:/root/anaconda3/envs/mmdcg_dta_env/lib/python3.9/site-packages/dgl:$LD_LIBRARY_PATH
export DGLBACKEND=pytorch
export CUDA_VISIBLE_DEVICES=1
cd /root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study
echo "Running on GPU:"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
echo ""
echo "Running full pipeline..."
rm -f hiv_protease_training_log_new.txt
python run_full_pipeline.py 2>&1 | tee hiv_protease_training_log_new.txt
echo ""
echo "PIPELINE COMPLETE"
