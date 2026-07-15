#!/bin/bash
set -e
source /root/anaconda3/etc/profile.d/conda.sh
conda activate mmdcg_dta_env
export LD_LIBRARY_PATH=/root/anaconda3/envs/mmdcg_dta_env/lib:/root/anaconda3/envs/mmdcg_dta_env/lib/python3.9/site-packages/torch/lib:/root/anaconda3/envs/mmdcg_dta_env/lib/python3.9/site-packages/dgl:$LD_LIBRARY_PATH
export DGLBACKEND=pytorch
export CUDA_VISIBLE_DEVICES=1
cd /root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print(f'Using GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')"
echo "Running full pipeline..."
python run_full_pipeline.py 2>&1 | tee hiv_protease_training_log_new.txt
echo "PIPELINE DONE"
