#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmdcg_dta_env
export LD_LIBRARY_PATH=/root/anaconda3/envs/mmdcg_dta_env/lib:/root/anaconda3/envs/mmdcg_dta_env/lib/python3.9/site-packages/torch/lib:/root/anaconda3/envs/mmdcg_dta_env/lib/python3.9/site-packages/dgl:$LD_LIBRARY_PATH
export DGLBACKEND=pytorch
export PYTHONUNBUFFERED=1
cd /root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study
echo "Pipeline started at $(date)"
python3 -u run_full_pipeline.py 2>&1 | tee -a pipeline.log
echo "Pipeline finished at $(date)"
