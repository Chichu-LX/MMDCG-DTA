#!/bin/bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate mmdcg_dta_env
export LD_LIBRARY_PATH=/root/anaconda3/envs/mmdcg_dta_env/lib:/root/anaconda3/envs/mmdcg_dta_env/lib/python3.9/site-packages/torch/lib:/root/anaconda3/envs/mmdcg_dta_env/lib/python3.9/site-packages/dgl:$LD_LIBRARY_PATH
export DGLBACKEND=pytorch
export CUDA_VISIBLE_DEVICES=4
cd /root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study
exec python3 -u run_full_pipeline.py
