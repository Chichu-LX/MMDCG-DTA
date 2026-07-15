#!/bin/bash
export LD_LIBRARY_PATH=/root/anaconda3/envs/mmdcg_dta_env/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=3
source /root/anaconda3/etc/profile.d/conda.sh
conda activate /root/anaconda3/envs/mmdcg_dta_env
cd /root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study
python run_multistage_pipeline.py
