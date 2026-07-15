#!/bin/bash
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmdcg_dta_env
export LD_LIBRARY_PATH=/root/anaconda3/envs/mmdcg_dta_env/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "============================================"
echo "MMDCG-DTA Complete Pipeline Run"
echo "Started at: $(date)"
echo "============================================"

# Part 1: Virtual Screening
echo ""
echo "########## PART 1: VIRTUAL SCREENING ##########"
cd ~/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/virtual_screening
python run_virtual_screening.py 2>&1 | tee vs_output.log
echo "VS exit code: $?"

# Part 2: Generate VS figures
python plot_vs_figures.py 2>&1 | tee vs_figures.log
echo "VS figures exit code: $?"

# Part 3: Case Study
echo ""
echo "########## PART 2: CASE STUDY ##########"
cd ~/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study
python run_case_study.py 2>&1 | tee cs_output.log
echo "CS exit code: $?"

# Part 4: Generate CS figures
python visualize_results.py 2>&1 | tee cs_figures.log
echo "CS figures exit code: $?"

echo ""
echo "============================================"
echo "Pipeline completed at: $(date)"
echo "============================================"
