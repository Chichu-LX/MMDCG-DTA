#!/bin/bash
# ============================================================================
# MMDCG-DTA Binding Site Validation - Server Runner
# ============================================================================
# Run on server: ~/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/
# Env: /root/anaconda3/envs/mmdcg_dta_env

set -e

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/root/anaconda3/envs/mmdcg_dta_env/lib:$LD_LIBRARY_PATH

PROJ_DIR=~/protein_ligand/MMDCG-DTA/MMDCG-DTA-main
cd $PROJ_DIR

echo "============================================"
echo "MMDCG-DTA Binding Site Validation Pipeline"
echo "============================================"
echo "Project dir: $PROJ_DIR"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Check prerequisites
echo ""
echo "[Check] Data files..."
ls -la case_study/hiv_protease_graphs.pkl 2>/dev/null && echo "  [OK] Graph data" || echo "  [MISSING] Graph data"
ls -la case_study/hiv_protease_best_model.pth 2>/dev/null && echo "  [OK] Fine-tuned model" || echo "  [MISSING] Fine-tuned model"
ls -la Data/stage1_model_final.pth 2>/dev/null && echo "  [OK] Pretrained model" || echo "  [MISSING] Pretrained model"

# Install optional dependencies
echo ""
echo "[Setup] Installing optional dependencies..."
pip install py3dmol biopython requests -q 2>/dev/null || true

# Try to install PyMOL for 3D rendering
echo "[Setup] Attempting PyMOL installation..."
conda install -c conda-forge pymol-open-source -y 2>/dev/null && echo "  [OK] PyMOL installed" || echo "  [WARN] PyMOL not available, will use py3Dmol"

# Run the pipeline
echo ""
echo "[Run] Starting binding validation pipeline..."
echo "============================================"

python -u binding_site_validation/binding_validation_pipeline.py \
    2>&1 | tee binding_site_validation/pipeline_output.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================"
echo "Pipeline exit code: $EXIT_CODE"

# If pipeline succeeds, run the analysis-only mode for additional figures
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "[Extra] Running analysis-only mode for supplementary figures..."
    python -u binding_site_validation/binding_validation_pipeline.py \
        --analysis-only 2>&1 | tee -a binding_site_validation/pipeline_output.log
fi

# Generate the report
echo ""
echo "[Report] Generating validation report..."
python -u binding_site_validation/generate_report.py \
    2>&1 | tee -a binding_site_validation/pipeline_output.log

echo ""
echo "============================================"
echo "Binding Site Validation Complete!"
echo "Results in: binding_site_validation/results/"
echo "Figures in: binding_site_validation/figures/"
echo "============================================"
