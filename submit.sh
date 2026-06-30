#!/bin/bash
#SBATCH --job-name=vertiport_hlp
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/ScalableVertiportHLP/logs/hlp_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/ScalableVertiportHLP/logs/hlp_%j.err

WORKDIR="/home/b5by/zhirong.b5by/work/ScalableVertiportHLP"
CODE_DIR="${WORKDIR}/code"
PYTHON="/home/b5by/zhirong.b5by/work/LLM-EventToConstraint-main/.venv/bin/python"

echo "========================================"
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : $(hostname)"
echo "Start    : $(date)"
echo "========================================"

cd "${CODE_DIR}"

# --- Step 1: run_experiment.py (Table5 reproduction + hub_coordinates.csv) ---
echo ""
echo ">>> [Step 1] run_experiment.py"
echo "----------------------------------------"
$PYTHON run_experiment.py
EXIT1=$?
echo "----------------------------------------"
echo "run_experiment.py exit code: ${EXIT1}"

# --- Step 2: visualize_hubs.py (p=10 maps for n_b=15,20,25) ---
echo ""
echo ">>> [Step 2] visualize_hubs.py"
echo "----------------------------------------"
$PYTHON visualize_hubs.py
EXIT2=$?
echo "----------------------------------------"
echo "visualize_hubs.py exit code: ${EXIT2}"

echo ""
echo "========================================"
echo "End: $(date)"
if [ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ]; then
    echo "All steps completed successfully."
else
    echo "WARNING: one or more steps failed (exit codes: ${EXIT1}, ${EXIT2})."
fi
echo "========================================"
