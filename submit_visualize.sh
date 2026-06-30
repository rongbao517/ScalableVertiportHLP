#!/bin/bash
#SBATCH --job-name=vertiport_viz
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/ScalableVertiportHLP/logs/viz_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/ScalableVertiportHLP/logs/viz_%j.err

WORKDIR="/home/b5by/zhirong.b5by/work/ScalableVertiportHLP"
CODE_DIR="${WORKDIR}/code"
PYTHON="/home/b5by/zhirong.b5by/work/LLM-EventToConstraint-main/.venv/bin/python"

echo "========================================"
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : $(hostname)"
echo "Start    : $(date)"
echo "========================================"

cd "${CODE_DIR}"

$PYTHON -u visualize_hubs.py

echo "========================================"
echo "End: $(date)"
echo "Exit code: $?"
echo "========================================"
