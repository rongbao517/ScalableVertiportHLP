#!/bin/bash
#SBATCH --job-name=diag_rebal_scope
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_diag_rebal_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_diag_rebal_%j.err

set -uo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/opt/cray/pe/python/3.11.7/bin/python

$PY scripts/fleet_sim/diagnose_rebalance_fix_scope.py \
  --sites-csv "$SITES_CSV" \
  --bucket-csv "$BUCKET_CSV" \
  --vehicles-per-vertiport "$VEHICLES_PER_VERTIPORT" \
  --tag "$TAG" \
  --out-csv "$OUT_CSV"
echo "DIAG_REBALANCE_SCOPE_DONE"
