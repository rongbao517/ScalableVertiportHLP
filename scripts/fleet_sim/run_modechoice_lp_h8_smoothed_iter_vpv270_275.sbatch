#!/bin/bash
# Local-smoothing strict D-W-R trajectories for paper-ready convergence figures.
#SBATCH --job-name=mc_lp_h8_smit
#SBATCH --partition=workq
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_smit_%A_%a.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_smit_%A_%a.err
set -euo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
FLEETS=(270 275)
VPV=${FLEETS[$SLURM_ARRAY_TASK_ID]}
/usr/bin/python3.11 scripts/fleet_sim/run_equilibrium_smoothed_iteration.py \
  --label "lp_mc_h8_smit_r965_vpv${VPV}_n1000" --vehicles-per-vertiport "$VPV" \
  --sites-csv data/selected_sites_kmeans_K30_modechoice_iter1.csv \
  --routed-csv outputs/route_assignment_layout_modechoice_iter1.csv \
  --rebalance-mechanism lp --positioning-lp-horizon 8 --n-bins 1000 \
  --smooth-radius-min 0.02 --damping 0.3 --max-iters 20 \
  --d-thresh-pct 1.0 --w-residual-min 0.05 --r-thresh-pp 0.05 \
  --min-service-rate 0.965 --max-wait-min 5.0 --consecutive-needed 3
