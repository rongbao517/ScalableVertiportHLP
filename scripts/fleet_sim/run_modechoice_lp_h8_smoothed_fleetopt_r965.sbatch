#!/bin/bash
# Locally smoothed root search for five fleet candidates; all outputs are new.
#SBATCH --job-name=mc_lp_h8_smroot
#SBATCH --partition=workq
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-4%3
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_smroot_%A_%a.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_smroot_%A_%a.err
set -euo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
FLEETS=(250 260 265 270 275)
VPV=${FLEETS[$SLURM_ARRAY_TASK_ID]}
/usr/bin/python3.11 scripts/fleet_sim/run_equilibrium_smoothed_root_search.py \
  --label "lp_mc_h8_smroot_r965_vpv${VPV}_n1000" \
  --sites-csv data/selected_sites_kmeans_K30_modechoice_iter1.csv \
  --routed-csv outputs/route_assignment_layout_modechoice_iter1.csv \
  --vehicles-per-vertiport "$VPV" --rebalance-mechanism lp --positioning-lp-horizon 8 \
  --n-bins 1000 --smooth-radius-min 0.02 --wait-tol-min 0.02 --bracket-tol-min 0.02 \
  --min-service-rate 0.965 --max-wait-min 5.0 --max-bisection-steps 8
