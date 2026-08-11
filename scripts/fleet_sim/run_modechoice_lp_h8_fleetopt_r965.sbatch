#!/bin/bash
# Minimum feasible fleet search: strict convergence, W<=5 min, and R>=96.5% are enforced.
#SBATCH --job-name=mc_lp_h8_r965
#SBATCH --partition=workq
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%3
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_r965_%A_%a.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_r965_%A_%a.err

set -euo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
FLEETS=(275 300 350 400)
VPV=${FLEETS[$SLURM_ARRAY_TASK_ID]}

/usr/bin/python3.11 scripts/fleet_sim/run_equilibrium_search.py \
  --label "lp_mc_h8_fleetopt_r965_vpv${VPV}_n1000" \
  --sites-csv data/selected_sites_kmeans_K30_modechoice_iter1.csv \
  --routed-csv outputs/route_assignment_layout_modechoice_iter1.csv \
  --vehicles-per-vertiport "$VPV" \
  --rebalance-mechanism lp --positioning-lp-horizon 8 --n-bins 1000 \
  --damping 0.2 --max-iters 40 \
  --d-thresh-pct 1.0 --w-thresh-pct 5.0 --w-residual-min 0.05 \
  --r-thresh-pp 0.05 --min-service-rate 0.965 --max-wait-min 5.0 --consecutive-needed 3
