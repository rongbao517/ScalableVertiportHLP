#!/bin/bash
# Continue three additional dense-smoothed iterations without overwriting the ten-round trajectory.
#SBATCH --job-name=mc_lp_h8_d5cont
#SBATCH --partition=workq
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_dense5_continue_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_dense5_continue_%j.err
set -euo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/usr/bin/python3.11 scripts/fleet_sim/run_equilibrium_smoothed_iteration.py \
  --label lp_mc_h8_dense5_r965_vpv275_continue3_n1000 \
  --resume-trajectory outputs/fleet_sim/smoothed_eqsearch_trajectory_lp_mc_h8_dense5_r965_vpv275_n1000.csv \
  --vehicles-per-vertiport 275 \
  --sites-csv data/selected_sites_kmeans_K30_modechoice_iter1.csv \
  --routed-csv outputs/route_assignment_layout_modechoice_iter1.csv \
  --rebalance-mechanism lp --positioning-lp-horizon 8 --n-bins 1000 \
  --smooth-radius-min 0.04 --smooth-points 5 --damping 0.3 --max-iters 3 --min-new-iters 3 \
  --d-thresh-pct 1.0 --w-residual-min 0.06 --r-thresh-pp 0.15 \
  --min-service-rate 0.965 --max-wait-min 5.0 --consecutive-needed 3
