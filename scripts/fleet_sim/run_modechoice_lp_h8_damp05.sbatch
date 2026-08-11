#!/bin/bash
# Damped demand--operations fixed-point search.  The preceding undamped
# 20-iteration diagnostic met the local threshold but W subsequently cycled.
# alpha=0.5 under-relaxes the wait-time feedback to suppress that overshoot.
#SBATCH --job-name=mc_lp_h8_damp05
#SBATCH --partition=workq
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=04:30:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_damp05_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_damp05_%j.err

set -euo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

/usr/bin/python3.11 scripts/fleet_sim/run_equilibrium_search.py \
  --label lp_mc_h8_damp05 \
  --sites-csv data/selected_sites_kmeans_K30_modechoice_iter1.csv \
  --routed-csv outputs/route_assignment_layout_modechoice_iter1.csv \
  --vehicles-per-vertiport 250 \
  --rebalance-mechanism lp \
  --positioning-lp-horizon 8 \
  --n-bins 500 \
  --damping 0.5 \
  --max-iters 30 \
  --consecutive-needed 3
