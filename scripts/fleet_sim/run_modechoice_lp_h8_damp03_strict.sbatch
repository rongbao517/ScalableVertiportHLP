#!/bin/bash
# Strict, isolated D-W-R equilibrium validation.
#SBATCH --job-name=mc_lp_h8_d03dwr
#SBATCH --partition=workq
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=04:30:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_d03dwr_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_d03dwr_%j.err

set -euo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

/usr/bin/python3.11 scripts/fleet_sim/run_equilibrium_search.py \
  --label lp_mc_h8_damp03_dwr_strict \
  --sites-csv data/selected_sites_kmeans_K30_modechoice_iter1.csv \
  --routed-csv outputs/route_assignment_layout_modechoice_iter1.csv \
  --vehicles-per-vertiport 250 \
  --rebalance-mechanism lp \
  --positioning-lp-horizon 8 \
  --n-bins 500 \
  --damping 0.3 \
  --max-iters 30 \
  --w-thresh-pct 2 \
  --r-thresh-pp 0.2 \
  --consecutive-needed 4
