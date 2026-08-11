#!/bin/bash
# High wait-aversion sensitivity: kappa_w=2.0.
#SBATCH --job-name=mc_lp_root_k20
#SBATCH --partition=workq
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=04:30:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_root_k20_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_root_k20_%j.err
set -euo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/usr/bin/python3.11 scripts/fleet_sim/run_equilibrium_root_search.py \
  --label lp_mc_h8_bisect_kappa20 \
  --sites-csv data/selected_sites_kmeans_K30_modechoice_iter1.csv \
  --routed-csv outputs/route_assignment_layout_modechoice_iter1.csv \
  --kappa-w 2.0 --vehicles-per-vertiport 250 --rebalance-mechanism lp \
  --positioning-lp-horizon 8 --n-bins 500 \
  --wait-low-min 0 --wait-high-min 2 --wait-tol-min 0.01 \
  --bracket-tol-min 0.02 --max-bisection-steps 8
