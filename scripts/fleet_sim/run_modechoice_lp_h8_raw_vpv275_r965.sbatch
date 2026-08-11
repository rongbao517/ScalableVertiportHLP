#!/bin/bash
# Original (unsmoothed) 275-vehicle/vertiport strict iteration for raw D-W-R figures.
#SBATCH --job-name=mc_lp_h8_raw275
#SBATCH --partition=workq
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_raw275_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_raw275_%j.err
set -euo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/usr/bin/python3.11 scripts/fleet_sim/run_equilibrium_search.py \
  --label lp_mc_h8_raw_vpv275_r965_n1000 \
  --sites-csv data/selected_sites_kmeans_K30_modechoice_iter1.csv \
  --routed-csv outputs/route_assignment_layout_modechoice_iter1.csv \
  --vehicles-per-vertiport 275 --rebalance-mechanism lp --positioning-lp-horizon 8 \
  --n-bins 1000 --damping 0.2 --max-iters 40 \
  --d-thresh-pct 1.0 --w-thresh-pct 5.0 --w-residual-min 0.05 \
  --r-thresh-pp 0.05 --min-service-rate 0.965 --max-wait-min 5.0 --consecutive-needed 3
