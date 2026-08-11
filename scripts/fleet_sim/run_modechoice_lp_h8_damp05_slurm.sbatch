#!/bin/bash
# Isolated CPU run: all outputs use a unique label and never overwrite the
# previous local or diagnostic equilibrium-search artifacts.
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
  --label lp_mc_h8_damp05_slurm \
  --sites-csv data/selected_sites_kmeans_K30_modechoice_iter1.csv \
  --routed-csv outputs/route_assignment_layout_modechoice_iter1.csv \
  --vehicles-per-vertiport 250 \
  --rebalance-mechanism lp \
  --positioning-lp-horizon 8 \
  --n-bins 500 \
  --damping 0.5 \
  --max-iters 30 \
  --consecutive-needed 3
