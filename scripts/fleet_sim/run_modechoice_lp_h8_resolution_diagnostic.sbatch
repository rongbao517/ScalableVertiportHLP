#!/bin/bash
# Nine isolated near-equilibrium probes: 3 waits x 3 temporal resolutions.
#SBATCH --job-name=mc_lp_h8_resdiag
#SBATCH --partition=workq
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --array=0-8%3
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_resdiag_%A_%a.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_resdiag_%A_%a.err

set -euo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

WAITS=(1.18 1.19 1.20 1.18 1.19 1.20 1.18 1.19 1.20)
WAIT_TAGS=(w118 w119 w120 w118 w119 w120 w118 w119 w120)
BINS=(500 500 500 1000 1000 1000 2000 2000 2000)
IDX=$SLURM_ARRAY_TASK_ID
LABEL="lp_mc_h8_resdiag_n${BINS[$IDX]}_${WAIT_TAGS[$IDX]}"

/usr/bin/python3.11 scripts/fleet_sim/run_equilibrium_probe.py \
  --label "$LABEL" \
  --wait-input-min "${WAITS[$IDX]}" \
  --sites-csv data/selected_sites_kmeans_K30_modechoice_iter1.csv \
  --routed-csv outputs/route_assignment_layout_modechoice_iter1.csv \
  --vehicles-per-vertiport 250 \
  --rebalance-mechanism lp \
  --positioning-lp-horizon 8 \
  --n-bins "${BINS[$IDX]}" \
  --positioning-lp-diagnostics
