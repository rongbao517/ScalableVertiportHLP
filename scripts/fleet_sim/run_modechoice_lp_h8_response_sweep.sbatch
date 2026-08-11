#!/bin/bash
# Five independent fixed-W probes run concurrently as a Slurm array.
#SBATCH --job-name=mc_lp_h8_response
#SBATCH --partition=workq
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --array=0-4
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_response_%A_%a.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_mc_lp_h8_response_%A_%a.err
set -euo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WAITS=(0 0.5 1.0 1.5 2.0)
SUFFIXES=(w000 w050 w100 w150 w200)
IDX=$SLURM_ARRAY_TASK_ID
/usr/bin/python3.11 scripts/fleet_sim/run_equilibrium_probe.py \
  --label "lp_mc_h8_response_${SUFFIXES[$IDX]}" \
  --wait-input-min "${WAITS[$IDX]}" \
  --sites-csv data/selected_sites_kmeans_K30_modechoice_iter1.csv \
  --routed-csv outputs/route_assignment_layout_modechoice_iter1.csv \
  --vehicles-per-vertiport 250 --rebalance-mechanism lp \
  --positioning-lp-horizon 8 --n-bins 500
