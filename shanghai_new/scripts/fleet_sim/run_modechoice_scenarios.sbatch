#!/bin/bash
#SBATCH --job-name=fleet_modechoice
#SBATCH --partition=workq
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_modechoice_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_modechoice_%j.err

set -euo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/opt/cray/pe/python/3.11.7/bin/python
LOGDIR=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp
OUT=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim

for s in 10 60 100; do
  tag="modechoice_scale${s}_fleet700_rebal1_predictive_fastcharge25"
  if [ -f "$OUT/run_summary_${tag}.csv" ]; then
    echo "SKIP $tag (already done)"
    continue
  fi
  (
    $PY scripts/fleet_sim/run_shanghai_fleet_simulation.py \
      --demand-source full_dynspeed \
      --demand-bucket-csv outputs/fleet_sim_dynspeed_full_demand_bucket_modechoice_scale${s}.csv \
      --n-bins 500 \
      --vehicles-per-vertiport 700 \
      --discharge-rate 1.0 \
      --charging-rate-per-bin 25 \
      --rebalance-interval 1 \
      --predictive-rebalancing \
      --skip-charging-log \
      --tag "$tag" > "$LOGDIR/${tag}.log" 2>&1
    echo "DONE $tag exit=$?"
  ) &
done
wait
echo "ALL_MODECHOICE_SCENARIOS_DONE"
