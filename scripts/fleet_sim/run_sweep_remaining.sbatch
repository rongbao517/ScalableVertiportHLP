#!/bin/bash
# NOTE (2026-07-20): this charging/discharge-rate sweep (fleet=21000, single-shot
# fixed-demand fleet_sim runs) predates the demand-operations equilibrium framework
# and is not part of the frozen 3-layout comparison -- see
# outputs/fleet_sim/_frozen_baseline_v1/MANIFEST.md for the current results.
# Its outputs have been archived to
# outputs/archive/legacy_charging_sensitivity_fleet21000/. If rerun, this script
# will NOT see the archived files (the skip-check below only looks in $OUT) and
# will recompute everything from scratch.
#SBATCH --job-name=fleet_sweep_remain
#SBATCH --partition=workq
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_sweep_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_sweep_%j.err

set -euo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/opt/cray/pe/python/3.11.7/bin/python
LOGDIR=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp
OUT=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim

COMBOS="15:2.0 15:3.0 25:0.5 25:1.0 25:2.0 25:3.0 40:0.5 40:1.0 40:2.0 40:3.0"

for combo in $COMBOS; do
  c=${combo%%:*}
  d=${combo##*:}
  tag="sens_c${c}_d${d}"
  if [ -f "$OUT/run_summary_${tag}.csv" ]; then
    echo "SKIP $tag (already done)"
    continue
  fi
  (
    $PY scripts/fleet_sim/run_shanghai_fleet_simulation.py \
      --demand-source full_dynspeed \
      --n-bins 500 \
      --vehicles-per-vertiport 700 \
      --discharge-rate "$d" \
      --charging-rate-per-bin "$c" \
      --rebalance-interval 1 \
      --predictive-rebalancing \
      --skip-charging-log \
      --tag "$tag" > "$LOGDIR/${tag}.log" 2>&1
    echo "DONE $tag exit=$?"
  ) &
done
wait
echo "ALL_REMAINING_SWEEP_DONE"
