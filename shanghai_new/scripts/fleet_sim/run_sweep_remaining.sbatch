#!/bin/bash
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
