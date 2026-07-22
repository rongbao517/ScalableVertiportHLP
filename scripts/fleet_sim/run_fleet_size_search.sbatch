#!/bin/bash
#SBATCH --job-name=fleet_size_search
#SBATCH --partition=workq
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_fleetsize_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_fleetsize_%j.err

set -euo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/opt/cray/pe/python/3.11.7/bin/python
LOGDIR=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp
OUT=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim

run_one () {
  local vpv=$1
  local tag="fleetsearch_logit30_vpv${vpv}"
  if [ -f "$OUT/run_summary_${tag}.csv" ]; then
    echo "SKIP $tag (already done)"
    return
  fi
  $PY scripts/fleet_sim/run_shanghai_fleet_simulation.py \
    --demand-source full_dynspeed \
    --demand-bucket-csv outputs/fleet_sim_dynspeed_full_demand_bucket_modechoice_scale30.csv \
    --n-bins 500 \
    --vehicles-per-vertiport "$vpv" \
    --discharge-rate 1.0 \
    --charging-rate-per-bin 25 \
    --rebalance-interval 1 \
    --predictive-rebalancing \
    --skip-charging-log \
    --tag "$tag" > "$LOGDIR/${tag}.log" 2>&1
  echo "DONE $tag exit=$?"
}

# vpv=700 already exists under tag "compare_logit30" (identical config) -- reused directly,
# not rerun here, to avoid duplicating a 500-bin run for no reason.
for vpv in 5 10 20 35 50 75 100 150 250; do
  run_one "$vpv" &
done
wait
echo "ALL_FLEET_SIZE_SEARCH_DONE"
