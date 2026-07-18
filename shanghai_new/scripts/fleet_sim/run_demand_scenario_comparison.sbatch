#!/bin/bash
#SBATCH --job-name=fleet_demand_compare
#SBATCH --partition=workq
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_compare_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_compare_%j.err

set -euo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/opt/cray/pe/python/3.11.7/bin/python
LOGDIR=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp
OUT=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim

run_one () {
  local bucket_csv=$1
  local tag=$2
  if [ -f "$OUT/run_summary_${tag}.csv" ]; then
    echo "SKIP $tag (already done)"
    return
  fi
  $PY scripts/fleet_sim/run_shanghai_fleet_simulation.py \
    --demand-source full_dynspeed \
    --demand-bucket-csv "$bucket_csv" \
    --n-bins 500 \
    --vehicles-per-vertiport 700 \
    --discharge-rate 1.0 \
    --charging-rate-per-bin 25 \
    --rebalance-interval 1 \
    --predictive-rebalancing \
    --skip-charging-log \
    --tag "$tag" > "$LOGDIR/${tag}.log" 2>&1
  echo "DONE $tag exit=$?"
}

run_one "outputs/fleet_sim_dynspeed_full_demand_bucket.csv" "compare_fulldemand" &
run_one "outputs/fleet_sim_dynspeed_full_demand_bucket_realistic_filtered.csv" "compare_threshold" &
run_one "outputs/fleet_sim_dynspeed_full_demand_bucket_modechoice_scale10.csv" "compare_logit10" &
run_one "outputs/fleet_sim_dynspeed_full_demand_bucket_modechoice_scale30.csv" "compare_logit30" &
run_one "outputs/fleet_sim_dynspeed_full_demand_bucket_modechoice_scale60.csv" "compare_logit60" &
run_one "outputs/fleet_sim_dynspeed_full_demand_bucket_modechoice_scale100.csv" "compare_logit100" &
wait
echo "ALL_DEMAND_SCENARIO_COMPARISON_DONE"
