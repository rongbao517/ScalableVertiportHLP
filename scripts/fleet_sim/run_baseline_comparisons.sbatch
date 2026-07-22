#!/bin/bash
#SBATCH --job-name=fleet_baseline_compare
#SBATCH --partition=workq
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_baseline_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_baseline_%j.err

set -euo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/opt/cray/pe/python/3.11.7/bin/python
LOGDIR=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp
OUT=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim

run_one () {
  local sites_csv=$1
  local bucket_csv=$2
  local vpv=$3
  local tag=$4
  if [ -f "$OUT/run_summary_${tag}.csv" ]; then
    echo "SKIP $tag (already done)"
    return
  fi
  $PY scripts/fleet_sim/run_shanghai_fleet_simulation.py \
    --demand-source full_dynspeed \
    --sites "$sites_csv" \
    --demand-bucket-csv "$bucket_csv" \
    --n-bins 500 --vehicles-per-vertiport "$vpv" --discharge-rate 1.0 --charging-rate-per-bin 25 \
    --rebalance-interval 1 --predictive-rebalancing --skip-charging-log \
    --tag "$tag" > "$LOGDIR/${tag}.log" 2>&1
  echo "DONE $tag exit=$?"
}

# NOTE (2026-07-20): these are legacy/deprecated candidate site sets, not part
# of the final frozen 3-layout comparison (see outputs/fleet_sim/_frozen_baseline_v1/MANIFEST.md).
# Archived to data/archive/legacy_layouts/ -- paths updated below, kept only for
# historical reproducibility of the already-completed runs this script drove.
PRED_SITES="data/archive/legacy_layouts/selected_sites_kmeans_K30_predicted_legacy.csv"
RAND_SITES="data/archive/legacy_layouts/selected_sites_kmeans_K30_random_baseline_legacy.csv"

run_one "$PRED_SITES" "outputs/fleet_sim_dynspeed_full_demand_bucket_predicted_sites_threshold.csv" 700 "predsited_threshold_vpv700" &
run_one "$PRED_SITES" "outputs/fleet_sim_dynspeed_full_demand_bucket_predicted_sites_scale10.csv"   700 "predsited_logit10_vpv700" &
run_one "$PRED_SITES" "outputs/fleet_sim_dynspeed_full_demand_bucket_predicted_sites_scale30.csv"   250 "predsited_logit30_vpv250" &
run_one "$PRED_SITES" "outputs/fleet_sim_dynspeed_full_demand_bucket_predicted_sites_scale60.csv"   700 "predsited_logit60_vpv700" &
run_one "$PRED_SITES" "outputs/fleet_sim_dynspeed_full_demand_bucket_predicted_sites_scale100.csv"  700 "predsited_logit100_vpv700" &

run_one "$RAND_SITES" "outputs/fleet_sim_dynspeed_full_demand_bucket_random_baseline_threshold.csv" 700 "randsited_threshold_vpv700" &
run_one "$RAND_SITES" "outputs/fleet_sim_dynspeed_full_demand_bucket_random_baseline_scale10.csv"   700 "randsited_logit10_vpv700" &
run_one "$RAND_SITES" "outputs/fleet_sim_dynspeed_full_demand_bucket_random_baseline_scale30.csv"   250 "randsited_logit30_vpv250" &
run_one "$RAND_SITES" "outputs/fleet_sim_dynspeed_full_demand_bucket_random_baseline_scale60.csv"   700 "randsited_logit60_vpv700" &
run_one "$RAND_SITES" "outputs/fleet_sim_dynspeed_full_demand_bucket_random_baseline_scale100.csv"  700 "randsited_logit100_vpv700" &

wait
echo "ALL_BASELINE_COMPARISONS_DONE"
