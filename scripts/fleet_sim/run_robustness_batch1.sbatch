#!/bin/bash
#SBATCH --job-name=eq_robustness1
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_robust1_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_robust1_%j.err

set -uo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/opt/cray/pe/python/3.11.7/bin/python
LOGDIR=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp

$PY scripts/fleet_sim/run_equilibrium_search.py --label initcheck50 --initial-wait-min 0.3610 --kappa-w 1.5 > "$LOGDIR/eqsearch_initcheck50.log" 2>&1 &
$PY scripts/fleet_sim/run_equilibrium_search.py --label initcheck150 --initial-wait-min 8.9695 --kappa-w 1.5 > "$LOGDIR/eqsearch_initcheck150.log" 2>&1 &
$PY scripts/fleet_sim/run_equilibrium_search.py --label kappa10 --initial-wait-min 0.0 --kappa-w 1.0 > "$LOGDIR/eqsearch_kappa10.log" 2>&1 &
$PY scripts/fleet_sim/run_equilibrium_search.py --label kappa20 --initial-wait-min 0.0 --kappa-w 2.0 > "$LOGDIR/eqsearch_kappa20.log" 2>&1 &
wait
echo "ALL_ROBUSTNESS_BATCH1_DONE"
