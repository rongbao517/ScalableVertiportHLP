#!/bin/bash
#SBATCH --job-name=eq_fleetsize
#SBATCH --partition=workq
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_fleetsize_eq_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_fleetsize_eq_%j.err

set -uo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/opt/cray/pe/python/3.11.7/bin/python
LOGDIR=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp

$PY scripts/fleet_sim/run_equilibrium_search.py --label fleetbelow --initial-wait-min 0.0 --kappa-w 1.5 --vehicles-per-vertiport 150 > "$LOGDIR/eqsearch_fleetbelow.log" 2>&1 &
$PY scripts/fleet_sim/run_equilibrium_search.py --label fleetabove --initial-wait-min 0.0 --kappa-w 1.5 --vehicles-per-vertiport 400 > "$LOGDIR/eqsearch_fleetabove.log" 2>&1 &
wait
echo "ALL_FLEETSIZE_EQ_DONE"
