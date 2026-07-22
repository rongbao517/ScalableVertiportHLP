#!/bin/bash
#SBATCH --job-name=validate_donorfilter
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_validate_donorfilter_%j.out
#SBATCH --error=/home/b5by/zhirong.b5by/work/shanghai_new/outputs/fleet_sim/_sweep_tmp/slurm_validate_donorfilter_%j.err

set -uo pipefail
cd /home/b5by/zhirong.b5by/work/shanghai_new
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/opt/cray/pe/python/3.11.7/bin/python

$PY -c "
import sys
sys.path.insert(0, 'scripts/fleet_sim')
import sweep_predictive_mechanism as sw

sw.run_one_config('predsweep_r1_baseline_donorfilter',
    'data/selected_sites_kmeans_K30.csv',
    'outputs/fleet_sim/fleet_sim_dynspeed_full_demand_bucket_eqsearch_v2_fleetabove_iter5.csv',
    400, 500,
    predictive_threshold=1.5, predictive_gain=1.0, predictive_top_k=3, predictive_max_drain=3)
"
echo "VALIDATE_DONORFILTER_DONE"
