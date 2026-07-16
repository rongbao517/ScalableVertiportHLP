#!/bin/bash
#SBATCH --job-name=stid_hparam_search
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_stid_hpsearch_%j.out
#SBATCH --error=slurm_stid_hpsearch_%j.err

set -euo pipefail
cd '/home/b5by/zhirong.b5by/work/shanghai_new/scripts'
export MPLBACKEND=Agg
export OMP_NUM_THREADS=8

/opt/cray/pe/python/3.11.7/bin/python -u hparam_search_stid.py \
    --data shanghai_demand_windows_kmeans30.npz \
    --n-trials 40 \
    --epochs 100 \
    --patience 15 \
    --seed 100 \
    --search-seed 0
