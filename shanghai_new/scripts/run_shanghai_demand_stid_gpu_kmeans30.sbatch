#!/bin/bash
#SBATCH --job-name=shanghai_stid_kmeans30
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_stid_kmeans30_%j.out
#SBATCH --error=slurm_stid_kmeans30_%j.err

set -euo pipefail
cd '/home/b5by/zhirong.b5by/work/shanghai_new/scripts'
export MPLBACKEND=Agg
export OMP_NUM_THREADS=8

/opt/cray/pe/python/3.11.7/bin/python -u train_shanghai_demand_stid.py \
    --data shanghai_demand_windows_kmeans30.npz \
    --epoch 100 \
    --patience 15 \
    --batchsize 64 \
    --lr 1e-3 \
    --hidden 64 \
    --embed-dim 16 \
    --n-layers 3 \
    --seed 100
