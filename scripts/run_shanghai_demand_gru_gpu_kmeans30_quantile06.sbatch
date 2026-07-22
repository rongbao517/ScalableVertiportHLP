#!/bin/bash
#SBATCH --job-name=shanghai_gru_kmeans30_quantile06
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_gru_kmeans30_quantile06_%j.out
#SBATCH --error=slurm_gru_kmeans30_quantile06_%j.err

set -euo pipefail
cd '/home/b5by/zhirong.b5by/work/shanghai_new/scripts'
export MPLBACKEND=Agg
export OMP_NUM_THREADS=8

/opt/cray/pe/python/3.11.7/bin/python -u train_shanghai_demand_gru.py \
    --data shanghai_demand_windows_kmeans30.npz \
    --loss quantile \
    --tau 0.6 \
    --epoch 100 \
    --patience 15 \
    --batchsize 64 \
    --lr 1e-3 \
    --hidden 64 \
    --layers 1 \
    --seed 100
