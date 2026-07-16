#!/bin/bash
#SBATCH --job-name=shanghai_gru
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_gru_%j.out
#SBATCH --error=slurm_gru_%j.err

set -euo pipefail
cd '/home/b5by/zhirong.b5by/work/shanghai_new/scripts'
export MPLBACKEND=Agg
export OMP_NUM_THREADS=8

/opt/cray/pe/python/3.11.7/bin/python -u train_shanghai_demand_gru.py \
    --epoch 100 \
    --patience 15 \
    --batchsize 64 \
    --lr 1e-3 \
    --hidden 64 \
    --layers 1
