#!/bin/bash
#SBATCH --job-name=shanghai_od_astt
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_od_astt_%j.out
#SBATCH --error=slurm_od_astt_%j.err

set -euo pipefail
cd '/home/b5by/zhirong.b5by/work/shanghai_new/scripts/od_astt'
export MPLBACKEND=Agg

/opt/cray/pe/python/3.11.7/bin/python -u train_shanghai_od_astt.py \
    --epoch 200 \
    --patience 30 \
    --batchsize 32 \
    --lr 1e-3 \
    --num_blocks 1 \
    --num_heads 4 \
    --head_dim 8 \
    --dropout 0.2 \
    --seed 100
