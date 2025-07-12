#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128GB
#SBATCH --time=250:00:00
#SBATCH --partition=batch

export PYTHONUNBUFFERED=1

python train.py

