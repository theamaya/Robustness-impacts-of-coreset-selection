#!/bin/bash

# SBATCH --account=visualai    # Specify VisualAI
# SBATCH --nodes=1             # nodes requested
# SBATCH --ntasks=1            # tasks requested
# SBATCH --cpus-per-task=8     # Specify the number of CPUs your task will need.
# SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
# SBATCH --mem=16G             # memory 
# SBATCH --error=./slurm_err/%J.err
# SBATCH --output=./slurm_out/%J.out
# SBATCH -t 48:00:00           # time requested in hour:minute:second

cd ..

python eval.py --fraction $1 --dataset CXR8 --data_path /n/fs/visualai-scr/Data/CXR8 --workers 10 --optimizer SGD --selection $2 --model ResNet50 --lr 0.001 -sp $3 --balance False --pretrain True --linear_probe False --score_path $4 --policy $5 --score_pretrain True -wd $6 
