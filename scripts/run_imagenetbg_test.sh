#!/bin/bash

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
#SBATCH --mem=16G             # memory 
#SBATCH --error=./slurm_err/%J.err
#SBATCH --output=./slurm_out/%J.out
#SBATCH -t 168:00:00           # time requested in hour:minute:second

cd ..

CUDA_VISIBLE_DEVICES=0 python -u eval.py --fraction $1 --dataset Imagenetbg --data_path /n/fs/visualai-scr/Data/ --num_exp 1 --workers 10 --optimizer SGD --selection $2 --model ResNet50 --lr 0.001 -sp $3 --balance False --pretrain True --linear_probe False --policy $4 --score_pretrain True --test_split $5

