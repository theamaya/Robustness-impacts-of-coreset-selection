#!/bin/bash

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:1          # the number of GPUs requested
#SBATCH --mem=16G             # memory 
#SBATCH --error=./slurm_err/%J.err
#SBATCH --output=./slurm_out/%J.out
#SBATCH -t 168:00:00           # time requested in hour:minute:second
# SBATCH --mail-type=end       # choice between begin, end, all to notify you via email
# SBATCH --mail-user=dk9893@princeton.edu

CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset Coco --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection $1 --model ResNet18 --lr 0.1 -sp ./result_coco --batch 128 --balance False --pretrain False --uncertainty Margin
# CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset Coco --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection $1 --model ResNet18 --lr 0.1 -sp ./result_coco_pretrain_multilabel --batch 128 --balance False --pretrain True
