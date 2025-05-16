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
# SBATCH --mail-type=end       # choice between begin, end, all to notify you via email
# SBATCH --mail-user=dk9893@princeton.edu

cd ..
# CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset CIFAR10 --data_path ~/datasets --num_exp 30 --workers 10 --optimizer SGD -se 10 --selection $1 --model ResNet18 --lr 0.1 -sp ./result_cifar --batch 128 --balance False --pretrain False #--save_model True #--uncertainty LeastConfidence
# CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset CIFAR10 --data_path ~/datasets --num_exp 30 --workers 10 --optimizer SGD -se 10 --selection $1 --model ResNet18 --lr 0.1 -sp ./result_cifar_pretrained --batch 128 --balance False --pretrain True --uncertainty Margin #--save_model True 
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset Imagenet_geode --data_path /n/fs/visualai-scr/Data/geode --num_exp 1 --workers 10 -se $1 --selection $2 --model ResNet50 -sp ./all_results_SGD/geodeimagenet_pretrained_imagenet --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 #--save_model False 


