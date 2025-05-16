#!/bin/bash

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
#SBATCH --mem=16G             # memory 
#SBATCH --error=./slurm_err/%J.err
#SBATCH --output=./slurm_out/%J.out
#SBATCH -t 48:00:00           # time requested in hour:minute:second
# SBATCH --mail-type=end       # choice between begin, end, all to notify you via email
# SBATCH --mail-user=dk9893@princeton.edu

cd ..

# CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset CelebAnose --data_path /n/fs/visualai-scr/Data/CelebA --num_exp 1 --workers 10 -se $1 --selection $2 -sp ./all_results_SGD/celeba_special_pretrained_imagenet --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 --precalcfeatures_path '' --model ResNet50
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset CelebAlipstick --data_path /n/fs/visualai-scr/Data/CelebA --num_exp 1 --workers 10 -se $1 --selection $2 -sp ./all_results_SGD/celeba_special_pretrained_imagenet --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 --precalcfeatures_path '' --model ResNet50
# CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset CelebAcheekbones --data_path /n/fs/visualai-scr/Data/CelebA --num_exp 1 --workers 10 -se $1 --selection $2 -sp ./all_results_SGD/celeba_special_pretrained_imagenet --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 --precalcfeatures_path '' --model ResNet50
# CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset CelebAyoung --data_path /n/fs/visualai-scr/Data/CelebA --num_exp 1 --workers 10 -se $1 --selection $2 -sp ./all_results_SGD/celeba_special_pretrained_imagenet --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 --precalcfeatures_path '' --model ResNet50