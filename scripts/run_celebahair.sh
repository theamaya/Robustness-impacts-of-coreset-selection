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

# CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset waterbirds --data_path /n/fs/visualai-scr/Data/waterbird/waterbird_complete95_forest2water2 --num_exp 1 --workers 10 -se $1 --selection $2 --model ResNet50 -sp ./all_results_SGD/waterbirds_pretrained_imagenet_ablate --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 #--save_model False 
# CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset waterbirds50 --data_path /n/fs/visualai-scr/Data/Waterbirds-varients/waterbird_complete50_forest2water2 --num_exp 1 --workers 10 -se $1 --selection $2 --model ResNet50 -sp ./all_results_SGD/waterbirds50_pretrained_imagenet --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 #--save_model False 
# CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset waterbirds --data_path /n/fs/visualai-scr/Data/waterbird/waterbird_complete95_forest2water2 --num_exp 1 --workers 10 -se $1 --selection $2 -sp ./all_results_SGD/waterbirds_pretrained_imagenet --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 --precalcfeatures_path '' --model ResNet50
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset CelebAhair --data_path /n/fs/dk-diffusion/repos/DomainBiasMitigation/data/celeba --num_exp 3 --workers 10 -se $1 --selection $2 -sp ./all_results_SGD/celebahair_pretrained_imagenet --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 --precalcfeatures_path '' --model ResNet50