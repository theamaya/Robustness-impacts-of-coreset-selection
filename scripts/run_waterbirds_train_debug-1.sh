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

cd ..

# CUDA_VISIBLE_DEVICES=0 python -u train.py --epochs 50 --fraction 1 --dataset waterbirds --data_path /n/fs/visualai-scr/Data/waterbird/waterbird_complete95_forest2water2 --num_exp 1 --workers 10 --optimizer SGD --selection EL2N --model ResNet50 --lr 0.001 -sp '/n/fs/dk-diffusion/repos/DeepCore/checkpoints_debug/' --balance False --pretrain True --linear_probe False --score_path /n/fs/dk-diffusion/repos/DeepCore/all_results_SGD/waterbirds_pretrained/waterbirds_ResNet50_EL2N_exp0_0.1_unknown.ckpt --policy random --score_pretrain True

CUDA_VISIBLE_DEVICES=0 python -u train_debug.py -wd 0.01 --scheduler None --epochs 50 --batch 32 --fraction 1 --dataset waterbirds --data_path /n/fs/visualai-scr/Data/waterbird/waterbird_complete95_forest2water2 --num_exp 1 --workers 10 --optimizer SGD --selection EL2N --model ResNet50 --lr 0.001 -sp '/n/fs/dk-diffusion/repos/DeepCore/checkpoints_debug/' --balance False --pretrain True --linear_probe False --score_path /n/fs/dk-diffusion/repos/DeepCore/all_results_SGD/waterbirds_pretrained/waterbirds_ResNet50_EL2N_exp0_0.1_unknown.ckpt --policy random --score_pretrain True

# CUDA_VISIBLE_DEVICES=0 python -u train.py -wd 0.01 --scheduler None --epochs 200 --batch 32 --fraction 1 --dataset waterbirds --data_path /n/fs/visualai-scr/Data/waterbird/waterbird_complete95_forest2water2 --num_exp 1 --workers 10 --optimizer SGD --selection EL2N --model ResNet50 --lr 0.001 -sp '/n/fs/dk-diffusion/repos/DeepCore/checkpoints_debug/' --balance False --pretrain True --linear_probe False --score_path /n/fs/dk-diffusion/repos/DeepCore/all_results_SGD/waterbirds_pretrained/waterbirds_ResNet50_EL2N_exp0_0.1_unknown.ckpt --policy random --score_pretrain True

# CUDA_VISIBLE_DEVICES=0 python -u train.py -wd 0.01 --scheduler None --epochs 200 --imagenet_pretrain True --batch 32 --fraction 1 --dataset waterbirds --data_path /n/fs/visualai-scr/Data/waterbird/waterbird_complete95_forest2water2 --num_exp 1 --workers 10 --optimizer SGD --selection EL2N --model ResNet50 --lr 0.001 -sp '/n/fs/dk-diffusion/repos/DeepCore/checkpoints_debug/' --balance False --pretrain True --linear_probe False --score_path /n/fs/dk-diffusion/repos/DeepCore/all_results_SGD/waterbirds_pretrained/waterbirds_ResNet50_EL2N_exp0_0.1_unknown.ckpt --policy random --score_pretrain True

# CUDA_VISIBLE_DEVICES=0 python -u train.py -wd 0.01 --scheduler None --epochs 50 --imagenet_pretrain True --batch 32 --fraction 1 --dataset waterbirds --data_path /n/fs/visualai-scr/Data/waterbird/waterbird_complete95_forest2water2 --num_exp 1 --workers 10 --optimizer SGD --selection EL2N --model ResNet50 --lr 0.001 -sp '/n/fs/dk-diffusion/repos/DeepCore/checkpoints_debug/' --balance False --pretrain True --linear_probe False --score_path /n/fs/dk-diffusion/repos/DeepCore/all_results_SGD/waterbirds_pretrained/waterbirds_ResNet50_EL2N_exp0_0.1_unknown.ckpt --policy random --score_pretrain True

