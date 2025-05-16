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
#SBATCH -x node018

cd ..

# CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction $1 --dataset waterbirds --data_path /n/fs/visualai-scr/Data/Waterbirds-varients/waterbird_complete50_forest2water2 --num_exp 3 --workers 10 --optimizer SGD -se 10 --selection $2 --model ResNet18 --lr 0.1 -sp ./result_waterbird50 --batch 128 #--balance False
# CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction $1 --dataset waterbirds --data_path /n/fs/visualai-scr/Data/Waterbirds-varients/waterbird_complete_nobg --num_exp 3 --workers 10 --optimizer SGD -se 10 --selection $2 --model ResNet18 --lr 0.1 -sp ./result_waterbirdnobg --batch 128 #--balance False
# CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction $1 --dataset waterbirds --data_path /n/fs/visualai-scr/Data/waterbird/waterbird_complete95_forest2water2 --num_exp 3 --workers 10 --optimizer SGD -se 10 --selection $2 --model ResNet18 --lr 0.1 -sp ./result_waterbird50 --batch 128 #--balance False
# CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction $1 --dataset waterbirds --data_path /n/fs/visualai-scr/Data/waterbird/waterbird_complete95_forest2water2 --num_exp 3 --workers 10 --optimizer SGD -se 0 --selection $2 --model ResNet18 --lr 0.1 -sp ./result_notrain --batch 128 #--balance False
# CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction $1 --dataset waterbirds --data_path /n/fs/visualai-scr/Data/waterbird/waterbird_complete95_forest2water2 --num_exp 3 --workers 10 --optimizer SGD -se 0 --selection $2 --model ResNet18 --lr 0.1 -sp ./result_notrain_nobalance --batch 128 --balance False

CUDA_VISIBLE_DEVICES=0 python -u train.py --fraction $1 --dataset Imagenet_geode --data_path /n/fs/visualai-scr/Data/geode --workers 10 --optimizer SGD --selection $2 --model ResNet50 --lr 0.001 -sp $3 --balance False --pretrain True --linear_probe False --score_path $4 --policy $5 --score_pretrain True -wd $6 --class_balance $7 --class_equal $8 --scheduler None --epochs $9 #--num_exp $10 

