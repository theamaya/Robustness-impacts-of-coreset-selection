#!/bin/bash

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:1          # the number of GPUs requested
#SBATCH --mem=16G             # memory 
#SBATCH --error=./slurm_err/%J.err
#SBATCH --output=./slurm_out/%J.out
#SBATCH -t 36:00:00           # time requested in hour:minute:second
# SBATCH --mail-type=end       # choice between begin, end, all to notify you via email
# SBATCH --mail-user=dk9893@princeton.edu
#SBATCH -x node018

cd ..

# CUDA_VISIBLE_DEVICES=0 python -u train.py --fraction $1 --dataset waterbirds --data_path /n/fs/visualai-scr/Data/waterbird/waterbird_complete95_forest2water2 --workers 10 --optimizer SGD --selection $2 --model ResNet50 --lr 0.001 -sp $3 --balance False --pretrain True --linear_probe True --score_path $4 --policy $5 --score_pretrain True -wd $6 --class_balance $7 --class_equal $8 --scheduler None --epochs $9  #--num_exp $10 

# CUDA_VISIBLE_DEVICES=0 python -u train_backup.py --fraction $1 --dataset waterbirds75 --data_path /n/fs/visualai-scr/Data/Waterbirds-varients/waterbird_complete75_forest2water2 --workers 10 --optimizer SGD --selection $2 --model ResNet50 --lr 0.001 -sp $3 --balance False --pretrain True --linear_probe False --score_path $4 --policy $5 --score_pretrain True -wd $6 --class_balance $7 --class_equal $8 --scheduler None --epochs $9 #--num_exp $10 
# CUDA_VISIBLE_DEVICES=0 python -u train.py --fraction $1 --dataset waterbirds50 --data_path /n/fs/visualai-scr/Data/Waterbirds-varients/waterbird_complete50_forest2water2 --workers 10 --optimizer SGD --selection $2 --model ResNet50 --lr 0.001 -sp $3 --balance False --pretrain True --linear_probe False --score_path $4 --policy $5 --score_pretrain True -wd $6 --class_balance $7 --class_equal $8 --scheduler None --epochs $9 #--num_exp $10 

CUDA_VISIBLE_DEVICES=0 python -u train_backup.py --fraction $1 --dataset CelebAhairBalanced --data_path /n/fs/dk-diffusion/repos/DomainBiasMitigation/data/celeba --workers 10 --optimizer SGD --selection $2 --model ResNet50 --lr 0.001 -sp $3 --balance False --pretrain True --linear_probe False --score_path $4 --policy $5 --score_pretrain True -wd $6 --class_balance False --class_equal $7 --scheduler None --epochs $8 #--features $9 #--num_exp $10 
