#!/bin/bash

cd ..
CUDA_VISIBLE_DEVICES=0 python -u train.py --fraction $1 --dataset MultiNLI --data_path ./data/multinli/ --workers 10 --optimizer bert_adamw_optimizer --selection $2 --model Bert --lr 1e-5 -sp $3 --balance False --pretrain True --linear_probe False --score_path $4 --policy $5 --score_pretrain True -wd $6 --class_balance False --class_equal $7 --scheduler None --epochs $8 --batch 16 --drop_percent 0.04 #--features $9 #--num_exp $10 