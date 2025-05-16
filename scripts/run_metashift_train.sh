#!/bin/bash

cd ..
CUDA_VISIBLE_DEVICES=0 python -u train.py --fraction $1 --dataset Metashift --data_path ./data/ --workers 10 --optimizer SGD --selection $2 --model ResNet50 --lr 0.001 -sp $3 --balance False --pretrain True --linear_probe False --score_path $4 --policy $5 --score_pretrain True -wd $6 --class_balance False --class_equal $7 --scheduler None --epochs $8 --drop_percent 0.03 #--features $9 #--num_exp $10 