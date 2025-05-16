#!/bin/bash

cd ..
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset Metashift --data_path ./data/ --num_exp 1 --workers 10 -se $1 --selection $2 --model ResNet50 -sp ./all_results_SGD/metashift_pretrained_imagenet --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 #--save_model False 

