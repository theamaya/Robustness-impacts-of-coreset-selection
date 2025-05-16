#!/bin/bash

cd ..
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset waterbirds --data_path ./data/waterbird/waterbird_complete95_forest2water2 --num_exp 3 --workers 10 -se $1 --selection $2 -sp ./all_results_SGD/waterbirds_pretrained_imagenet --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 --precalcfeatures_path '' --model ResNet50