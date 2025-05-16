#!/bin/bash
cd ..

CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset Cmnist --data_path ./data/ --num_exp 1 --workers 10 -se $1 --selection $2 -sp ./all_results_SGD/cmnist_pretrained --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 --precalcfeatures_path '' --model ResNet18 