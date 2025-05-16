#!/bin/bash

cd ..
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset MultiNLI --data_path ./data/multinli/ --num_exp 3 --workers 10 -se $1 --selection $2 -sp ./all_results_adam/multinli_pretrained --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 --precalcfeatures_path '' --model Bert --optimizer adamw --lr 0.0001 --batch 16 --selection_batch 16 --selection_lr 0.0001