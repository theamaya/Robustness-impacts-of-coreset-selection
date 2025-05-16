#!/bin/bash

cd ..

CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset Urbancars_both --data_path ./data/ --num_exp 1 --workers 10 -se $1 --selection $2 -sp ./all_results_SGD/Urbancars_pretrained_imagenet --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 --precalcfeatures_path '' --model ResNet50
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset Urbancars_bg --data_path ./data/ --num_exp 1 --workers 10 -se $1 --selection $2 -sp ./all_results_SGD/Urbancars_pretrained_imagenet --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 --precalcfeatures_path '' --model ResNet50
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset Urbancars_cooccur --data_path ./data/ --num_exp 1 --workers 10 -se $1 --selection $2 -sp ./all_results_SGD/Urbancars_pretrained_imagenet --batch 128 --balance False --pretrain True --linear_probe False --uncertainty $3 --precalcfeatures_path '' --model ResNet50