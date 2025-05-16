import os
import time

# coreset_methods= ['Uniform', 'Herding', 'kCenterGreedy', 'Submodular', 'GraNd', 'Forgetting']
# coreset_methods= ['Uniform', 'Herding', 'kCenterGreedy']
# coreset_methods= ['kCenterGreedy','Herding']

# coreset_sizes=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.04, 0.06, 0.08, 0.14, 0.16]
# coreset_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# for i in range(1): #add more for robustness
#     for j in range(6): 
#         for k in range(15):
#             os.system(f"sbatch run_waterbirds.sh {coreset_sizes[k]} {coreset_methods[j]}")

            # assert(False)
            # break
            # time.sleep(1)


# coreset_methods= ['Uniform', 'Herding', 'GraNd', 'Forgetting']
# coreset_sizes=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.04, 0.06, 0.08, 0.14, 0.16]

# for i in range(1): #add more for robustness
#     for j in range(4): 
#         for k in range(15):
#             os.system(f"sbatch run_celebahair.sh {coreset_sizes[k]} {coreset_methods[j]}")

#             # assert(False)
#             # break
#             time.sleep(1)


# coreset_methods= ['GraNd', 'Uncertainty', 'DeepFool', 'Consistent', 'Areaum', 'Loss_Acc', 'Forgetting']
# uncertainty_methods=['Entropy', 'Margin', 'LeastConfidence']
# for i in range(1): #add more for robustness
#     for j in range(7): 
#         if j== 1:
#             for k in range(3):
#                 os.system(f"sbatch run_nico95underrep.sh {coreset_methods[j]} {uncertainty_methods[k]}")
#                 time.sleep(1)
#         else:
#             os.system(f"sbatch run_nico95underrep.sh {coreset_methods[j]} {uncertainty_methods[0]}")

#         # assert(False)
#         # break
#         time.sleep(1)


# coreset_methods= ['EL2N', 'Uncertainty', 'Forgetting', 'Moderate', 'SelfSup']
# uncertainty_methods=['Entropy']
# for j in range(1,2,1): 
#     if coreset_methods[j]== 'Forgetting':
#         se= 100
#     else:
#         se= 20
#     os.system(f"sbatch run_waterbirds.sh {se} {coreset_methods[j]} {uncertainty_methods[0]}")
#     time.sleep(1)


# coreset_methods= ['EL2N', 'Uncertainty', 'Forgetting', 'Moderate', 'SelfSup']
# uncertainty_methods=['Entropy']
# for j in range(2,3,1): 
#     se= 200
#     os.system(f"sbatch run_waterbirds.sh {se} {coreset_methods[j]} {uncertainty_methods[0]}")
#     time.sleep(1)

coreset_methods= ['Moderate', 'SelfSup']
models=['CLIP-ViTB32', 'Dinov2-ViTb14', 'ViTB16', 'ViTb32']
feature_paths=['/n/fs/dk-diffusion/repos/DeepCore/features/nicospurious/CLIP-ViTB32_features.pt',
                '/n/fs/dk-diffusion/repos/DeepCore/features/nicospurious/dinov2_vitb14_features.pt',
                '/n/fs/dk-diffusion/repos/DeepCore/features/nicospurious/ViT_Imagenet1k_features.pt',
                '/n/fs/dk-diffusion/repos/DeepCore/features/nicospurious/ViTb32_Imagenet1k_features.pt']
uncertainty_methods=['Entropy']
for j in range(2): 
    se= 20
    for k in range(4):
        # os.system(f"sbatch run_waterbirds.sh {se} {coreset_methods[j]} {uncertainty_methods[0]} {feature_paths[k]} {mols[k]}")
        os.system(f"CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset Nico_95_spurious --data_path ~/datasets --num_exp 1 --workers 10 -se {se} --selection {coreset_methods[j]} -sp ./all_results_SGD/nicospurious_pretrained_feature_extractors --batch 128 --balance False --pretrain True --linear_probe False --uncertainty {uncertainty_methods[0]} --precalcfeatures_path {feature_paths[k]} --model {models[k]}")
        time.sleep(1)