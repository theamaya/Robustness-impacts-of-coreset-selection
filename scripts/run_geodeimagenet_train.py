import os
import time


# methods= ['GraNd', 'Forgetting', 'Uncertainty', 'Uncertainty_Margin', 'Uncertainty_LeastConfidence', 'DeepFool', 'Consistent', 'Loss', 'Accuracy', 'Areaum']
# # CUDA_VISIBLE_DEVICES=0 python -u train.py --fraction $1 --dataset waterbirds --data_path /n/fs/visuala --selection $2 --model ResNet18 --lr 0.1 -sp $3 --batch 128 --balance False --pretrain False --linear_probe False --subset_path $4 --score_path $5 --level $6 --policy $7 
# for method in methods:
#     for level in range(5,6,1):
#         for fraction in range(1,11,1):
#             for policy in ['difficult', 'easy','random','median']:
#                 sp= '/n/fs/dk-diffusion/repos/DeepCore/checkpoints_waterbird_subsets/linear_probe/level_'+str(level)
#                 subset_path= '/n/fs/dk-diffusion/repos/DeepCore/deepcore/datasets/waterbirds_subset_'+str(level)+'.pt'
#                 if method== 'Accuracy' or method=='Loss':
#                     score_path= f'/n/fs/dk-diffusion/repos/DeepCore/result_waterbird_subsets/from_scratch/level_{level}/waterbirds_ResNet18_Loss_Acc_exp0_0.1_unknown.ckpt'
#                 else:
#                     score_path= f'/n/fs/dk-diffusion/repos/DeepCore/result_waterbird_subsets/from_scratch/level_{level}/waterbirds_ResNet18_{method}_exp0_0.1_unknown.ckpt'
#                 os.system(f"sbatch train_waterbird_subset.sh {fraction*0.1} {method} {sp} {subset_path} {score_path} {level} {policy}")
#                 # time.sleep(1)       
#         #         break
#         #     break
#         # break
#     break

wd= 0.01 #0.01 #5e-4
class_balance = False
class_equal= True
epochs=100

fix_iterations= True

# methods= ['EL2N', 'Uncertainty', 'Forgetting']
methods= ['Forgetting']
fractions= [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98]
policies= ['difficult', 'easy']
for method in methods:
    for fraction in fractions:
        if fix_iterations:
            train_epochs=int(epochs/fraction)
        else:
            train_epochs= epochs
        for policy in policies:
            sp= '/n/fs/dk-diffusion/repos/DeepCore/checkpoints_SGD_classequal/'
            score_path= f'/n/fs/dk-diffusion/repos/DeepCore/all_results_SGD/geodeimagenet/Imagenet_geode_ResNet50_{method}_exp0_0.1_unknown.ckpt'
            os.system(f"sbatch run_geodeimagenet_train.sh {fraction} {method} {sp} {score_path} {policy} {wd} {class_balance} {class_equal} {train_epochs}")
            # time.sleep(1)       


methods= ['Moderate', 'SelfSup']
fractions= [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98]
policies= ['difficult', 'easy']
for method in methods:
    for fraction in fractions:
        if fix_iterations:
            train_epochs=int(epochs/fraction)
        else:
            train_epochs= epochs
        for policy in policies:
            sp= '/n/fs/dk-diffusion/repos/DeepCore/checkpoints_SGD_classequal/'
            score_path= f'/n/fs/dk-diffusion/repos/DeepCore/all_results/geodeimagenet_pretrained/Imagenet_geode_ResNet50_{method}_exp0_0.1_unknown.ckpt'
            os.system(f"sbatch run_geodeimagenet_train.sh {fraction} {method} {sp} {score_path} {policy} {wd} {class_balance} {class_equal} {train_epochs}")
            # time.sleep(1)      