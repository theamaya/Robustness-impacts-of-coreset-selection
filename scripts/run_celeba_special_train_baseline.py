import os
import time


wd= 0.01 #0.01 #5e-4

epochs=50
fix_iterations= True

methods= ['EL2N']
fractions= [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 1.0]
# fractions= [1.0]
# fractions= [0.6, 0.8, 0.9, 0.95, 0.98, 1.0]
num_exp=1

policies= ['random']
class_balance = False
class_equal= True

for method in methods:
    for fraction in fractions:
        if fix_iterations:
            train_epochs=int(epochs/fraction)
        else:
            train_epochs= epochs
        for policy in policies:
            sp= '/n/fs/dk-diffusion/repos/DeepCore/checkpoints_SGD_baselines_classequal/'
            score_path= f'/n/fs/dk-diffusion/repos/DeepCore/all_results_SGD/celeba_special_pretrained_imagenet/CelebAlipstick_ResNet50_{method}_exp0_0.1_unknown.ckpt'
            os.system(f"sbatch run_celeba_special_train_baseline.sh {fraction} {method} {sp} {score_path} {policy} {wd} {class_equal} {train_epochs}")
            # time.sleep(1)   
    # break 

policies= ['group-bal']
class_balance = False
class_equal= True

for method in methods:
    for fraction in fractions:
        if fix_iterations:
            train_epochs=int(epochs/fraction)
        else:
            train_epochs= epochs
        for policy in policies:
            sp= '/n/fs/dk-diffusion/repos/DeepCore/checkpoints_SGD_baselines_classequal/'
            score_path= f'/n/fs/dk-diffusion/repos/DeepCore/all_results_SGD/celeba_special_pretrained_imagenet/CelebAlipstick_ResNet50_{method}_exp0_0.1_unknown.ckpt'
            os.system(f"sbatch run_celeba_special_train_baseline.sh {fraction} {method} {sp} {score_path} {policy} {wd} {class_equal} {train_epochs}")
            # time.sleep(1)   
    # break