import os
import time

wd= 0.01 #0.01 #5e-4
class_balance = False
class_equal= True
epochs=50

fix_iterations= True

num_exp=1

methods= ['EL2N', 'Uncertainty', 'Forgetting', 'SelfSup', 'supProto']
methods= ['EL2N', 'SelfSup']
# methods= ['EL2N']
fractions= [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 1.0]
# fractions =[1.0]
policies= ['difficult', 'easy', 'median', 'random']#, 'difficult-groupbal', 'easy-groupbal', 'groupbal-hard_majority', 'groupbal-easy_majority']
# policies= ['random']

methods= ['EL2N']
policies= ['difficult-groupbal','difficult-filtered-groupbal','easy-groupbal']

dataset= 'CelebAlipstick'

for method in methods:
    for fraction in fractions:
        if fix_iterations:
            train_epochs=int(epochs/fraction)
        else:
            train_epochs= epochs
        for policy in policies:
            sp= '/n/fs/dk-diffusion/repos/DeepCore/checkpoints_SGD_classequal/'
            score_path= f'/n/fs/dk-diffusion/repos/DeepCore/all_results_SGD/celeba_special_pretrained_imagenet/{dataset}_ResNet50_{method}_exp0_0.1_unknown.ckpt'
            os.system(f"sbatch run_celeba_special_train.sh {fraction} {method} {sp} {score_path} {policy} {wd} {class_equal} {train_epochs}")
    # break
