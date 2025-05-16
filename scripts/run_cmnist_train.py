import os
import time

wd= 0.01 #0.01 #5e-4
class_balance = False
class_equal= True
epochs=50

fix_iterations= True

num_exp=1

methods= ['EL2N', 'Uncertainty', 'Forgetting', 'Moderate2', 'SelfSup']#, 'unsupHerding', 'supProto', 'supHerding']
# methods= ['supHerding']
fractions= [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98]
# fractions= [0.6, 0.8, 0.9, 0.95, 0.98]
# fractions= [0.04, 0.05, 0.06, 0.07, 0.075, 0.08, 0.09, 0.11, 0.12, 0.13]
policies= ['difficult', 'easy', 'median']#, 'difficult-groupbal', 'easy-groupbal', 'groupbal-hard_majority', 'groupbal-easy_majority']
# policies= ['random']
# # policies= ['difficult-groupbal', 'easy-groupbal']
# # policies=['groupbal-hard_majority', 'groupbal-easy_majority']
# # policies=['half-difficult-easy']

methods= ['EL2N']
policies= ['difficult-groupbal','difficult-filtered-groupbal','easy-groupbal']

for method in methods:
    for fraction in fractions:
        if fix_iterations:
            train_epochs=int(epochs/fraction)
        else:
            train_epochs= epochs
        for policy in policies:
            sp= '/n/fs/dk-diffusion/repos/DeepCore/checkpoints_SGD_classequal/'
            score_path= f'/n/fs/dk-diffusion/repos/DeepCore/all_results_SGD/cmnist_pretrained/Cmnist_ResNet18_{method}_exp0_0.1_unknown.ckpt'
            os.system(f"sbatch run_cmnist_train.sh {fraction} {method} {sp} {score_path} {policy} {wd} {class_equal} {train_epochs}")


