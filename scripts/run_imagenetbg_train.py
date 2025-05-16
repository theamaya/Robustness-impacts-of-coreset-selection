import os
import time


wd= 0.01 #0.01 #5e-4
class_balance = False
class_equal= True
epochs=50

fix_iterations= True

methods= ['EL2N', 'SelfSup']
fractions= [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98]
policies= ['difficult', 'easy']

fractions= [0.8, 0.9, 0.95]
policies= ['difficult-filtered']
methods= ['EL2N']

for method in methods:
    for fraction in fractions:
        if fix_iterations:
            train_epochs=int(epochs/fraction)
        else:
            train_epochs= epochs
        for policy in policies:
            sp= '/n/fs/dk-diffusion/repos/DeepCore/checkpoints_SGD_classequal/'
            score_path= f'/n/fs/dk-diffusion/repos/DeepCore/all_results_SGD/imagenetbg_pretrained_imagenet/Imagenetbg_ResNet50_{method}_exp0_0.1_unknown.ckpt'
            os.system(f"sbatch run_imagenetbg_train.sh {fraction} {method} {sp} {score_path} {policy} {wd} {class_balance} {class_equal} {train_epochs}")
            # time.sleep(1)       
