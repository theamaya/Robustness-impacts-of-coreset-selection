import os
import time


wd= 1e-4 #0.01 #5e-4
epochs=10
fix_iterations= True

class_balance = False
class_equal= True

num_exp=1

# methods= ['EL2N', 'Uncertainty', 'Forgetting']
# methods= ['EL2N']
fractions= [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98]
# fractions= [0.98]
methods= ['EL2N', 'SelfSup']
policies= ['difficult', 'easy']

methods= ['EL2N']
policies= ['difficult-groupbal','difficult-filtered-groupbal','easy-groupbal']

for method in methods:
    for fraction in fractions:
        if fix_iterations:
            train_epochs=int(epochs/fraction)
        else:
            train_epochs= epochs
        for policy in policies:
            sp= '/n/fs/dk-diffusion/repos/DeepCore/checkpoints_ADAM_classequal/'
            score_path= f'/n/fs/dk-diffusion/repos/DeepCore/all_results_adam/civilcomments_pretrained/Civilcomments_Bert_{method}_exp0_0.1_unknown.ckpt'
            os.system(f"sbatch run_civilcomments_train.sh {fraction} {method} {sp} {score_path} {policy} {wd} {class_equal} {train_epochs}")
            # time.sleep(1)       
    # break