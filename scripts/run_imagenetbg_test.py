import os
import time

test_splits=['test','mixed_rand','no_fg','only_fg']
# test_splits=['test']
sp= '/n/fs/dk-diffusion/repos/DeepCore/checkpoints_SGD_classequal'


# methods= ['EL2N']
# fractions= [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98]
# policies= ['easy', 'difficult']
# for method in methods:
#     for fraction in fractions:
#         for policy in policies:
#             for test_split in test_splits:
#                 # print(f"bash run_imagenetbg_test.sh {fraction} {method} {sp} {policy} {test_split}")
#                 os.system(f"sbatch run_imagenetbg_test.sh {fraction} {method} {sp} {policy} {test_split}")

methods= ['EL2N']
fractions= [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 1.0]
policies= ['random']
for method in methods:
    for fraction in fractions:
        for policy in policies:
            for test_split in test_splits:
                os.system(f"sbatch run_imagenetbg_test.sh {fraction} {method} {sp} {policy} {test_split}")
            

# methods= ['SelfSup']
# fractions= [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98]
# policies= ['easy', 'difficult']
# for method in methods:
#     for fraction in fractions:
#         for policy in policies:
#             for test_split in test_splits:
#                 os.system(f"sbatch run_imagenetbg_test.sh {fraction} {method} {sp} {policy} {test_split}")