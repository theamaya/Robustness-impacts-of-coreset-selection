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


coreset_methods= ['GraNd', 'Uncertainty', 'Forgetting']
uncertainty_methods=['Entropy', 'Margin', 'LeastConfidence']
for i in range(1): #add more for robustness
    for j in range(3): 
        if j== 1:
            for k in range(3):
                for level in range(10,11,1):
                    sp = './result_nico95underrep_subsets/from_scratch/level_'+str(level)+'/'
                    subset_path= '/n/fs/dk-diffusion/repos/DeepCore/deepcore/datasets/nico++underrep_subset_'+str(level)+'.pt'
                    os.system(f"sbatch run_nico95underrep.sh {coreset_methods[j]} {sp} {subset_path} {uncertainty_methods[k]}")
                    time.sleep(1)
        else:
            for level in range(10,11,1):
                sp = './result_nico95underrep_subsets/from_scratch/level_'+str(level)+'/'
                subset_path= '/n/fs/dk-diffusion/repos/DeepCore/deepcore/datasets/nico++underrep_subset_'+str(level)+'.pt'
                os.system(f"sbatch run_nico95underrep.sh {coreset_methods[j]} {sp} {subset_path} {uncertainty_methods[0]}")
                time.sleep(1)