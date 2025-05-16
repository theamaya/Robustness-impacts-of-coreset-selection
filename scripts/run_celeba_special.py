import os
import time

# coreset_methods= ['EL2N', 'Uncertainty', 'Forgetting', 'SelfSup','supProto']
coreset_methods= ['Uncertainty', ]
uncertainty_methods=['Entropy']
for j in range(1): 
    if coreset_methods[j]== 'Forgetting':
        se= 100
    else:
        se= 20
    os.system(f"sbatch run_celeba_special.sh {se} {coreset_methods[j]} {uncertainty_methods[0]}")
    time.sleep(1)