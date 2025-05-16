import os
import time

coreset_methods= ['EL2N', 'Uncertainty', 'Forgetting', 'SelfSup', 'supProto']
uncertainty_methods=['Entropy']
for j in range(3): 
    if coreset_methods[j]== 'Forgetting':
        se= 50
    else:
        se= 10
    os.system(f"sbatch run_cxr8.sh {se} {coreset_methods[j]} {uncertainty_methods[0]}")
    time.sleep(1)