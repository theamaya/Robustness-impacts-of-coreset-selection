import os
import time

coreset_methods= ['EL2N', 'Uncertainty', 'Forgetting', 'Moderate2', 'SelfSup', 'unsupHerding', 'supProto', 'supHerding']
uncertainty_methods=['Entropy']
for j in range(8): 
    if coreset_methods[j]== 'Forgetting':
        se= 100
    else:
        se= 20
    os.system(f"sbatch run_celebahair_balanced.sh {se} {coreset_methods[j]} {uncertainty_methods[0]}")
    time.sleep(1)