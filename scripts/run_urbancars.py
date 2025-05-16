import os
import time

coreset_methods= ['EL2N', 'Uncertainty', 'Forgetting', 'supProto', 'SelfSup']
uncertainty_methods=['Entropy']
for j in range(5): 
    if coreset_methods[j]== 'Forgetting':
        se= 100
    else:
        se= 20
    os.system(f"bash run_urbancars.sh {se} {coreset_methods[j]} {uncertainty_methods[0]}")
    time.sleep(1)