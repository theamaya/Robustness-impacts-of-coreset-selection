import os
import time

coreset_methods= ['EL2N', 'Uncertainty', 'Forgetting', 'SelfSup', 'supProto',]
uncertainty_methods=['Entropy']
for j in range(5): 
    if coreset_methods[j]== 'Forgetting':
        se= 20
    else:
        se= 5
    os.system(f"bash run_civilcomments.sh {se} {coreset_methods[j]} {uncertainty_methods[0]}")
    time.sleep(1)