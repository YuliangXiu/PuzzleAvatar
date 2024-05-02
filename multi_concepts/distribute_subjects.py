import os
import numpy as np

subjects = np.loadtxt("./clusters/lst/subjects_test.txt", dtype=str)
np.random.shuffle(subjects)
group_size = 5  

with open(f"./clusters/lst/group_{group_size}.txt", "w") as f:
    for start_idx in range(len(subjects)//group_size):
        group = subjects[start_idx*group_size:(start_idx+1)*group_size][:,1:]
        line = "_".join([f"{g[0]}_{g[1]}" for g in group])
        f.write(line + "\n")
    

