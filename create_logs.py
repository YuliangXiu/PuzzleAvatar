import os
import numpy as np

subjects = np.loadtxt("clusters/subjects_all.txt", dtype=str)[:,1:]

for (subject, outfit) in subjects:
    os.makedirs(f"logs/{subject}/{outfit}", exist_ok=True)