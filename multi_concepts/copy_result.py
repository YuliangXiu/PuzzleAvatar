import os
from glob import glob
import numpy as np
import argparse
from tqdm import tqdm
from safetensors.torch import load_file
from shutil import copyfile, copytree

if __name__ == "__main__":

    subjects = np.loadtxt(f"./clusters/lst/subjects_all.txt", dtype=str, delimiter=" ")[:, 1:]
    
    os.makedirs(f"./results/full_preview", exist_ok=True)
    for (subject, outfit) in tqdm(subjects):
        src_path = f"./data/PuzzleIOI_4views/puzzle_cam_full/{subject}/{outfit}/render/000.png"
        tgt_path = f"./results/full_preview/{subject}_{outfit}.png"
        
        copyfile(src_path, tgt_path)
    
