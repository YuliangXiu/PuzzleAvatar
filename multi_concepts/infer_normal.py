import os
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from multi_concepts.puzzle_utils import PyRenderer

data_root = "./data/PuzzleIOI/fitting"

pbar = tqdm(glob(f"{data_root}/*/outfit*/"))

renderer = PyRenderer(data_root, torch.device("cuda:0"))

for subject_outfit in pbar:
    subject, outfit = subject_outfit.split("/")[-3:-1]
    renderer.load_assets(subject, outfit)
    normal_dir = f"{data_root}/{subject}/{outfit}/normal"
    os.makedirs(normal_dir, exist_ok=True)
    
    for cam_id in renderer.cameras.keys():
        pbar.set_description(f"Rendering normal for {subject}/{outfit}/{cam_id}")
        normal = renderer.render_normal(cam_id)
        plt.imsave(os.path.join(normal_dir, f"{cam_id}.png"), normal)