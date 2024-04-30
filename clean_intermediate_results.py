import os
import numpy as np
from tqdm import tqdm

subjects = np.loadtxt("clusters/subjects_all.txt", dtype=str)[:, 1:]

keep_files = [".mp4", "full-body.jpg"]

for idx in ['0001', '5000', '10000']:
    keep_files.append(f"prompt2img_{idx}_prompt.png")
    keep_files.append(f"{idx}_guidance.png")
    keep_files.append(f"{idx}_render.png")

for (subject, outfit) in tqdm(subjects):
    geometry_dir = f"results/mvdream/PuzzleIOI/puzzle_cam/{subject}/{outfit}/geometry/visualize"
    texture_dir = f"results/mvdream/PuzzleIOI/puzzle_cam/{subject}/{outfit}/texture/visualize"

    if os.path.exists(geometry_dir):
        for file in os.listdir(geometry_dir):
            if not any([file.endswith(keep) for keep in keep_files]):
                os.remove(os.path.join(geometry_dir, file))

    if os.path.exists(texture_dir):
        for file in os.listdir(texture_dir):
            if not any([file.endswith(keep) for keep in keep_files]):
                os.remove(os.path.join(texture_dir, file))
