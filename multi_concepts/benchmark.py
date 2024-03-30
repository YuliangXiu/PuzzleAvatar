import os
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from multi_concepts.puzzle_utils import Evaluation_EASY as Evaluation

data_root = "./data/PuzzleIOI/fitting"
result_geo_root = "./results/PuzzleIOI/puzzle_cam"
result_img_root = "./data/PuzzleIOI_4views"

pbar = tqdm(glob(f"{data_root}/*/outfit*/"))

results = {}

evaluator = Evaluation(data_root, result_geo_root, result_img_root, torch.device("cuda:0"))

for subject_outfit in pbar:
    subject, outfit = subject_outfit.split("/")[-3:-1]
    results[subject] = {outfit: {}}

    results[subject][outfit] = {}

    evaluator.load_paths(subject, outfit)

    if os.path.exists(evaluator.recon_file) and len(evaluator.pelvis_file) > 0:
        evaluator.load_assets()
        results[subject][outfit].update(evaluator.calculate_p2s())
        results[subject][outfit].update(evaluator.calculate_visual_similarity())
        pbar_desc = f"{subject}/{outfit} --- "
        for key in results[subject][outfit].keys():
            pbar_desc += f"{key}: {results[subject][outfit][key]:.3f} "
        pbar.set_description(pbar_desc)

np.save(f"./results/PuzzleIOI/results.npy", results, allow_pickle=True)
