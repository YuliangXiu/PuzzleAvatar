import os
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from multi_concepts.puzzle_utils import Evaluation

data_root = "./data/PuzzleIOI/fitting"
result_root = "./results/PuzzleIOI/puzzle_cam"
pbar = tqdm(glob(f"{data_root}/*/outfit*/"))

results = {}

evaluator = Evaluation(data_root, result_root, torch.device("cuda:0"))

for subject in pbar:
    results[subject] = {}
    for outfit in os.listdir(os.path.join(data_root, subject)):
        pbar.set_description(f"Processing {subject}/{outfit}")

        results[subject][outfit] = {}

        evaluator.load_assets(subject, outfit)

        if os.path.exists(evaluator.recon_file) and len(evaluator.pelvis_file) > 0:

            results[subject][outfit] = evaluator.calculate_p2s()

        for cam_id in evaluator.cameras.keys():
            results[subject][outfit][cam_id] = evaluator.calculate_visual_similarity(cam_id)

np.save(f"./results/PuzzleIOI/results.npy", results, allow_pickle=True)
