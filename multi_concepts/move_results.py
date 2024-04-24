import numpy as np
from shutil import copyfile, copytree
from tqdm import tqdm
import os

if __name__ == "__main__":

    all_outfits = np.loadtxt("clusters/subjects_all.txt", dtype=str, delimiter=" ")[:, 0]

    out_root = "/is/cluster/fast/yxiu/PuzzleAvatar_data/results/backup"

    # move objs
    # for outfit in tqdm(all_outfits):
    #     os.makedirs(f"{out_root}/{outfit}", exist_ok=True)
    #     for file in os.listdir(f"./results/{outfit}/obj"):
    #         if file.endswith("geometry_final.obj") or "texture" in file:
    #             copyfile(f"./results/{outfit}/obj/{file}", f"{out_root}/{outfit}/{file}")

    # move renders
    for outfit in tqdm(all_outfits):
        os.makedirs(f"{out_root}/{outfit}", exist_ok=True)
        copytree(
            f"./data/{outfit.replace('PuzzleIOI', 'PuzzleIOI_4views')}/normal",
            f"{out_root}/{outfit}/normal"
        )
        copytree(
            f"./data/{outfit.replace('PuzzleIOI', 'PuzzleIOI_4views')}/render",
            f"{out_root}/{outfit}/render"
        )
