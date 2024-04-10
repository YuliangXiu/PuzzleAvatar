import os
import argparse
import random
from glob import glob

import cv2
import numpy as np
import rpack


def rectpack(outfit_dir, outdir):

    cnt = 20

    filename = "_".join(outfit_dir.split("/")[-3:-1])
    outpath = os.path.join(outdir, f"{filename}.jpg")

    if not os.path.exists(outpath):

        sizes = []
        positions = []

        img_files = [item for item in os.listdir(outfit_dir) if "raw" in item]

        if len(img_files) < cnt:
            return
        select_idx = random.sample(range(len(img_files)), cnt)
        img_files = [sorted(img_files)[idx] for idx in select_idx]
        images = [cv2.imread(os.path.join(outfit_dir, img_file)) for img_file in img_files]

        for img_file in img_files:
            img = cv2.imread(os.path.join(outfit_dir, img_file))
            sizes.append(img.shape[:2][::-1])

        positions = rpack.pack(
            sizes,
            max_height=int(np.sqrt(cnt) + 2) * sizes[0][1],
            max_width=int(np.sqrt(cnt) + 2) * sizes[0][0]
        )

        carvas_w = max([p[0] + sizes[idx][0] for idx, p in enumerate(positions)])
        carvas_h = max([p[1] + sizes[idx][1] for idx, p in enumerate(positions)])
        positions = [(p[0], carvas_h - p[1] - sizes[idx][1]) for idx, p in enumerate(positions)]
        carvas_img = np.zeros((carvas_h, carvas_w, 3), dtype=np.uint8)

        for idx, image in enumerate(images):
            carvas_img[positions[idx][1]:positions[idx][1] + sizes[idx][1],
                       positions[idx][0]:positions[idx][0] + sizes[idx][0]] = image
        cv2.imwrite(outpath, carvas_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True, help="input folder")
    parser.add_argument('--out_dir', type=str, required=True, help="output folder")
    opt = parser.parse_args()

    all_outfit_dirs = glob(f"{opt.in_dir}/*/outfit*/image")

    from functools import partial
    from glob import glob
    from multiprocessing import Pool
    import multiprocessing as mp
    from tqdm import tqdm

    print("CPU:", mp.cpu_count())
    print("propress", len(all_outfit_dirs))

    with Pool(processes=mp.cpu_count(), maxtasksperchild=1) as pool:
        for _ in tqdm(
            pool.imap_unordered(partial(rectpack, outdir=opt.out_dir), all_outfit_dirs),
            total=len(all_outfit_dirs)
        ):
            pass
        pool.close()
        pool.join()
