import json
import os
import argparse
import shutil
from glob import glob

import cv2
import numpy as np
import rpack

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True, help="output mask folder")
    parser.add_argument('--overwrite', action="store_true")
    opt = parser.parse_args()

    output_dir = f"{opt.out_dir}/packed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if opt.overwrite:
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))

    image_dir = f"{opt.out_dir}/image"
    mask_dir = f"{opt.out_dir}/mask"

    with open(os.path.join(f"{opt.out_dir}", 'gpt4v_simple.json'), 'r') as f:
        gpt4v_response = json.load(f)

    classes = list(gpt4v_response.keys())
    classes.remove("gender")

    for cls in classes:
        sizes = []
        corners = []
        mask_paths = glob(f"{opt.out_dir}/mask/*{cls}.png")
        if len(mask_paths) > 0:
            masks = [cv2.imread(mask_path) for mask_path in mask_paths]
            def read_images(file_path_base): # read all image format
                images = []
                for file_path in glob(file_path_base + "*"):
                    image = cv2.imread(file_path)
                    if image is not None:
                        images.append(image)
                return images
            dir_name = os.path.join(os.path.dirname(mask_paths[0].replace("mask", "image")), "")
            images = read_images(dir_name)

            for mask in masks:
                contours, _ = cv2.findContours(
                    mask[:, :, 0], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                cnt = sorted(contours, key=cv2.contourArea)
                if cls == "face":
                    # may be segmented by eyeglasses
                    top_k = 2
                else:
                    top_k = 1
                x, y, w, h = cv2.boundingRect(np.concatenate(cnt[::-1][:top_k], axis=0))
                sizes.append((w, h))
                corners.append((x, y))

            max_h = max([item[1] for item in sizes])
            max_w = max([item[0] for item in sizes])

            positions = rpack.pack(
                sizes,
                max_height=int(np.sqrt(len(mask_paths)) + 2) * max_h,
                max_width=int(np.sqrt(len(mask_paths)) + 2) * max_w
            )

            carvas_w = max([p[0] + sizes[idx][0] for idx, p in enumerate(positions)])
            carvas_h = max([p[1] + sizes[idx][1] for idx, p in enumerate(positions)])
            positions = [(p[0], carvas_h - p[1] - sizes[idx][1]) for idx, p in enumerate(positions)]
            carvas_img = np.zeros((carvas_h, carvas_w, 3), dtype=np.uint8)

            for idx, mask in enumerate(masks):
                carvas_img[positions[idx][1]:positions[idx][1]+sizes[idx][1],
                        positions[idx][0]:positions[idx][0]+sizes[idx][0]] = \
                            (images[idx]* ((mask[:,:,0:1]>0)).astype(np.float32))[corners[idx][1]:corners[idx][1]+sizes[idx][1],
                                    corners[idx][0]:corners[idx][0]+sizes[idx][0]]

            output_path = f"{output_dir}/{cls}.png"
            cv2.imwrite(output_path, carvas_img[:, :, [0, 1, 2]])
