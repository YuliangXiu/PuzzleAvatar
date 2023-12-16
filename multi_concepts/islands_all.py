import json
import os
from glob import glob

import cv2
import numpy as np
import rpack

image_dir = "examples/multi_concepts/human/yuliang/image"
mask_dir = "examples/multi_concepts/human/yuliang/mask"

with open(os.path.join("examples/multi_concepts/human/yuliang", 'gpt4v_response.json'), 'r') as f:
    gpt4v_response = json.load(f)

classes = list(gpt4v_response.keys())
classes.remove("gender")

for cls in classes:
    sizes = []
    corners = []
    mask_paths = glob(f"examples/multi_concepts/human/yuliang/mask/*{cls}.png")
    masks = [cv2.imread(mask_path) for mask_path in mask_paths]
    images = [
        cv2.imread("_".join(mask_path.replace("mask", "image").split("_")[:2]) + ".png")
        for mask_path in mask_paths
    ]

    for mask in masks:
        contours, _ = cv2.findContours(mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        sizes.append((w, h))
        corners.append((x, y))

    positions = rpack.pack(sizes)
    carvas_w = max([p[0] + sizes[idx][0] for idx, p in enumerate(positions)])
    carvas_h = max([p[1] + sizes[idx][1] for idx, p in enumerate(positions)])
    positions = [(p[0], carvas_h - p[1] - sizes[idx][1]) for idx, p in enumerate(positions)]
    carvas_img = np.zeros((carvas_h, carvas_w, 3), dtype=np.uint8)

    for idx, mask in enumerate(masks):
        carvas_img[positions[idx][1]:positions[idx][1]+sizes[idx][1],
                   positions[idx][0]:positions[idx][0]+sizes[idx][0]] = \
                       (images[idx]* (mask[:,:,0:1]>0))[corners[idx][1]:corners[idx][1]+sizes[idx][1],
                            corners[idx][0]:corners[idx][0]+sizes[idx][0]]
    cv2.imwrite(
        f"examples/multi_concepts/human/yuliang/packed/{cls}.png", carvas_img[:, :, [0, 1, 2]]
    )
