# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

import os
import numpy as np
import torch
import argparse
import torchvision
import trimesh
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    # loading cfg file
    parser = argparse.ArgumentParser()
    parser.add_argument("-tag", "--tag", type=str, default="full")
    parser.add_argument("-method", "--method", type=str, default="puzzle_cam")
    parser.add_argument("-split", "--split", type=str, default="test")
    args = parser.parse_args()

    subjects_outfits = np.loadtxt(
        f"./clusters/subjects_{args.split}.txt", dtype=str, delimiter=" "
    )[:, 1:]
    subfolders = [
        f"./data/PuzzleIOI_4views/{args.method}_{args.tag}/{subject}/{outfit}"
        for subject, outfit in subjects_outfits
    ]

    normal_consist = []
    normal_diff = []
    for subfolder in tqdm(subfolders):

        if os.path.exists(subfolder):

            normal_obj = [
                torch.as_tensor(plt.imread(img_file))
                for img_file in sorted(glob(f"{subfolder}/normal/*.png"))
            ]

            normal_est = [
                torch.as_tensor(plt.imread(img_file))
                for img_file in sorted(glob(f"{subfolder}/normal_est/*.png"))
            ]

            normal_gt = [
                torch.as_tensor(plt.imread(img_file)) for img_file in sorted(
                    glob(
                        f"{subfolder.replace(f'{args.method}_{args.tag}', 'fitting')}/normal/*.png"
                    )
                )
            ]

            normal_body = [
                torch.as_tensor(plt.imread(img_file)) for img_file in sorted(
                    glob(f"{subfolder.replace(f'{args.method}_{args.tag}', 'fitting')}/body/*.png")
                )
            ]

            bottomline = [
                torch.where(normal_body_img[..., -1] > 0)[0].max().item()
                for normal_body_img in normal_body
            ]
            new_normal_gt = []

            for idx, normal_gt_img in enumerate(normal_gt):
                normal_gt_img[bottomline[idx]:, :, -1] *= 0
                new_normal_gt.append(normal_gt_img)

            normal_obj_arr = make_grid(torch.cat(normal_obj, dim=1), nrow=4, padding=0)
            normal_est_arr = make_grid(torch.cat(normal_est, dim=1), nrow=4, padding=0)
            normal_gt_arr = make_grid(torch.cat(new_normal_gt, dim=1), nrow=4, padding=0)

            # torchvision.utils.save_image(normal_gt_arr.permute(2, 0, 1), "./tmp/normal_gt_arr.png")
            # # torchvision.utils.save_image(normal_est_arr.permute(2,0,1), "./tmp/normal_est.png")
            # # torchvision.utils.save_image(normal_obj_arr[...,[-1]].permute(2,0,1), "./tmp/mask_obj.png")
            # import sys
            # sys.exit()

            mask = normal_obj_arr[..., [-1]] * normal_est_arr[..., [-1]]

            normal_consist.append((((((normal_obj_arr[..., :3] - normal_est_arr[..., :3]) * mask)**
                                     2).sum(dim=2).mean()) * 4.0).item())

            mask = normal_gt_arr[..., [-1]]
            normal_diff.append((((((normal_obj_arr[..., :3] - normal_gt_arr[..., :3]) * mask)**
                                  2).sum(dim=2).mean()) * 4.0).item())
        else:
            print(subfolder)

    print(f"Consist: {args.method}-{args.tag}-{args.split}: {np.mean(normal_consist)}")
    print(f"Diff: {args.method}-{args.tag}-{args.split}: {np.mean(normal_diff)}")
