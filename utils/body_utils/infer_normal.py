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
from glob import glob
from tqdm import tqdm
from lib.common.config import cfg
from lib.common.train_util import Format
from lib.Normal import Normal
from termcolor import colored

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    # loading cfg file
    parser = argparse.ArgumentParser()
    parser.add_argument("-tag", "--tag", type=str, default="full")
    parser.add_argument("-method", "--method", type=str, default="puzzle_cam")
    parser.add_argument("-split", "--split", type=str, default="test")
    parser.add_argument("-overwrite", "--overwrite", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    device = torch.device("cuda:0")

    # cfg read and merge
    cfg.merge_from_file("utils/body_utils/configs/body.yaml")
    cfg.freeze()

    # load normal model
    normal_net = Normal.load_from_checkpoint(
        cfg=cfg,
        checkpoint_path="./data/body_data/ckpt/normal.ckpt",
        map_location=device,
        strict=False
    )
    normal_net = normal_net.to(device)
    normal_net.netG.eval()
    print(
        colored(
            f"Resume Normal Estimator from {Format.start} ./data/body_data/ckpt/normal.ckpt {Format.end}",
            "green"
        )
    )

    subjects_outfits = np.loadtxt(
        f"./clusters/lst/subjects_{args.split}.txt", dtype=str, delimiter=" "
    )[:, 1:]
    subfolders = [
        f"./data/PuzzleIOI_4views/{args.method}_{args.tag}/{subject}/{outfit}"
        for subject, outfit in subjects_outfits
    ]

    def load_img(file, device):

        output = torchvision.io.read_image(file).to(device)[:3, :, :].unsqueeze(0).float()
        mask = torchvision.io.read_image(file).to(device)[3, :, :].unsqueeze(0).float() / 255.0
        output = output * mask + (1.0 - mask) * 255.0 * 0.5
        
        output = output / 255.0
        output = output * 2.0 - 1.0

        return output

    for subfolder in tqdm(subfolders):
        
        for rgb_file in glob(f"{subfolder}/render/*.png"):

            T_normal_F_path = f"{subfolder.replace(f'{args.method}_{args.tag}', 'fitting')}/body/{os.path.basename(rgb_file).replace('.png', '_F.png')}"
            T_normal_B_path = f"{subfolder.replace(f'{args.method}_{args.tag}', 'fitting')}/body/{os.path.basename(rgb_file).replace('.png', '_B.png')}"

            out_normal_F_path = f"{subfolder}/normal_est/{os.path.basename(rgb_file)}"
            
            if not os.path.exists(out_normal_F_path) or args.overwrite:
            
                os.makedirs(os.path.dirname(out_normal_F_path), exist_ok=True)

                # load image
                rgb = load_img(rgb_file, device)
                T_normal_F = load_img(T_normal_F_path, device)
                T_normal_B = load_img(T_normal_B_path, device)
                in_tensor = {"image": rgb, "T_normal_F": T_normal_F, "T_normal_B": T_normal_B}

                # forward pass

                with torch.no_grad():
                    in_tensor["normal_F"], in_tensor["normal_B"] = normal_net.netG(in_tensor)
                    
                    
                # save result
                torchvision.utils.save_image((in_tensor["normal_F"] + 1.0) / 2.0, out_normal_F_path)
