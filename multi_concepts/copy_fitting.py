import os
import sys
import json
import cv2
import numpy as np
from shutil import copyfile
from glob import glob
from tqdm import tqdm

tgt_root = "./data/PuzzleIOI/fitting"
src_root = "/ps/scratch/priyanka/alignment_yuliang_corrected"

# copy priyanak's fitting results to the fitting folder
for person in tqdm(os.listdir(tgt_root)):
    for motion in os.listdir(os.path.join(tgt_root, person)):
        if os.path.isdir(os.path.join(tgt_root, person, motion)):
            cur_output_dir = os.path.join(tgt_root, person, motion, "smplx")
            os.makedirs(cur_output_dir, exist_ok=True)
            if motion == 'apose':
                src_obj = os.path.join(src_root, person, f"{person}_1_0.obj")
                src_pkl = os.path.join(src_root, person, f"{person}_1_0.pkl")
                src_obj_files = [src_obj]
                src_pkl_files = [src_pkl]
            else:
                src_obj_files = glob(f"{src_root}/{person}/*{motion}_seq*_0_0.obj")
                src_pkl_files = glob(f"{src_root}/{person}/*{motion}_seq*_0_0.pkl")
                src_obj = src_obj_files[0]
                src_pkl = src_pkl_files[0]

            tgt_obj = os.path.join(cur_output_dir, "smplx.obj")
            tgt_pkl = os.path.join(cur_output_dir, "smplx.pkl")

            if os.path.exists(src_obj) and os.path.exists(
                src_pkl
            ) and len(src_obj_files) == len(src_pkl_files) == 1:
                # if os.path.exists(tgt_obj):
                #     os.remove(tgt_obj)
                copyfile(src_obj, tgt_obj)
                # if os.path.exists(tgt_pkl):
                #     os.remove(tgt_pkl)
                copyfile(src_pkl, tgt_pkl)
            else:
                print(f"{src_obj} or {src_pkl} not exists \n")

# tgt_root = "./data/PuzzleIOI/puzzle_cam"
# src_root = "./data/PuzzleIOI/puzzle"

# # copy priyanak's fitting results to the fitting folder
# for person in tqdm(os.listdir(tgt_root)):
#     for motion in os.listdir(os.path.join(tgt_root, person)):
#         src_path = os.path.join(src_root, person, motion, "gpt4v_response.json")
#         tgt_path = os.path.join(tgt_root, person, motion, "gpt4v_response.json")
#         if os.path.exists(src_path):
#             copyfile(src_path, tgt_path)
#         else:
#             print(f"{src_path} not exists \n")

# src_root = "/ps/data/DynamicClothCap"
# tgt_root = "./data/PuzzleIOI/fitting"

# pbar = tqdm(glob(f"{tgt_root}/*/outfit*"))

# for tgt_outfit_dir in pbar:
#     person, outfit = tgt_outfit_dir.split("/")[-2:]
#     pbar.set_description(f"{person}/{outfit}")
#     src_outfit_dir = glob(f"{src_root}/DynamicClothCap*{person}_*")[0]
#     tgt_img_path = glob(f"{tgt_outfit_dir}/*.jpg")[0]
#     tgt_mtl_path = glob(f"{tgt_outfit_dir}/*.mtl")[0]
#     tgt_obj_path = glob(f"{tgt_outfit_dir}/*.obj")[0]
#     tgt_outfit, tgt_frame = tgt_mtl_path.split("/")[-1].split(".")[-3:-1]
    
#     src_img_path = os.path.join(
#         src_outfit_dir, tgt_outfit,
#         f"meshes/{tgt_outfit}.{tgt_frame}.jpg"
#     )
#     src_mtl_path = os.path.join(
#         src_outfit_dir, tgt_outfit,
#         f"meshes/{tgt_outfit}.{tgt_frame}.mtl"
#     )
#     src_obj_path = os.path.join(
#         src_outfit_dir, tgt_outfit,
#         f"meshes/{tgt_outfit}.{tgt_frame}.obj"
#     )
    
#     if not os.path.exists(src_img_path):
#         src_outfit_dir = glob(f"{src_root}/DynamicClothCap*{person}_*")[1]
#         src_img_path = os.path.join(
#             src_outfit_dir, tgt_outfit,
#             f"meshes/{tgt_outfit}.{tgt_frame}.jpg"
#         )
#         src_mtl_path = os.path.join(
#             src_outfit_dir, tgt_outfit,
#             f"meshes/{tgt_outfit}.{tgt_frame}.mtl"
#         )
#         src_obj_path = os.path.join(
#             src_outfit_dir, tgt_outfit,
#             f"meshes/{tgt_outfit}.{tgt_frame}.obj"
#         )
#     try:
#         import trimesh
#         mesh = trimesh.load(tgt_obj_path)
#     except:
#         print(tgt_obj_path)
#         os.remove(tgt_obj_path)
#         copyfile(src_obj_path, tgt_obj_path)
    
#     from PIL import Image
#     try:
#         img=Image.open(tgt_img_path).convert('RGB')
#     except:
#         if os.path.exists(tgt_img_path):
#             os.remove(tgt_img_path)
#             os.remove(tgt_mtl_path)
#         copyfile(src_img_path, tgt_img_path)
#         copyfile(src_mtl_path, tgt_mtl_path)
#         print(f"{tgt_img_path}: {src_img_path}")
        
