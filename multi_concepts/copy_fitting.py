import os
import sys
import json
import numpy as np
from shutil import copyfile
from glob import glob
from tqdm import tqdm

# tgt_root = "./data/PuzzleIOI/fitting"
# src_root = "/ps/scratch/priyanka/alignment_yuliang_corrected"

# # copy priyanak's fitting results to the fitting folder
# for person in tqdm(os.listdir(tgt_root)):
#     for motion in os.listdir(os.path.join(tgt_root, person)):
#         if os.path.isdir(os.path.join(tgt_root, person, motion)):
#             cur_output_dir = os.path.join(tgt_root, person, motion, "smplx")
#             os.makedirs(cur_output_dir, exist_ok=True)
#             if motion == 'apose':
#                 src_obj = os.path.join(src_root, person, f"{person}_1_0.obj")
#                 src_pkl = os.path.join(src_root, person, f"{person}_1_0.pkl")
#                 src_obj_files = [src_obj]
#                 src_pkl_files = [src_pkl]
#             else:
#                 src_obj_files = glob(f"{src_root}/{person}/*{motion}_seq*.obj")
#                 src_pkl_files = glob(f"{src_root}/{person}/*{motion}_seq*.pkl")
#                 src_obj = src_obj_files[0]
#                 src_pkl = src_pkl_files[0]

#             tgt_obj = os.path.join(cur_output_dir, "smplx.obj")
#             tgt_pkl = os.path.join(cur_output_dir, "smplx.pkl")
            
#             if os.path.exists(src_obj) and os.path.exists(
#                 src_pkl
#             ) and len(src_obj_files) == len(src_pkl_files) == 1:
#                 # if not os.path.exists(tgt_obj):
#                 copyfile(src_obj, tgt_obj)
#                 # if not os.path.exists(tgt_pkl):
#                 copyfile(src_pkl, tgt_pkl)
#             else:
#                 print(f"{src_obj} or {src_pkl} not exists \n")
                
                
tgt_root = "./data/PuzzleIOI/puzzle_cam"
src_root = "./data/PuzzleIOI/puzzle"

# copy priyanak's fitting results to the fitting folder
for person in tqdm(os.listdir(tgt_root)):
    for motion in os.listdir(os.path.join(tgt_root, person)):
        src_path = os.path.join(src_root, person, motion, "gpt4v_response.json")
        tgt_path = os.path.join(tgt_root, person, motion, "gpt4v_response.json")
        if os.path.exists(src_path):
            copyfile(src_path, tgt_path)
        else:
            print(f"{src_path} not exists \n")
           
