import os
import sys
import json
import numpy as np
from shutil import copyfile
from glob import glob
from tqdm import tqdm
from multi_concepts.grounding_dino_sam import gpt4v_captioning

tgt_root = "./data/multi_concepts_data/thuman2"
src_root = "/home/yxiu/Code/DC-PIFu/data/thuman2_36views"

all_src_rgbs_dirs = sorted(glob(f"{src_root}/*/render"))
pbar = tqdm(all_src_rgbs_dirs)

for src_rgb_dir in pbar:
    pbar.set_description(f"Processing {src_rgb_dir.split('/')[-2]}")

    json_path = os.path.join(src_rgb_dir, "../gpt4v_response.json")

    # if there is no gpt4v_response.json, then generate one
    if not os.path.exists(json_path):
        try:
            gpt4v_response = gpt4v_captioning(src_rgb_dir)
            with open(json_path, "w") as f:
                f.write(gpt4v_response)
            gpt4v_dict = json.loads(gpt4v_response)
            assert isinstance(gpt4v_dict, dict)
        except Exception as e:
            print(src_rgb_dir, e)
            if os.path.exists(json_path):
                os.remove(json_path)
    else:
        with open(json_path, "r") as f:
            gpt4v_response = f.read()
        gpt4v_dict = json.loads(gpt4v_response)

    gender = 'man' if gpt4v_dict["gender"] == "male" else "woman"

    picked_rgbs = np.random.choice(glob(f"{src_rgb_dir}/*"), 3, replace=False)

    for picked_rgb in picked_rgbs:

        picked_norm = picked_rgb.replace("render", "normal_F")

        tgt_rgb = os.path.join(
            tgt_root, gender, f"{src_rgb_dir.split('/')[-2]}_{picked_rgb.split('/')[-1]}"
        )
        tgt_norm = os.path.join(
            tgt_root, f"{gender}_norm", f"{src_rgb_dir.split('/')[-2]}_{picked_rgb.split('/')[-1]}"
        )
        tgt_json = os.path.join(
            tgt_root, f"{gender}_desc",
            f"{src_rgb_dir.split('/')[-2]}_{picked_rgb.split('/')[-1].replace('png', 'json')}"
        )
        os.makedirs(os.path.dirname(tgt_rgb), exist_ok=True)
        os.makedirs(os.path.dirname(tgt_json), exist_ok=True)
        os.makedirs(os.path.dirname(tgt_norm), exist_ok=True)

        copyfile(picked_rgb, tgt_rgb)
        copyfile(json_path, tgt_json)
        copyfile(picked_norm, tgt_norm)
