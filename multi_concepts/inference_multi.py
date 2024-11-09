"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import json
import os
import random
import numpy as np

import torch
from diffusers import DiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from peft import PeftModel

negative_prompt = 'unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy, shadow, worst quality, jpeg artifacts, poorly drawn, dehydrated, noisy, poorly drawn, bad proportions, bad anatomy, bad lighting, bad composition, bad framing, fused fingers, noisy, duplicate characters'


class BreakASceneInference:
    def __init__(self):

        self.classes = []
        self.tokens = []
        self.descs = []
        self.gender = []

        self.instance_dirs = []

        self._parse_args()
        self._load_meta_data()
        self._load_pipeline()

    def _load_meta_data(self):

        token_id = 0

        multi_dict = {}

        for instance_dir in self.instance_dirs:

            with open(os.path.join(instance_dir, 'gpt4v_simple.json'), 'r') as f:
                gpt4v_response = json.load(f)
                cur_gender = 'man' if gpt4v_response['gender'] in ['man', 'male'] else 'woman'

                self.classes.append(list(gpt4v_response.keys()))

                for key in ["gender", "eyeglasses", "sunglasses", "glasses"]:
                    if key in self.classes[-1]:
                        self.classes[-1].remove(key)

                self.tokens.append([f"<asset{i+token_id}>" for i in range(len(self.classes[-1]))])
                self.descs.append([gpt4v_response[cls] for cls in self.classes[-1]])
                self.gender.append(cur_gender)
                token_id += len(self.classes[-1])

            so_name = "_".join(instance_dir.split("/")[-2:])
            multi_dict[so_name] = {}
            multi_dict[so_name]["classes"] = self.classes[-1]
            multi_dict[so_name]["tokens"] = self.tokens[-1]
            multi_dict[so_name]["descs"] = self.descs[-1]

        np.save(f"./results/multi/{self.args.so_lst}/multi_dict.npy", multi_dict, allow_pickle=True)

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
        parser.add_argument("--model_dir", type=str, required=True)
        parser.add_argument("--so_lst", type=str, required=True)
        parser.add_argument("--num_samples", type=int, required=True)
        parser.add_argument("--step", type=str, default="")
        parser.add_argument("--output_dir", type=str, default="outputs/result.jpg")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--use_peft", type=str, default="none")
        parser.add_argument("--use_shape_description", action="store_true")
        self.args = parser.parse_args()

        self.args.output_dir = os.path.join(self.args.model_dir, "output")
        so_lst = self.args.so_lst.split("_")
        for (subject, outfit) in zip(so_lst[0::2], so_lst[1::2]):
            if "human" == subject:
                self.instance_dirs.append(f"./data/{subject}/{outfit}")
            else:
                self.instance_dirs.append(f"./data/PuzzleIOI/puzzle_capture/{subject}/{outfit}")

        os.makedirs(self.args.output_dir, exist_ok=True)

    def _load_pipeline(self):

        all_tokens = []
        for tokens_group in self.tokens:
            all_tokens += tokens_group

        self.pipeline = DiffusionPipeline.from_pretrained(
            self.args.model_dir,
            torch_dtype=torch.float32,
            requires_safety_checker=False,
        )

        num_added_tokens = self.pipeline.tokenizer.add_tokens(all_tokens)
        print(f"Added {num_added_tokens} tokens")
        self.pipeline.text_encoder.resize_token_embeddings(len(self.pipeline.tokenizer))

        if is_xformers_available():
            self.pipeline.unet.enable_xformers_memory_efficient_attention()

        # self.pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.4, b2=1.6)
        self.pipeline.text_encoder.eval()
        self.pipeline.unet.eval()
        self.pipeline.to(self.args.device)

    @torch.no_grad()
    def infer_and_save(self, prompts, tokens):

        group_size = 8

        os.makedirs(self.args.output_dir, exist_ok=True)

        cur_size = 0

        for prompt_group in np.array_split(prompts, len(prompts) // group_size):

            prompt_group = list(prompt_group)
            batch_size = len(prompt_group)

            images = self.pipeline(
                prompt_group,
                negative_prompt=[negative_prompt] * batch_size,
                guidance_scale=7.5,
                num_inference_steps=50
            ).images

            for idx in range(len(images)):
                images[idx].save(
                    os.path.join(
                        self.args.output_dir, f"img{idx+cur_size:02d}_{tokens[idx+cur_size]}.png"
                    )
                )

            cur_size += batch_size

    def construct_prompt(self, idx, so_idx=None):

        cur_gender = None

        if so_idx is None:

            classes_to_use = [self.classes[i] for i in idx]
            tokens_to_use = [self.tokens[i] for i in idx]
            descs_to_use = [self.descs[i] for i in idx]
            cur_gender = self.gender

        else:
            classes_to_use = [self.classes[so_idx][i] for i in idx]
            tokens_to_use = [self.tokens[so_idx][i] for i in idx]
            descs_to_use = [self.descs[so_idx][i] for i in idx]
            cur_gender = self.gender[so_idx]

        # formulate the prompt and prompt_raw
        prompt_head = f"a high-resolution DSLR colored image of a {cur_gender}"
        facial_classes = ['face', 'haircut', 'hair']
        with_classes = [cls for cls in classes_to_use if cls in facial_classes]
        wear_classes = [cls for cls in classes_to_use if cls not in facial_classes]

        prompt_raw = prompt = f"{prompt_head}, "

        for class_token in with_classes:
            idx = classes_to_use.index(class_token)

            if len(wear_classes) == 0 and with_classes.index(class_token) == len(with_classes) - 1:
                ending = "."
            else:
                ending = ", "

            if self.args.use_shape_description:
                prompt += f"{tokens_to_use[idx]} {descs_to_use[idx]} {class_token}{ending}"
                prompt_raw += f"{descs_to_use[idx]} {class_token}{ending}"
            else:
                prompt += f"{tokens_to_use[idx]} {class_token}{ending}"
                prompt_raw += f"{class_token}{ending}"

        if len(wear_classes) > 0:
            prompt += "wearing "
            prompt_raw += "wearing "

            for class_token in wear_classes:
                idx = classes_to_use.index(class_token)

                if wear_classes.index(class_token) < len(wear_classes) - 2:
                    ending = ", "
                elif wear_classes.index(class_token) == len(wear_classes) - 2:
                    ending = ", and "
                else:
                    ending = "."
                if self.args.use_shape_description:
                    prompt += f"{tokens_to_use[idx]} {descs_to_use[idx]} {class_token}{ending}"
                    prompt_raw += f"{descs_to_use[idx]} {class_token}{ending}"
                else:
                    prompt += f"{tokens_to_use[idx]} {class_token}{ending}"
                    prompt_raw += f"{class_token}{ending}"

        return prompt, prompt_raw, classes_to_use


if __name__ == "__main__":

    break_a_scene_inference = BreakASceneInference()

    prompts = []
    tokens = []

    for i in range(break_a_scene_inference.args.num_samples):

        so_idx = random.randint(0, len(break_a_scene_inference.instance_dirs) - 1)

        num_of_tokens = random.randrange(1, len(break_a_scene_inference.classes[so_idx]) + 1)
        tokens_ids_to_use = sorted(
            random.sample(range(len(break_a_scene_inference.classes[so_idx])), k=num_of_tokens)
        )
        full_tokens_ids_to_use = list(range(len(break_a_scene_inference.classes[so_idx])))

        prompt, prompt_raw, classes_to_use = break_a_scene_inference.construct_prompt(
            tokens_ids_to_use, so_idx
        )

        prompt_full, prompt_raw_full, classes_to_use_full = break_a_scene_inference.construct_prompt(
            full_tokens_ids_to_use, so_idx
        )

        tokens.append(f"asset_{i:02d}_" + "_".join(classes_to_use))
        tokens.append(f"asset_{i:02d}_" + "_".join(classes_to_use) + "_raw")
        tokens.append(f"full_{i:02d}_" + "_".join(classes_to_use_full))
        tokens.append(f"full_{i:02d}_" + "_".join(classes_to_use_full) + "_raw")

        prompts.append(prompt)
        prompts.append(prompt_raw)
        prompts.append(prompt_full)
        prompts.append(prompt_raw_full)

    # print(prompts)

    break_a_scene_inference.infer_and_save(prompts=prompts, tokens=tokens)
