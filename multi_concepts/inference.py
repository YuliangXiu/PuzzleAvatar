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
from diffusers import DiffusionPipeline, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from peft import PeftModel

negative_prompt = 'unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy, shadow, worst quality, jpeg artifacts, poorly drawn, dehydrated, noisy, poorly drawn, bad proportions, bad anatomy, bad lighting, bad composition, bad framing, fused fingers, noisy, duplicate characters'

class BreakASceneInference:
    def __init__(self):

        self.classes = None
        self.tokens = None
        self.descs = None
        self.gender = None

        self._parse_args()
        self._load_meta_data()
        self._load_pipeline()

    def _load_meta_data(self):

        with open(os.path.join(self.args.instance_dir, 'gpt4v_response.json'), 'r') as f:
            gpt4v_response = json.load(f)
            self.gender = 'man' if gpt4v_response['gender'] in ['man', 'male'] else 'woman'

            self.classes = list(gpt4v_response.keys())

            self.classes.remove("gender")
            for key in ["eyeglasses", "sunglasses", "glasses"]:
                if key in self.classes:
                    self.classes.remove(key)

            self.tokens = [f"<asset{i}>" for i in range(len(self.classes))]
            self.descs = [gpt4v_response[cls] for cls in self.classes]

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
        parser.add_argument("--model_dir", type=str, required=True)
        parser.add_argument("--instance_dir", type=str, required=True)
        parser.add_argument("--num_samples", type=int, required=True)
        parser.add_argument("--step", type=str, default="")
        parser.add_argument("--output_dir", type=str, default="outputs/result.jpg")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--use_peft", type=str, default="none")
        parser.add_argument("--use_shape_description", action="store_true")
        self.args = parser.parse_args()

        self.args.output_dir = os.path.join(self.args.model_dir, "output")
        os.makedirs(self.args.output_dir, exist_ok=True)

    def _load_pipeline(self):

        person_id = self.args.model_dir.split("/")[-1]

        if self.args.use_peft != "none":

            del_files = ["config.json", "model.safetensors", "diffusion_pytorch_model.safetensors"]

            for del_file in del_files:
                if os.path.exists(os.path.join(self.args.model_dir, "text_encoder", del_file)):
                    os.remove(os.path.join(self.args.model_dir, "text_encoder", del_file))
                if os.path.exists(os.path.join(self.args.model_dir, "unet", del_file)):
                    os.remove(os.path.join(self.args.model_dir, "unet", del_file))

            self.pipeline = DiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                torch_dtype=torch.float32,
                requires_safety_checker=False,
            )

            self.pipeline.scheduler = DDIMScheduler.from_pretrained(
                self.args.pretrained_model_name_or_path, subfolder="scheduler"
            )

            num_added_tokens = self.pipeline.tokenizer.add_tokens(self.tokens)
            print(f"Added {num_added_tokens} tokens")
            self.pipeline.text_encoder.resize_token_embeddings(len(self.pipeline.tokenizer))

            self.pipeline.text_encoder = PeftModel.from_pretrained(
                self.pipeline.text_encoder,
                os.path.join(self.args.model_dir, "text_encoder", self.args.step),
                adapter_name=person_id,
            )

            self.pipeline.unet = PeftModel.from_pretrained(
                self.pipeline.unet,
                os.path.join(self.args.model_dir, "unet", self.args.step),
                adapter_name=person_id,
            )

        else:

            self.pipeline = DiffusionPipeline.from_pretrained(
                self.args.model_dir,
                torch_dtype=torch.float32,
                requires_safety_checker=False,
            )

        num_added_tokens = self.pipeline.tokenizer.add_tokens(self.tokens)
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
                num_inference_steps=30
            ).images

            for idx in range(len(images)):
                images[idx].save(
                    os.path.join(
                        self.args.output_dir, f"img{idx+cur_size:02d}_{tokens[idx+cur_size]}.png"
                    )
                )

            cur_size += batch_size

    def construct_prompt(self, idx):

        classes_to_use = [self.classes[i] for i in idx]
        tokens_to_use = [self.tokens[i] for i in idx]
        descs_to_use = [self.descs[i] for i in idx]

        # formulate the prompt and prompt_raw
        prompt_head = f"a high-resolution DSLR colored image of a {self.gender}"
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
    with_ids = []
    for cls in ['face', 'haircut', 'hair']:
        if cls in break_a_scene_inference.classes:
            with_ids.append(break_a_scene_inference.classes.index(cls))

    for i in range(break_a_scene_inference.args.num_samples):
        num_of_tokens = random.randrange(1, len(break_a_scene_inference.classes) + 1)
        tokens_ids_to_use = sorted(
            random.sample(range(len(break_a_scene_inference.classes)), k=num_of_tokens)
        )

        prompt, prompt_raw, classes_to_use = break_a_scene_inference.construct_prompt(
            tokens_ids_to_use
        )

        tokens.append(f"img{i:02d}_" + "_".join(classes_to_use))
        tokens.append(f"img{i:02d}_" + "_".join(classes_to_use) + "_raw")

        prompts.append(prompt)
        prompts.append(prompt_raw)

    print(prompts)

    break_a_scene_inference.infer_and_save(prompts=prompts, tokens=tokens)
