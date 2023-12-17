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
from diffusers import DDIMScheduler, DiffusionPipeline


class BreakASceneInference:
    def __init__(self):
        self._parse_args()
        self._load_pipeline()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_dir", type=str, required=True)
        parser.add_argument("--instance_dir", type=str, required=True)
        parser.add_argument("--num_samples", type=int, required=True)
        parser.add_argument("--output_dir", type=str, default="outputs/result.jpg")
        parser.add_argument("--device", type=str, default="cuda")
        self.args = parser.parse_args()

        self.args.output_dir = os.path.join(self.args.model_dir, "output")

    def _load_pipeline(self):
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.args.model_dir,
            torch_dtype=torch.float16,
        )
        self.pipeline.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.pipeline.to(self.args.device)

    @torch.no_grad()
    def infer_and_save(self, prompts, tokens):
        images = self.pipeline(prompts).images
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, exist_ok=True)
        for idx in range(len(images)):
            print(prompts[idx])
            images[idx].save(os.path.join(self.args.output_dir, f"img{idx:02d}_{tokens[idx]}.png"))


if __name__ == "__main__":
    break_a_scene_inference = BreakASceneInference()
    with open(
        os.path.join(break_a_scene_inference.args.instance_dir, 'gpt4v_response.json'), 'r'
    ) as f:
        gpt4v_response = json.load(f)

    gender = 'man' if gpt4v_response['gender'] in ['man', 'male'] else 'woman'
    classes = list(gpt4v_response.keys())
    classes.remove("gender")
    placeholders = [f"<asset{i}>" for i in range(len(classes))]
    prompt_words = [
        f"{placeholders[i]} {gpt4v_response[classes[i]]} {classes[i]}" for i in range(len(classes))
    ]

    prompts = []
    tokens = []
    with_ids = [classes.index(cls) for cls in ['face', 'haircut']]

    for i in range(break_a_scene_inference.args.num_samples):
        num_of_tokens = random.randrange(1, len(classes) + 1)
        tokens_ids_to_use = sorted(random.sample(range(len(classes)), k=num_of_tokens))
        prompt_head = f"a high-resolution DSLR image of {gender}, walking on the beach, "
        tokens.append("_".join([f"{id}_{classes[id]}" for id in tokens_ids_to_use]))

        if not np.isin(with_ids, tokens_ids_to_use).any():
            prompt_garments = "wearing " + " and ".join([
                prompt_words[i] for i in tokens_ids_to_use
            ]) + "."
        else:
            for with_id in with_ids:
                if with_id in tokens_ids_to_use:
                    prompt_head += f"with {prompt_words[with_id]}, "
            prompt_garments = "wearing " + " and ".join([
                prompt_words[id] for id in tokens_ids_to_use if id not in with_ids
            ]) + "."
            prompt_garments = prompt_garments.replace(", wearing .", ".")

        prompts.append(f"{prompt_head}{prompt_garments}")

    break_a_scene_inference.infer_and_save(prompts=prompts, tokens=tokens)
