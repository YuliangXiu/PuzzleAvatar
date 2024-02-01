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


class BreakASceneInference:
    def __init__(self):

        self.prompt_words = None
        self.classes = None
        self.placeholder_tokens = None
        self.gender = None

        self._parse_args()
        self._load_prompts()
        self._load_pipeline()

    def _load_prompts(self):

        with open(os.path.join(self.args.instance_dir, 'gpt4v_response.json'), 'r') as f:
            gpt4v_response = json.load(f)
            self.gender = 'man' if gpt4v_response['gender'] in ['man', 'male'] else 'woman'

            self.classes = list(gpt4v_response.keys())

            self.classes.remove("gender")
            for key in ["eyeglasses", "sunglasses", "glasses"]:
                if key in self.classes:
                    self.classes.remove(key)

            self.placeholder_tokens = [f"<asset{i}>" for i in range(len(self.classes))]

            if self.args.use_shape_description:
                self.prompt_words = [
                    f"{self.placeholder_tokens[i]} {gpt4v_response[self.classes[i]]} {self.classes[i]}"
                    for i in range(len(self.classes))
                ]
            else:
                self.prompt_words = [
                    f"{self.placeholder_tokens[i]} {self.classes[i]}"
                    for i in range(len(self.classes))
                ]

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

            num_added_tokens = self.pipeline.tokenizer.add_tokens(self.placeholder_tokens)
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

        num_added_tokens = self.pipeline.tokenizer.add_tokens(self.placeholder_tokens)
        print(f"Added {num_added_tokens} tokens")
        self.pipeline.text_encoder.resize_token_embeddings(len(self.pipeline.tokenizer))

        if is_xformers_available():
            self.pipeline.unet.enable_xformers_memory_efficient_attention()

        self.pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.4, b2=1.6)
        self.pipeline.text_encoder.eval()
        self.pipeline.unet.eval()
        self.pipeline.to(self.args.device)

    @torch.no_grad()
    def infer_and_save(self, prompts, tokens):

        negative_prompt = 'unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy, shadow, worst quality, jpeg artifacts, poorly drawn, dehydrated, noisy, poorly drawn, bad proportions, bad anatomy, bad lighting, bad composition, bad framing, fused fingers, noisy, duplicate characters'

        group_size = 8
        
        for group_idx, prompt_group in enumerate(
            np.array_split(prompts,
                           len(prompts) // group_size)
        ):

            prompt_group = list(prompt_group)
            batch_size = len(prompt_group)

            images = self.pipeline(
                prompt_group,
                negative_prompt=[negative_prompt] * batch_size,
                guidance_scale=7.5,
                num_inference_steps=30
            ).images

            os.makedirs(self.args.output_dir, exist_ok=True)

            for idx in range(len(images) // 2):
                images[2 * idx + group_idx * batch_size].save(
                    os.path.join(
                        self.args.output_dir, f"img{idx+group_idx*group_size:02d}_{tokens[idx]}.png"
                    )
                )
                images[2 * idx + group_idx * batch_size + 1].save(
                    os.path.join(
                        self.args.output_dir,
                        f"img{idx+group_idx*group_size:02d}_{tokens[idx]}_raw.png"
                    )
                )


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
        prompt_head = f"a high-resolution DSLR colored image of a {break_a_scene_inference.gender}, "
        prompt_head_raw = f"a high-resolution DSLR colored image of a {break_a_scene_inference.gender}, "
        tokens.append(
            "_".join([f"{id}_{break_a_scene_inference.classes[id]}" for id in tokens_ids_to_use])
        )

        if not np.isin(with_ids, tokens_ids_to_use).any():
            prompt_garments = "wearing " + " and ".join([
                break_a_scene_inference.prompt_words[i] for i in tokens_ids_to_use
            ]) + " at the beach."
            prompt_garments_raw = "wearing " + " and ".join([
                break_a_scene_inference.classes[i] for i in tokens_ids_to_use
            ]) + " at the beach."
        else:
            for with_id in with_ids:
                if with_id in tokens_ids_to_use:
                    prompt_head += f"{break_a_scene_inference.prompt_words[with_id]}, "
                    prompt_head_raw += f"{break_a_scene_inference.classes[with_id]}, "

            prompt_garments = "wearing " + " and ".join([
                break_a_scene_inference.prompt_words[id]
                for id in tokens_ids_to_use if id not in with_ids
            ]) + " at the beach."

            prompt_garments_raw = "wearing " + " and ".join([
                break_a_scene_inference.classes[id]
                for id in tokens_ids_to_use if id not in with_ids
            ]) + " at the beach."

        prompts.append(f"{prompt_head}{prompt_garments}".replace(", wearing .", "."))
        prompts.append(f"{prompt_head_raw}{prompt_garments_raw}".replace(", wearing .", "."))

    print(prompts)

    break_a_scene_inference.infer_and_save(prompts=prompts, tokens=tokens)
