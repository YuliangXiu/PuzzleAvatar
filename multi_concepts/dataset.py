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

import glob
import json
import os
import random
from typing import Union
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
from kornia.morphology import dilation
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
        self,
        instance_data_root,
        placeholder_tokens,
        tokenizer,
        initializer_tokens,
        gpt4v_response,
        use_shape_desc,
        with_prior_preservation,
        num_class_images,
        gender,
        class_data_root=None,
        size=512,
        center_crop=False,
        flip_p=0.5,
        sd_version="stable-diffusion-2-1",
        use_view_prompt=True,
        use_full_shot=True,
        use_half_data=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.flip_p = flip_p
        self.sd_version = sd_version
        self.gpt4v_response = gpt4v_response
        self.use_shape_desc = use_shape_desc
        self.gender = gender
        self.with_prior_preservation = with_prior_preservation
        self.use_view_prompt = use_view_prompt

        self.cam_view_dict = {
            "01": ["side", "front"],
            "02": ["side", "front"],
            "03": ["side", "front"],
            "04": ["overhead", "side"],
            "05": ["front"],
            "06": ["front"],
            "07": ["front"],
            "08": ["overhead", "front"],
            "09": ["side", "front"],
            "10": ["side", "front"],
            "11": ["side", "front"],
            "12": ["overhead", "side"],
            "13": ["side", "back"],
            "14": ["side", "back"],
            "15": ["side", "back"],
            "16": ["overhead", "back"],
            "17": ["back"],
            "18": ["overhead", "back"],
            "19": ["side", "back"],
            "20": ["side", "back"],
            "21": ["side", "back"],
            "22": ["overhead", "side"],
        }

        self.image_transforms = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
        ])

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.placeholder_full = placeholder_tokens
        self.placeholder_tokens = []
        self.class_tokens = []
        self.initializer_tokens = initializer_tokens
        self.class_data_root = class_data_root
        self.num_class_images = num_class_images

        instance_img_paths = sorted(glob.glob(f"{instance_data_root}/image/*[!raw].[jp][pn]g"))

        if not use_full_shot:
            instance_img_paths = [
                filename for filename in instance_img_paths if "_07_C" not in filename
            ]

        if use_half_data:
            instance_len = len(instance_img_paths)
            instance_img_paths = instance_img_paths[:instance_len // 2]

        self.instance_images = []
        self.instance_masks = []
        self.instance_descriptions = []
        self.instance_views = []

        for instance_img_path in instance_img_paths:
            instance_idx = instance_img_path.split("/")[-1].split(".")[0]
            instance_mask_paths = glob.glob(f"{instance_data_root}/mask/{instance_idx}_*.png")

            if len(instance_mask_paths) > 0:

                if "PuzzleIOI" in instance_img_path and self.use_view_prompt:
                    cam_id = instance_img_path.split("/")[-1].split("_")[1]
                    self.instance_views.append(self.cam_view_dict[cam_id])
                else:
                    self.instance_views.append([])

                self.instance_images.append(
                    self.image_transforms(Image.open(instance_img_path))[:3]
                )
                instance_mask = []
                instance_placeholder_token = []
                instance_class_tokens = []
                instance_description = []

                for instance_mask_path in instance_mask_paths:
                    curr_mask = Image.open(instance_mask_path)
                    curr_mask = self.mask_transforms(curr_mask)[0, None, None, ...]
                    instance_mask.append(curr_mask)
                    curr_cls = instance_mask_path.split(".")[0].split("_")[-1]

                    instance_placeholder_token.append(
                        self.placeholder_full[self.initializer_tokens.index(curr_cls)]
                    )
                    instance_class_tokens.append(curr_cls)
                    instance_description.append(self.gpt4v_response[curr_cls])

                self.instance_masks.append(torch.cat(instance_mask))
                self.placeholder_tokens.append(instance_placeholder_token)
                self.class_tokens.append(instance_class_tokens)
                self.instance_descriptions.append(instance_description)

        self.class_images_path = {}
        self.class_normals_path = {}
        self.syn_images_path = {}
        self.syn_normals_path = {}
        self.syn_json_path = {}

        if self.class_data_root is not None and self.with_prior_preservation:
            class_rgb_dir = Path(class_data_root) / self.sd_version / self.gender

            self.class_images_path[self.gender] = [
                item
                for item in sorted(list(class_rgb_dir.iterdir())) if str(item).endswith(".jpg")
            ]

            syn_dataset = "thuman2_orbit"
            syn_rgb_dir = Path(class_data_root) / syn_dataset / self.gender
            syn_normal_dir = Path(class_data_root) / syn_dataset / f"{self.gender}_norm"
            syn_json_dir = Path(class_data_root) / syn_dataset / f"{self.gender}_desc"

            self.syn_images_path[self.gender] = [
                item for item in sorted(list(syn_rgb_dir.iterdir())) if str(item).endswith(".png")
            ]
            self.syn_normals_path[self.gender] = [
                item
                for item in sorted(list(syn_normal_dir.iterdir())) if str(item).endswith(".png")
            ]
            self.syn_json_path[self.gender] = [
                item
                for item in sorted(list(syn_json_dir.iterdir())) if str(item).endswith(".json")
            ]
        else:
            self.class_data_root = None

        self.num_syn_images = len(self.syn_images_path[self.gender])
        self._length = max(len(self.instance_images), self.num_class_images, self.num_syn_images)

    def __len__(self):

        return self._length

    def construct_prompt(
        self, classes_to_use, tokens_to_use, descs_to_use, views_to_use, is_face=False
    ):

        replace_for_normal = lambda x: x.replace(
            "a high-resolution DSLR colored image of ", "a detailed sculpture of "
        )

        # formulate the prompt and prompt_raw
        if is_face:
            prompt_head = f"a high-resolution DSLR colored image of the headshot of a {self.gender}"
        else:
            prompt_head = f"a high-resolution DSLR colored image of a {self.gender}"

        facial_classes = ['face', 'haircut', 'hair']
        with_classes = [cls for cls in classes_to_use if cls in facial_classes]
        wear_classes = [cls for cls in classes_to_use if cls not in facial_classes]

        prompt_raw = prompt = f"{prompt_head}, "

        if len(views_to_use) > 0:
            view_prompt = ", ".join([f"{view} view" for view in views_to_use])
        else:
            view_prompt = ""

        for class_token in with_classes:
            idx = classes_to_use.index(class_token)

            if len(wear_classes) == 0 and with_classes.index(class_token) == len(with_classes) - 1:
                ending = f", {view_prompt}."
            else:
                ending = ", "

            if self.use_shape_desc:
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
                    ending = f", {view_prompt}."

                if self.use_shape_desc:
                    prompt += f"{tokens_to_use[idx]} {descs_to_use[idx]} {class_token}{ending}"
                    prompt_raw += f"{descs_to_use[idx]} {class_token}{ending}"
                else:
                    prompt += f"{tokens_to_use[idx]} {class_token}{ending}"
                    prompt_raw += f"{class_token}{ending}"

        prompt_dict = {
            "prompt": prompt,
            "prompt_raw": prompt_raw,
            "prompt_norm": replace_for_normal(prompt),
            "prompt_raw_norm": replace_for_normal(prompt_raw),
        }

        return prompt_dict

    def tokener(self, prompt, mode='padding'):

        if mode == 'padding':
            return self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
        else:
            return self.tokenizer(
                prompt,
                return_tensors="pt",
            ).input_ids[0][1:-1]

    def __getitem__(self, index):

        example = {}
        example_len = len(self.instance_images)

        # instance_masks, instance_images, placeholder_tokens are all lists
        num_of_tokens = random.randrange(1, len(self.placeholder_tokens[index % example_len]) + 1)
        sample_prop = np.ones(len(self.class_tokens[index % example_len]))

        for attn_cls in ['face']:
            if attn_cls in self.class_tokens[index % example_len]:
                sample_prop[self.class_tokens[index % example_len].index(attn_cls)] = 3.0

        sample_prop /= sample_prop.sum()

        tokens_ids_to_use = np.random.choice(
            range(len(self.placeholder_tokens[index % example_len])),
            size=num_of_tokens,
            p=sample_prop,
            replace=False,
        )

        tokens_to_use = [
            self.placeholder_tokens[index % example_len][tkn_i] for tkn_i in tokens_ids_to_use
        ]
        classes_to_use = [
            self.class_tokens[index % example_len][tkn_i] for tkn_i in tokens_ids_to_use
        ]

        descs_to_use = [
            self.instance_descriptions[index % example_len][tkn_i] for tkn_i in tokens_ids_to_use
        ]

        views_to_use = self.instance_views[index % example_len]

        prompt_dict = self.construct_prompt(
            classes_to_use, tokens_to_use, descs_to_use, views_to_use
        )

        example["instance_images"] = self.instance_images[index % example_len]
        example["instance_masks"] = self.instance_masks[index % example_len][tokens_ids_to_use]

        # dilation a bit to fill holes
        example["instance_masks"] = dilation(
            example["instance_masks"],
            kernel=torch.ones(5, 5),
            border_value=1.0,
        )

        if 'face' in classes_to_use:
            face_index = classes_to_use.index('face')
            face_mask = example["instance_masks"][face_index]

            # large dilation the face mask
            bbox = bbox2(face_mask[0].numpy())
            k_size = max(bbox[1] - bbox[0], bbox[3] - bbox[2]) // 4
            example["instance_masks"][face_index] = dilation(
                face_mask.unsqueeze(0),
                kernel=torch.ones(k_size, k_size),
                border_value=1.0,
            ).squeeze(0)

        example["token_ids"] = torch.tensor([
            self.placeholder_full.index(token) for token in tokens_to_use
        ])

        if random.random() > self.flip_p:
            example["instance_images"] = TF.hflip(example["instance_images"])
            example["instance_masks"] = TF.hflip(example["instance_masks"])

        example["instance_prompt_ids"] = self.tokener(prompt_dict["prompt"])
        example["instance_prompt_ids_raw"] = self.tokener(prompt_dict["prompt_raw"])
        example["class_ids"] = self.tokener(" ".join(classes_to_use), "non_padding")

        if self.class_data_root and self.with_prior_preservation:

            cls_img_path = self.class_images_path[self.gender][index % self.num_class_images]
            syn_img_path = self.syn_images_path[self.gender][index % self.num_syn_images]
            syn_norm_path = self.syn_normals_path[self.gender][index % self.num_syn_images]
            syn_json_path = self.syn_json_path[self.gender][index % self.num_syn_images]

            # prompt for synthetic data
            with open(syn_json_path, 'r') as f:
                gpt4v_syn = json.load(f)

            syn_classes = list(gpt4v_syn.keys())
            syn_classes.remove("gender")
            syn_descs = [gpt4v_syn[cls] for cls in syn_classes]
            syn_tokens = ["" for _ in syn_classes]

            is_face = True if 'head' in str(syn_json_path) else False

            example["syn_prompt_ids_raw"] = self.tokener(
                self.construct_prompt(syn_classes, syn_tokens, syn_descs, [], is_face)["prompt_raw"]
            )
            example["syn_prompt_ids_raw_norm"] = self.tokener(
                self.construct_prompt(syn_classes, syn_tokens, syn_descs, [],
                                      is_face)["prompt_raw_norm"]
            )

            # for full human
            person_image = Image.open(cls_img_path)
            syn_image = Image.open(syn_img_path)
            syn_normal = Image.open(syn_norm_path)
            syn_mask = syn_normal.split()[-1]

            if not person_image.mode == "RGB":
                person_image = person_image.convert("RGB")
            if not syn_image.mode == "RGB":
                syn_image = syn_image.convert("RGB")
            if not syn_normal.mode == "RGB":
                syn_normal = syn_normal.convert("RGB")

            example["person_images"] = self.image_transforms(person_image)
            example["syn_images"] = self.image_transforms(syn_image)
            example["syn_normals"] = self.image_transforms(syn_normal)
            example["syn_mask"] = self.mask_transforms(syn_mask)
            example["person_filename"] = self.tokener(os.path.basename(str(cls_img_path)))

        return example


class MultiDreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
        self,
        placeholder_tokens,
        so_map_lst,
        tokenizer,
        initializer_tokens,
        gpt4v_response,
        use_shape_desc,
        with_prior_preservation,
        num_class_images,
        gender,
        class_data_root=None,
        size=512,
        center_crop=False,
        flip_p=0.5,
        sd_version="stable-diffusion-2-1",
        use_view_prompt=True,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.flip_p = flip_p
        self.sd_version = sd_version
        self.gpt4v_response = gpt4v_response
        self.use_shape_desc = use_shape_desc
        self.with_prior_preservation = with_prior_preservation
        self.use_view_prompt = use_view_prompt
        self.so_map_lst = so_map_lst

        self.cam_view_dict = {
            "01": ["side", "front"],
            "02": ["side", "front"],
            "03": ["side", "front"],
            "04": ["overhead", "side"],
            "05": ["front"],
            "06": ["front"],
            "07": ["front"],
            "08": ["overhead", "front"],
            "09": ["side", "front"],
            "10": ["side", "front"],
            "11": ["side", "front"],
            "12": ["overhead", "side"],
            "13": ["side", "back"],
            "14": ["side", "back"],
            "15": ["side", "back"],
            "16": ["overhead", "back"],
            "17": ["back"],
            "18": ["overhead", "back"],
            "19": ["side", "back"],
            "20": ["side", "back"],
            "21": ["side", "back"],
            "22": ["overhead", "side"],
        }

        self.image_transforms = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
        ])

        self.instance_data_roots = []
        instance_img_paths = {}

        for subject_outfit in self.gpt4v_response.keys():
            subject, outfit = subject_outfit.split("_")
            if subject == 'human':
                data_dir = Path(f"./data/{subject}/{outfit}")
            else:
                data_dir = Path(f"./data/PuzzleIOI/puzzle_capture/{subject}/{outfit}")
            self.instance_data_roots.append(data_dir)
            if not data_dir.exists():
                raise ValueError(f"Instance {data_dir} images root doesn't exists.")
            else:
                instance_img_paths[subject_outfit] = sorted(
                    glob.glob(f"{data_dir}/image/*[!raw].jpg")
                )

        self.placeholder_tokens = []
        self.class_tokens = []
        self.gender_tokens = []

        self.class_data_root = class_data_root
        self.num_class_images = num_class_images

        self.instance_images = []
        self.instance_masks = []
        self.instance_descriptions = []
        self.instance_views = []

        self.placeholder_full = placeholder_tokens

        self.class_images_path = {}
        self.class_normals_path = {}
        self.syn_images_path = {}
        self.syn_normals_path = {}
        self.syn_json_path = {}

        self.cur_initializer = []
        self.cur_placeholder = []
        self.cur_gender = "man"
        self.num_syn_images = {"man": 0, "woman": 0}

        for instance_data_root in self.instance_data_roots:

            subject, outfit = str(instance_data_root).split("/")[-2:]
            so_name = f"{subject}_{outfit}"

            self.cur_placeholder = [
                token for token_idx, token in enumerate(placeholder_tokens)
                if self.so_map_lst[token_idx] == so_name
            ]
            self.cur_initializer = [
                cls for cls_idx, cls in enumerate(initializer_tokens)
                if self.so_map_lst[cls_idx] == so_name
            ]

            self.cur_gender = [
                sex for sex_idx, sex in enumerate(gender) if self.so_map_lst[sex_idx] == so_name
            ][0]

            for instance_img_path in instance_img_paths[so_name]:
                instance_filename = instance_img_path.split("/")[-1].split(".")[0]
                instance_mask_paths = glob.glob(
                    f"{instance_data_root}/mask/{instance_filename}_*.png"
                )

                if len(instance_mask_paths) > 0:

                    if "PuzzleIOI" in instance_img_path and self.use_view_prompt:
                        cam_id = instance_img_path.split("/")[-1].split("_")[1]
                        self.instance_views.append(self.cam_view_dict[cam_id])
                    else:
                        self.instance_views.append([])

                    self.instance_images.append(
                        self.image_transforms(Image.open(instance_img_path))[:3]
                    )

                    instance_mask = []
                    instance_placeholder_token = []
                    instance_class_tokens = []
                    instance_description = []

                    for instance_mask_path in instance_mask_paths:
                        curr_mask = Image.open(instance_mask_path)
                        curr_mask = self.mask_transforms(curr_mask)[0, None, None, ...]
                        instance_mask.append(curr_mask)
                        curr_cls = instance_mask_path.split(".")[0].split("_")[-1]

                        instance_placeholder_token.append(
                            self.cur_placeholder[self.cur_initializer.index(curr_cls)]
                        )
                        instance_class_tokens.append(curr_cls)
                        instance_description.append(self.gpt4v_response[so_name][curr_cls])

                    self.instance_masks.append(torch.cat(instance_mask))
                    self.placeholder_tokens.append(instance_placeholder_token)
                    self.class_tokens.append(instance_class_tokens)
                    self.instance_descriptions.append(instance_description)
                    self.gender_tokens.append(self.cur_gender)

            if (self.class_data_root is not None) and self.with_prior_preservation and (
                self.cur_gender not in self.class_images_path.keys()
            ):
                class_rgb_dir = Path(self.class_data_root) / self.sd_version / self.cur_gender

                self.class_images_path[self.cur_gender] = [
                    item
                    for item in sorted(list(class_rgb_dir.iterdir())) if str(item).endswith(".jpg")
                ]

                syn_dataset = "thuman2_orbit"
                syn_rgb_dir = Path(self.class_data_root) / syn_dataset / self.cur_gender
                syn_normal_dir = Path(
                    self.class_data_root
                ) / syn_dataset / f"{self.cur_gender}_norm"
                syn_json_dir = Path(self.class_data_root) / syn_dataset / f"{self.cur_gender}_desc"

                self.syn_images_path[self.cur_gender] = [
                    item
                    for item in sorted(list(syn_rgb_dir.iterdir())) if str(item).endswith(".png")
                ]
                self.syn_normals_path[self.cur_gender] = [
                    item for item in sorted(list(syn_normal_dir.iterdir()))
                    if str(item).endswith(".png")
                ]
                self.syn_json_path[self.cur_gender] = [
                    item
                    for item in sorted(list(syn_json_dir.iterdir())) if str(item).endswith(".json")
                ]

            self.num_syn_images[self.cur_gender] = len(self.syn_images_path[self.cur_gender])

        self._length = max(
            len(self.instance_images), self.num_class_images, self.num_syn_images["man"],
            self.num_syn_images["woman"]
        )

    def __len__(self):

        return self._length

    def construct_prompt(
        self,
        classes_to_use,
        tokens_to_use,
        descs_to_use,
        views_to_use,
        gender_to_use,
        is_face=False
    ):

        replace_for_normal = lambda x: x.replace(
            "a high-resolution DSLR colored image of ", "a detailed sculpture of "
        )

        # formulate the prompt and prompt_raw
        if is_face:
            prompt_head = f"a high-resolution DSLR colored image of the headshot of a {gender_to_use}"
        else:
            prompt_head = f"a high-resolution DSLR colored image of a {gender_to_use}"

        facial_classes = ['face', 'haircut', 'hair']
        with_classes = [cls for cls in classes_to_use if cls in facial_classes]
        wear_classes = [cls for cls in classes_to_use if cls not in facial_classes]

        prompt_raw = prompt = f"{prompt_head}, "

        if len(views_to_use) > 0:
            view_prompt = ", ".join([f"{view} view" for view in views_to_use])
        else:
            view_prompt = ""

        for class_token in with_classes:
            idx = classes_to_use.index(class_token)

            if len(wear_classes) == 0 and with_classes.index(class_token) == len(with_classes) - 1:
                ending = f", {view_prompt}."
            else:
                ending = ", "

            if self.use_shape_desc:
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
                    ending = f", {view_prompt}."

                if self.use_shape_desc:
                    prompt += f"{tokens_to_use[idx]} {descs_to_use[idx]} {class_token}{ending}"
                    prompt_raw += f"{descs_to_use[idx]} {class_token}{ending}"
                else:
                    prompt += f"{tokens_to_use[idx]} {class_token}{ending}"
                    prompt_raw += f"{class_token}{ending}"

        prompt_dict = {
            "prompt": prompt,
            "prompt_raw": prompt_raw,
            "prompt_norm": replace_for_normal(prompt),
            "prompt_raw_norm": replace_for_normal(prompt_raw),
        }

        return prompt_dict

    def tokener(self, prompt, mode='padding'):

        if mode == 'padding':
            return self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
        else:
            return self.tokenizer(
                prompt,
                return_tensors="pt",
            ).input_ids[0][1:-1]

    def __getitem__(self, index):

        example = {}
        example_len = len(self.instance_images)

        # instance_masks, instance_images, placeholder_tokens are all lists
        num_of_tokens = random.randrange(1, len(self.placeholder_tokens[index % example_len]) + 1)
        sample_prop = np.ones(len(self.class_tokens[index % example_len]))

        for attn_cls in ['face']:
            if attn_cls in self.class_tokens[index % example_len]:
                sample_prop[self.class_tokens[index % example_len].index(attn_cls)] = 3.0

        sample_prop /= sample_prop.sum()

        tokens_ids_to_use = np.random.choice(
            range(len(self.placeholder_tokens[index % example_len])),
            size=num_of_tokens,
            p=sample_prop,
            replace=False,
        )

        tokens_to_use = [
            self.placeholder_tokens[index % example_len][tkn_i] for tkn_i in tokens_ids_to_use
        ]
        classes_to_use = [
            self.class_tokens[index % example_len][tkn_i] for tkn_i in tokens_ids_to_use
        ]

        descs_to_use = [
            self.instance_descriptions[index % example_len][tkn_i] for tkn_i in tokens_ids_to_use
        ]

        gender_to_use = self.gender_tokens[index % example_len]

        views_to_use = self.instance_views[index % example_len]

        prompt_dict = self.construct_prompt(
            classes_to_use,
            tokens_to_use,
            descs_to_use,
            views_to_use,
            gender_to_use,
        )

        example["instance_images"] = self.instance_images[index % example_len]
        example["instance_masks"] = self.instance_masks[index % example_len][tokens_ids_to_use]

        # dilation a bit to fill holes
        example["instance_masks"] = dilation(
            example["instance_masks"],
            kernel=torch.ones(5, 5),
            border_value=1.0,
        )

        if 'face' in classes_to_use:
            face_index = classes_to_use.index('face')
            face_mask = example["instance_masks"][face_index]

            # large dilation the face mask
            bbox = bbox2(face_mask[0].numpy())
            k_size = max(bbox[1] - bbox[0], bbox[3] - bbox[2]) // 4
            example["instance_masks"][face_index] = dilation(
                face_mask.unsqueeze(0),
                kernel=torch.ones(k_size, k_size),
                border_value=1.0,
            ).squeeze(0)

        example["token_ids"] = torch.tensor([
            self.placeholder_full.index(token) for token in tokens_to_use
        ])

        if random.random() > self.flip_p:
            example["instance_images"] = TF.hflip(example["instance_images"])
            example["instance_masks"] = TF.hflip(example["instance_masks"])

        example["instance_prompt_ids"] = self.tokener(prompt_dict["prompt"])
        example["instance_prompt_ids_raw"] = self.tokener(prompt_dict["prompt_raw"])
        example["class_ids"] = self.tokener(" ".join(classes_to_use), "non_padding")

        if self.class_data_root and self.with_prior_preservation:

            cls_img_path = self.class_images_path[gender_to_use][index % self.num_class_images]
            syn_img_path = self.syn_images_path[gender_to_use][index %
                                                               self.num_syn_images[gender_to_use]]
            syn_norm_path = self.syn_normals_path[gender_to_use][index %
                                                                 self.num_syn_images[gender_to_use]]
            syn_json_path = self.syn_json_path[gender_to_use][index %
                                                              self.num_syn_images[gender_to_use]]

            # prompt for synthetic data
            with open(syn_json_path, 'r') as f:
                gpt4v_syn = json.load(f)

            syn_classes = list(gpt4v_syn.keys())
            syn_classes.remove("gender")
            syn_descs = [gpt4v_syn[cls] for cls in syn_classes]
            syn_tokens = ["" for _ in syn_classes]

            is_face = True if 'head' in str(syn_json_path) else False

            example["syn_prompt_ids_raw"] = self.tokener(
                self.construct_prompt(
                    syn_classes, syn_tokens, syn_descs, [], gender_to_use, is_face
                )["prompt_raw"]
            )
            example["syn_prompt_ids_raw_norm"] = self.tokener(
                self.construct_prompt(
                    syn_classes, syn_tokens, syn_descs, [], gender_to_use, is_face
                )["prompt_raw_norm"]
            )

            # for full human
            person_image = Image.open(cls_img_path)
            syn_image = Image.open(syn_img_path)
            syn_normal = Image.open(syn_norm_path)
            syn_mask = syn_normal.split()[-1]

            if not person_image.mode == "RGB":
                person_image = person_image.convert("RGB")
            if not syn_image.mode == "RGB":
                syn_image = syn_image.convert("RGB")
            if not syn_normal.mode == "RGB":
                syn_normal = syn_normal.convert("RGB")

            example["person_images"] = self.image_transforms(person_image)
            example["syn_images"] = self.image_transforms(syn_image)
            example["syn_normals"] = self.image_transforms(syn_normal)
            example["syn_mask"] = self.mask_transforms(syn_mask)
            example["person_filename"] = self.tokener(os.path.basename(str(cls_img_path)))

        return example


def bbox2(img):

    # from https://stackoverflow.com/a/31402351/19249364
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    r_height = rmax - rmin
    c_width = cmax - cmin
    rmin, rmax = max(0, rmin - r_height // 4), min(img.shape[0] - 1, rmax + r_height // 4)
    cmin, cmax = max(0, cmin - c_width // 4), min(img.shape[1] - 1, cmax + c_width // 4)
    return rmin, rmax, cmin, cmax


def padded_stack(
    tensors: List[torch.Tensor],
    mode: str = "constant",
    value: Union[int, float] = 0,
    dim: int = 0,
    full_size: int = 10,
) -> torch.Tensor:
    """
    Stack tensors along a dimension and pad them along last dimension to ensure their size is equal.

    Args:
        tensors (List[torch.Tensor]): list of tensors to stack
        mode (str): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        value (Union[int, float]): value to use for constant padding
        dim (int): dimension along which to stack
        full_size (int): size to pad to

    Returns:
        torch.Tensor: stacked tensor
    """
    def make_padding(pad):
        padding = [0] * (tensors[0].dim() * 2)
        padding[tensors[0].dim() * 2 - (dim * 2 + 1)] = pad
        return padding

    out = torch.stack(
        [
            F.pad(x, make_padding(full_size - x.size(dim)), mode=mode, value=value) if full_size -
            x.size(dim) > 0 else x for x in tensors
        ],
        dim=0,
    )
    return out


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    raw_input_ids = [example["instance_prompt_ids_raw"] for example in examples]

    syn_norm_ids = [example["syn_prompt_ids_raw_norm"] for example in examples]
    syn_rgb_ids = [example["syn_prompt_ids_raw"] for example in examples]

    pixel_values = [example["instance_images"] for example in examples]

    masks = [example["instance_masks"] for example in examples]
    token_ids = [example["token_ids"] for example in examples]
    class_ids = [example["class_ids"] for example in examples]

    batch = {}

    if with_prior_preservation:
        person_pixel_values = [example["person_images"] for example in examples]
        syn_pixel_values = [example["syn_images"] for example in examples]
        syn_norm_values = [example["syn_normals"] for example in examples]
        syn_mask_values = [example["syn_mask"] for example in examples]

        input_ids = raw_input_ids + syn_rgb_ids + syn_norm_ids + input_ids
        pixel_values = person_pixel_values + syn_pixel_values + syn_norm_values + pixel_values

        batch["person_filenames"] = torch.cat([example["person_filename"] for example in examples],
                                              dim=0)
        batch["syn_mask"] = torch.stack(syn_mask_values)

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    masks = padded_stack(masks)
    token_ids = padded_stack(token_ids)
    class_ids = padded_stack(class_ids)

    batch.update({
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "instance_masks": masks,
        "token_ids": token_ids,
        "class_ids": class_ids,
    })

    return batch


def construct_prior_prompt(gender, all_classes, gpt4v_dict, use_shape_desc):

    if all_classes == "":
        all_classes = list(gpt4v_dict.keys())
        all_classes.remove("gender")
        for key in ["eyeglasses", "sunglasses", "glasses"]:
            if key in all_classes:
                all_classes.remove(key)

    facial_classes = ['face', 'haircut', 'hair']

    prompt_head = f"a high-resolution DSLR colored image of a {gender}"
    with_classes = [cls for cls in all_classes if cls in facial_classes]
    wear_classes = [cls for cls in all_classes if cls not in facial_classes]

    class_prompt = f"{prompt_head}, "

    for class_token in with_classes:
        if use_shape_desc:
            class_prompt += f"{gpt4v_dict[class_token]} {class_token}, "
        else:
            class_prompt += f"normal {class_token}, "

    if len(wear_classes) > 0:
        class_prompt += "wearing "

        for class_token in wear_classes:

            if wear_classes.index(class_token) < len(wear_classes) - 2:
                ending = ", "
            elif wear_classes.index(class_token) == len(wear_classes) - 2:
                ending = ", and "
            else:
                ending = "."
            if use_shape_desc:
                class_prompt += f"{gpt4v_dict[class_token]} {class_token}{ending}"
            else:
                class_prompt += f"daily {class_token}{ending}"

    return class_prompt


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."
    def __init__(self, tokenizer, prompt, num_samples):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        example["prompt_ids"] = self.tokenizer(
            self.prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return example
