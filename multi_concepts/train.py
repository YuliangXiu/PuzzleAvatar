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

import hashlib
import itertools
import logging
import math
import os
import sys
from pathlib import Path
from typing import List

import datasets
import diffusers
import numpy as np
import ptp_utils
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    PNDMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from ptp_utils import AttentionStore, P2PCrossAttnProcessor
from tqdm.auto import tqdm
from train_utils import (
    import_model_class_from_model_name_or_path,
    parse_args,
    clean_decode_prompt,
)
from dataset import (
    DreamBoothDataset,
    PromptDataset,
    collate_fn,
    construct_prior_prompt,
)

import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), "../thirdparties/peft/src"))

from peft import BOFTConfig, LoraConfig, get_peft_model

check_min_version("0.12.0")

logger = get_logger(__name__)

logging.getLogger("wandb").setLevel(logging.ERROR)

UNET_TARGET_MODULES = [
    "to_q", "to_v", "to_k", "proj_in", "proj_out", "to_out.0", "add_k_proj", "add_v_proj",
    "ff.net.2"
]

TEXT_ENCODER_TARGET_MODULES = [
    "embed_tokens", "q_proj", "k_proj", "v_proj", "out_proj", "mlp.fc1", "mlp.fc2"
]

negative_prompt = 'unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy, shadow, worst quality, jpeg artifacts, poorly drawn, dehydrated, noisy, poorly drawn, bad proportions, bad anatomy, bad lighting, bad composition, bad framing, fused fingers, noisy, duplicate characters'


class SpatialDreambooth:
    def __init__(self):
        self.args, self.gpt4v_response = parse_args()
        self.attn_mask_cache = {}
        self.class_ids = []
        self.save_attn_mask_cache = False
        if 'base' in self.args.pretrained_model_name_or_path:
            self.attn_res = 16
            self.mask_res = 64
            self.args.resolution = 512
        else:
            self.attn_res = 24
            self.mask_res = 96
            self.args.resolution = 768

        self.main()

    def main(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_dir=logging_dir,
        )

        if self.args.report_to == "wandb":

            wandb_init = {
                "wandb": {
                    "name": self.args.project_name,
                    "mode": self.args.wandb_mode,
                }
            }

        if (
            self.args.train_text_encoder and self.args.gradient_accumulation_steps > 1 and
            self.accelerator.num_processes > 1
        ):
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if self.args.seed is not None:
            set_seed(self.args.seed)

        # Handle the repository creation
        if self.accelerator.is_main_process:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(
            self.args.pretrained_model_name_or_path, self.args.revision
        )

        # Load scheduler and models
        self.noise_scheduler = PNDMScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.text_encoder = text_encoder_cls.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.args.revision,
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.args.revision,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.args.revision,
        )

        # Load the tokenizer
        if self.args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.tokenizer_name, revision=self.args.revision, use_fast=False
            )
        elif self.args.pretrained_model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=self.args.revision,
                use_fast=False,
            )

        # Add assets tokens to tokenizer
        self.placeholder_tokens = [
            self.args.placeholder_token.replace(">", f"{idx}>")
            for idx in range(self.args.num_of_assets)
        ]
        num_added_tokens = self.tokenizer.add_tokens(self.placeholder_tokens)
        assert num_added_tokens == self.args.num_of_assets
        self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(self.placeholder_tokens)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        self.class_ids = self.tokenizer(" ".join(self.args.initializer_tokens), ).input_ids[1:-1]

        if len(self.args.initializer_tokens) > 0:
            # Use initializer tokens
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            for tkn_idx, initializer_token in enumerate(self.args.initializer_tokens):
                curr_token_ids = self.tokenizer.encode(initializer_token, add_special_tokens=False)
                token_embeds[self.placeholder_token_ids[tkn_idx]] = token_embeds[curr_token_ids[0]]
        else:
            # Initialize new tokens randomly
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            token_embeds[-self.args.num_of_assets:] = token_embeds[-3 * self.args.num_of_assets:-2 *
                                                                   self.args.num_of_assets]

        self.validation_scheduler = PNDMScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.validation_scheduler.set_timesteps(50)

        # We start by only optimizing the embeddings
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        # Freeze all parameters except for the token embeddings in text encoder
        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        (
            self.unet,
            self.text_encoder,
            self.tokenizer,
        ) = self.accelerator.prepare(self.unet, self.text_encoder, self.tokenizer)

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if self.args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.args.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate * self.args.gradient_accumulation_steps *
                self.args.train_batch_size * self.accelerator.num_processes
            )

        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # We start by only optimizing the embeddings
        params_to_optimize = self.text_encoder.get_input_embeddings().parameters()
        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.args.initial_learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        # Create attention controller
        self.controller = AttentionStore()
        self.register_attention_control(self.controller)

        pretrained_name = self.args.pretrained_model_name_or_path.split("/")[-1]
        cached_masks_path = Path(
            self.args.class_data_dir, f"{pretrained_name}/{self.args.gender}/attn_masks_cache.pt"
        )

        # Generate class images if prior preservation is enabled.
        with torch.no_grad():
            if self.args.with_prior_preservation:

                if cached_masks_path.exists():
                    self.attn_mask_cache = torch.load(cached_masks_path)
                else:

                    # generate attn_mask_cache file
                    pipeline = DiffusionPipeline.from_pretrained(
                        self.args.pretrained_model_name_or_path,
                        torch_dtype=self.weight_dtype,
                        safety_checker=None,
                        revision=self.args.revision,
                    )
                    self.unet.eval()
                    pipeline.unet = self.unet
                    # pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.4, b2=1.6)
                    pipeline.set_progress_bar_config(disable=True)
                    pipeline.to(self.accelerator.device)

                    for class_name in [self.args.gender]:

                        class_prompt = construct_prior_prompt(
                            self.args.gender,
                            self.args.initializer_tokens,
                            self.gpt4v_response,
                            self.args.use_shape_description,
                        )

                        class_images_dir = Path(
                            self.args.class_data_dir
                        ) / pretrained_name / class_name

                        if not class_images_dir.exists():
                            class_images_dir.mkdir(parents=True)
                        cur_class_images = len(list(class_images_dir.iterdir()))

                        if cur_class_images < self.args.num_class_images:

                            num_new_images = self.args.num_class_images - cur_class_images
                            logger.info(f"Number of class images to sample: {num_new_images}.")

                            sample_dataset = PromptDataset(
                                self.tokenizer, class_prompt, num_new_images
                            )
                            sample_dataloader = torch.utils.data.DataLoader(
                                sample_dataset, batch_size=self.args.sample_batch_size
                            )

                            sample_dataloader = self.accelerator.prepare(sample_dataloader)

                            for example in tqdm(
                                sample_dataloader,
                                desc=f"Generating {class_name} images",
                                disable=not self.accelerator.is_local_main_process,
                            ):

                                images = pipeline(
                                    example["prompt"],
                                    negative_prompt=[negative_prompt] * len(example["prompt"])
                                ).images

                                for batch_idx in range(self.args.sample_batch_size):

                                    hash_image = hashlib.sha1(images[batch_idx].tobytes()
                                                             ).hexdigest()
                                    base_name = f"{example['index'][batch_idx] + cur_class_images}-{hash_image}.jpg"
                                    image_filename = (class_images_dir / base_name)
                                    images[batch_idx].save(image_filename)
                                    self.attn_mask_cache[base_name] = {}

                                    agg_attn_pre = self.aggregate_attention(
                                        res=self.attn_res,
                                        from_where=("up", "down"),
                                        is_cross=True,
                                        select=batch_idx,
                                        batch_size=self.args.sample_batch_size,
                                    )

                                    for class_token_id in self.class_ids:

                                        # agg_attn [24,24,77]
                                        class_idx = ((
                                            example["prompt_ids"][batch_idx] == class_token_id
                                        ).nonzero().item())

                                        cls_attn_mask = agg_attn_pre[..., class_idx]
                                        cls_attn_mask = (cls_attn_mask / cls_attn_mask.max())
                                        self.attn_mask_cache[base_name][class_token_id
                                                                       ] = cls_attn_mask

                                self.controller.attention_store = {}
                                self.controller.cur_step = 0

                    del pipeline

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    torch.save(self.attn_mask_cache, cached_masks_path)

        # Dataset and DataLoaders creation:
        train_dataset = DreamBoothDataset(
            instance_data_root=self.args.instance_data_dir,
            placeholder_tokens=self.placeholder_tokens,
            initializer_tokens=self.args.initializer_tokens,
            gpt4v_response=self.gpt4v_response,
            use_shape_desc=self.args.use_shape_description,
            with_prior_preservation=self.args.with_prior_preservation,
            num_class_images=self.args.num_class_images,
            gender=self.args.gender,
            class_data_root=self.args.class_data_dir if self.args.with_prior_preservation else None,
            tokenizer=self.tokenizer,
            size=self.args.resolution,
            center_crop=self.args.center_crop,
            sd_version=self.args.pretrained_model_name_or_path.split("/")[-1],
            use_view_prompt=self.args.use_view_prompt,
            use_full_shot=self.args.use_full_shot,
            use_half_data=self.args.use_half_data,
        )

        prompt_dict = train_dataset.construct_prompt(
            self.args.initializer_tokens, self.placeholder_tokens,
            [self.gpt4v_response[cls_name] for cls_name in self.args.initializer_tokens], []
        )

        self.args.instance_prompt = prompt_dict["prompt"]
        self.args.instance_prompt_raw = prompt_dict["prompt_raw"]

        logger.info(f"Placeholder Tokens: {self.placeholder_tokens}")
        logger.info(f"Initializer Tokens: {self.args.initializer_tokens}")

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, self.args.with_prior_preservation),
            num_workers=self.args.dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if self.args.max_train_steps is None:
            self.args.max_train_steps = (self.args.num_train_epochs * num_update_steps_per_epoch)
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        (
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = self.accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if self.accelerator.unwrap_model(self.unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {self.accelerator.unwrap_model(self.unet).dtype}. {low_precision_error_string}"
            )

        if (
            self.args.train_text_encoder and
            self.accelerator.unwrap_model(self.text_encoder).dtype != torch.float32
        ):
            raise ValueError(
                f"Text encoder loaded as datatype {self.accelerator.unwrap_model(self.text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            self.args.max_train_steps = (self.args.num_train_epochs * num_update_steps_per_epoch)
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch
        )

        if len(self.args.initializer_tokens) > 0:
            # Only for logging
            self.args.initializer_tokens = ", ".join(self.args.initializer_tokens)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                "multi-concepts", config=vars(self.args), init_kwargs=wandb_init
            )

        # Train
        total_batch_size = (
            self.args.train_batch_size * self.accelerator.num_processes *
            self.args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(self.args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.args.output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * self.args.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (
                    num_update_steps_per_epoch * self.args.gradient_accumulation_steps
                )

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(global_step, self.args.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        # keep original embeddings as reference
        orig_embeds_params = (
            self.accelerator.unwrap_model(self.text_encoder
                                         ).get_input_embeddings().weight.data.clone()
        )

        for epoch in range(first_epoch, self.args.num_train_epochs):

            self.unet.train()
            if self.args.train_text_encoder:
                self.text_encoder.train()

            for step, batch in enumerate(train_dataloader):

                if self.args.phase1_train_steps == global_step:

                    if self.args.use_peft == "none":
                        self.unet.requires_grad_(True)

                    else:
                        if self.args.use_peft == "boft":
                            unet_config = BOFTConfig(
                                boft_block_size=self.args.boft_block_size,
                                boft_block_num=self.args.boft_block_num,
                                boft_n_butterfly_factor=self.args.boft_n_butterfly_factor,
                                target_modules=UNET_TARGET_MODULES,
                                boft_dropout=self.args.boft_dropout,
                                bias=self.args.boft_bias
                            )
                        elif self.args.use_peft == "lora":

                            unet_config = LoraConfig(
                                r=self.args.lora_r,
                                use_rslora=True,
                                lora_alpha=self.args.lora_r,
                                init_lora_weights="gaussian",
                                target_modules=UNET_TARGET_MODULES,
                                lora_dropout=self.args.boft_dropout,
                                bias="lora_only"
                            )

                        self.unet = get_peft_model(self.unet, unet_config)
                        self.unet.to(self.accelerator.device)
                        self.unet.print_trainable_parameters()

                        logger.info("***** Training with BOFT fine-tuning (unet) *****")
                        # logger.info(f"Structure of UNet be like: {self.unet} \n")

                    if self.args.train_text_encoder:
                        if self.args.use_peft == "none":
                            self.text_encoder.requires_grad_(True)
                        else:
                            if self.args.use_peft == "boft":
                                text_config = BOFTConfig(
                                    boft_block_size=self.args.boft_block_size,
                                    boft_block_num=self.args.boft_block_num,
                                    boft_n_butterfly_factor=self.args.boft_n_butterfly_factor,
                                    target_modules=TEXT_ENCODER_TARGET_MODULES,
                                    boft_dropout=self.args.boft_dropout,
                                    bias=self.args.boft_bias
                                )
                            elif self.args.use_peft == "lora":
                                text_config = LoraConfig(
                                    r=self.args.lora_r,
                                    use_rslora=True,
                                    lora_alpha=self.args.lora_r,
                                    init_lora_weights="gaussian",
                                    target_modules=TEXT_ENCODER_TARGET_MODULES,
                                    lora_dropout=self.args.boft_dropout,
                                    bias="lora_only"
                                )

                            self.text_encoder = get_peft_model(self.text_encoder, text_config)
                            self.text_encoder.to(self.accelerator.device)
                            self.text_encoder.print_trainable_parameters()

                            logger.info("***** Training with BOFT fine-tuning (text_encoder) *****")
                            # logger.info(
                            #     f"Structure of Text Encoder be like: {self.text_encoder} \n"
                            # )

                    unet_params = [param for param in self.unet.parameters() if param.requires_grad]
                    text_params = [
                        param for param in self.text_encoder.parameters() if param.requires_grad
                    ]

                    params_to_optimize = (
                        itertools.chain(unet_params, text_params)
                        if self.args.train_text_encoder else itertools.chain(
                            unet_params,
                            self.text_encoder.get_input_embeddings().parameters(),
                        )
                    )

                    del optimizer
                    optimizer = optimizer_class(
                        params_to_optimize,
                        lr=self.args.learning_rate,
                        betas=(self.args.adam_beta1, self.args.adam_beta2),
                        weight_decay=self.args.adam_weight_decay,
                        eps=self.args.adam_epsilon,
                    )
                    del lr_scheduler
                    lr_scheduler = get_scheduler(
                        self.args.lr_scheduler,
                        optimizer=optimizer,
                        step_rules=self.args.lr_step_rules,
                        num_warmup_steps=self.args.lr_warmup_steps *
                        self.args.gradient_accumulation_steps,
                        num_training_steps=self.args.max_train_steps *
                        self.args.gradient_accumulation_steps,
                        num_cycles=self.args.lr_num_cycles,
                        power=self.args.lr_power,
                    )
                    optimizer, lr_scheduler = self.accelerator.prepare(optimizer, lr_scheduler)

                logs = {}

                # Skip steps until we reach the resumed step
                if (
                    self.args.resume_from_checkpoint and epoch == first_epoch and step < resume_step
                ):
                    if step % self.args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with self.accelerator.accumulate(self.unet):
                    # Convert images to latent space
                    latents = self.vae.encode(batch["pixel_values"].to(dtype=self.weight_dtype)
                                             ).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (bsz, ),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                        )

                    loss = 0.

                    if self.args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts
                        # compute the loss on each part separately.

                        prior_pred, syn_pred, syn_norm_pred, model_pred = torch.chunk(
                            model_pred, bsz, dim=0
                        )
                        prior_target, syn_target, syn_norm_target, target = torch.chunk(
                            target, bsz, dim=0
                        )

                        downsampled_syn_mask = F.interpolate(
                            input=batch["syn_mask"], size=(self.mask_res, self.mask_res)
                        )

                    if self.args.apply_masked_loss:
                        max_masks = torch.max(batch["instance_masks"], dim=1).values

                        # [1,1,96,96]
                        downsampled_mask = F.interpolate(
                            input=max_masks, size=(self.mask_res, self.mask_res)
                        )

                        # partial masked
                        model_pred = model_pred * downsampled_mask
                        target = target * downsampled_mask

                    # Compute instance loss
                    loss_inst = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    loss += loss_inst
                    logs["l_inst"] = loss_inst.detach().item()

                    # Attention loss
                    if self.args.lambda_attention != 0:
                        attn_loss = 0
                        cls_masks = []
                        cls_masks_gt = []

                        for batch_idx in range(self.args.train_batch_size):

                            GT_masks = F.interpolate(
                                input=batch["instance_masks"][batch_idx],
                                size=(self.attn_res, self.attn_res)
                            )

                            if self.args.with_prior_preservation:
                                curr_cond_batch_idx = self.args.train_batch_size * (
                                    bsz - 1
                                ) + batch_idx

                                agg_attn_prior = self.aggregate_attention(
                                    res=self.attn_res,
                                    from_where=("up", "down"),
                                    is_cross=True,
                                    select=batch_idx,
                                    batch_size=self.args.train_batch_size * bsz,
                                    use_half=False,
                                )

                            else:
                                curr_cond_batch_idx = batch_idx

                            agg_attn = self.aggregate_attention(
                                res=self.attn_res,
                                from_where=("up", "down"),
                                is_cross=True,
                                select=curr_cond_batch_idx,
                                batch_size=self.args.train_batch_size * bsz,
                                use_half=False,
                            )

                            cls_attn_masks = []
                            cls_attn_masks_gt = []

                            person_filename = batch["person_filenames"][batch_idx]
                            person_filename = clean_decode_prompt(
                                person_filename, self.tokenizer, remove_blank=True
                            )

                            for mask_id in range(len(batch["class_ids"][batch_idx].nonzero())):

                                curr_placeholder_token_id = self.placeholder_token_ids[
                                    batch["token_ids"][batch_idx][mask_id]]
                                curr_class_token_id = batch["class_ids"][batch_idx][mask_id]

                                if self.args.with_prior_preservation:

                                    # agg_attn [24,24,77]
                                    class_idx = ((
                                        batch["input_ids"][batch_idx] == curr_class_token_id
                                    ).nonzero().item())

                                    cls_attn_mask = agg_attn_prior[..., class_idx]
                                    cls_attn_mask = (cls_attn_mask / cls_attn_mask.max())
                                    cls_attn_masks.append(cls_attn_mask)

                                    if self.args.apply_masked_prior:
                                        # use first infered attn_mask as the groundtruth attn_mask
                                        if curr_class_token_id.item(
                                        ) not in self.attn_mask_cache[person_filename].keys():
                                            self.attn_mask_cache[person_filename][
                                                curr_class_token_id.item()] = cls_attn_mask.detach()
                                            self.save_attn_mask_cache = True

                                        cls_attn_masks_gt.append(
                                            self.attn_mask_cache[person_filename][
                                                curr_class_token_id.item()]
                                        )

                                asset_idx = ((
                                    batch["input_ids"][curr_cond_batch_idx] ==
                                    curr_placeholder_token_id
                                ).nonzero().item())

                                # <asset>
                                for offset in range(1):
                                    asset_attn_mask = agg_attn[..., asset_idx + offset]
                                    asset_attn_mask = (asset_attn_mask / asset_attn_mask.max())

                                    attn_loss += F.mse_loss(
                                        GT_masks[mask_id, 0].float(),
                                        asset_attn_mask.float(),
                                        reduction="mean",
                                    )

                            if self.args.with_prior_preservation:

                                max_cls_mask_all = torch.max(
                                    torch.stack(cls_attn_masks), dim=0, keepdims=True
                                ).values
                                cls_masks.append(max_cls_mask_all)

                                if self.args.apply_masked_prior:
                                    max_cls_mask_gt_all = torch.max(
                                        torch.stack(cls_attn_masks_gt), dim=0, keepdims=True
                                    ).values
                                    cls_masks_gt.append(max_cls_mask_gt_all)

                        if self.args.with_prior_preservation:
                            cls_mask = F.interpolate(
                                torch.stack(cls_masks), size=(self.mask_res, self.mask_res)
                            )

                            syn_rgb_loss = F.mse_loss(
                                syn_pred.float() * downsampled_syn_mask.float(),
                                syn_target.float() * downsampled_syn_mask.float(),
                                reduction="mean",
                            ) * float(
                                self.args.syn_loss_weight.split(",")[0]
                            ) / self.args.train_batch_size

                            syn_norm_rgb_loss = F.mse_loss(
                                syn_norm_pred.float() * downsampled_syn_mask.float(),
                                syn_norm_target.float() * downsampled_syn_mask.float(),
                                reduction="mean",
                            ) * float(
                                self.args.syn_loss_weight.split(",")[1]
                            ) / self.args.train_batch_size

                            loss += syn_rgb_loss + syn_norm_rgb_loss
                            logs["l_syn"] = syn_rgb_loss.detach().item() + syn_norm_rgb_loss.detach(
                            ).item()

                            if self.args.apply_masked_prior:
                                cls_mask_gt = F.interpolate(
                                    torch.stack(cls_masks_gt), size=(self.mask_res, self.mask_res)
                                )
                                prior_pred = prior_pred * cls_mask_gt
                                prior_target = prior_target * cls_mask_gt

                                mask_loss = F.mse_loss(
                                    cls_mask.float(),
                                    cls_mask_gt.float(),
                                    reduction="mean",
                                ) * self.args.mask_loss_weight / self.args.train_batch_size

                                loss += mask_loss
                                logs["l_mask"] = mask_loss.detach().item()

                            prior_rgb_loss = F.mse_loss(
                                prior_pred.float(),
                                prior_target.float(),
                                reduction="mean",
                            ) * self.args.prior_loss_weight / self.args.train_batch_size

                            prior_loss = prior_rgb_loss

                            loss += prior_loss
                            logs["l_prior"] = prior_loss.detach().item()

                        attn_loss = self.args.lambda_attention * (
                            attn_loss / self.args.train_batch_size
                        )
                        loss += attn_loss
                        logs["l_attn"] = attn_loss.detach().item()

                    self.accelerator.backward(loss)

                    # No need to keep the attention store
                    self.controller.attention_store = {}
                    self.controller.cur_step = 0

                    if self.accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
                            if self.args.train_text_encoder else self.unet.parameters()
                        )
                        self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=self.args.set_grads_to_none)

                    if global_step < self.args.phase1_train_steps:
                        # Let's make sure we don't update any embedding weights besides the newly added token
                        with torch.no_grad():
                            self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings(
                            ).weight[:-self.args.num_of_assets] = orig_embeds_params[:-self.args.
                                                                                     num_of_assets]

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % self.args.checkpointing_steps == 0 and global_step > self.args.phase1_train_steps:
                        if self.accelerator.is_main_process:

                            self.save_adaptor(step=global_step)

                    if (self.args.log_checkpoints and global_step % self.args.img_log_steps == 0):

                        img_logs_path = os.path.join(self.args.output_dir, "img_logs")
                        os.makedirs(img_logs_path, exist_ok=True)

                        if self.args.lambda_attention != 0:
                            last_sentence = batch["input_ids"][curr_cond_batch_idx]
                            last_sentence = clean_decode_prompt(last_sentence, self.tokenizer)

                            step_attn_vis = self.save_cross_attention_vis(
                                last_sentence,
                                batch_pixels=batch["pixel_values"]
                                [curr_cond_batch_idx].detach().cpu(),
                                attention_maps=agg_attn.detach().cpu(),
                                path=os.path.join(
                                    img_logs_path, f"{global_step:05}_step_raw_attn.jpg"
                                ),
                            )

                            self.accelerator.trackers[0].log({
                                "step_attn":
                                [wandb.Image(step_attn_vis, caption=f"{last_sentence}")]
                            })

                            if self.args.with_prior_preservation:
                                last_sentence = batch["input_ids"][batch_idx]
                                last_sentence = clean_decode_prompt(last_sentence, self.tokenizer)

                                prior_attn_vis = self.save_cross_attention_vis(
                                    last_sentence,
                                    batch_pixels=batch["pixel_values"][batch_idx].detach().cpu(),
                                    attention_maps=agg_attn_prior.detach().cpu(),
                                    path=os.path.join(
                                        img_logs_path, f"{global_step:05}_step_prior_attn.jpg"
                                    ),
                                )

                                self.accelerator.trackers[0].log({
                                    "prior_attn":
                                    [wandb.Image(prior_attn_vis, caption=f"{last_sentence}")]
                                })

                        self.controller.cur_step = 0
                        self.controller.attention_store = {}

                        full_rgb, full_rgb_raw = self.perform_full_inference()

                        vis_dict = {}

                        for idx, name in enumerate(["", "_raw"]):

                            cur_prompt = self.args.instance_prompt if name == "" else self.args.instance_prompt_raw
                            cur_rgb = full_rgb if name == "" else full_rgb_raw

                            full_agg_attn = self.aggregate_attention(
                                res=self.attn_res,
                                from_where=("up", "down"),
                                is_cross=True,
                                select=idx,
                                batch_size=2,
                            )
                            full_attn_vis = self.save_cross_attention_vis(
                                cur_prompt,
                                batch_pixels=cur_rgb,
                                attention_maps=full_agg_attn.detach().cpu(),
                                path=os.path.join(
                                    img_logs_path, f"{global_step:05}_full_attn{name}.jpg"
                                ),
                            )
                            vis_dict.update({
                                f"full_attn{name}":
                                [wandb.Image(full_attn_vis, caption=f"{cur_prompt}")]
                            })

                        self.accelerator.trackers[0].log(vis_dict)

                        self.controller.cur_step = 0
                        self.controller.attention_store = {}

                logs["loss"] = loss.detach().item()
                logs["lr"] = lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                torch.cuda.empty_cache()

                if global_step >= self.args.max_train_steps:
                    break

        self.accelerator.end_training()
        self.save_adaptor()

        # save again
        if self.save_attn_mask_cache:
            existed_masks = torch.load(cached_masks_path)
            for filename in existed_masks.keys():
                for mask_id in existed_masks[filename].keys():
                    if mask_id not in self.attn_mask_cache[filename].keys():
                        self.attn_mask_cache[filename][mask_id] = existed_masks[filename][mask_id]
            torch.save(self.attn_mask_cache, cached_masks_path)

    def save_adaptor(self, step=""):

        if self.args.use_peft != "none":

            unwarpped_unet = self.accelerator.unwrap_model(self.unet)
            unwarpped_unet.save_pretrained(
                os.path.join(self.args.output_dir, f"unet/{step}"),
                state_dict=self.accelerator.get_state_dict(self.unet)
            )
            logger.info(f"Saved unet to {os.path.join(self.args.output_dir, f'unet/{step}')}")

            if self.args.train_text_encoder:

                unwarpped_text_encoder = self.accelerator.unwrap_model(self.text_encoder)
                unwarpped_text_encoder.save_pretrained(
                    os.path.join(self.args.output_dir, f"text_encoder/{step}"),
                    state_dict=self.accelerator.get_state_dict(self.text_encoder),
                    save_embedding_layers=True,
                )
                logger.info(
                    f"Saved text_encoder to {os.path.join(self.args.output_dir, f'text_encoder/{step}')}"
                )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            )
            pipeline.save_pretrained(self.args.output_dir)

    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count

    def get_average_attention(self):
        average_attention = {
            key: [item / self.controller.cur_step for item in self.controller.attention_store[key]]
            for key in self.controller.attention_store
        }
        return average_attention

    def aggregate_attention(
        self,
        res: int,
        from_where: List[str],
        is_cross: bool,
        select: int,
        batch_size: int,
        use_half: bool = True
    ):
        out = []
        attention_maps = self.get_average_attention()
        num_pixels = res**2

        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    if use_half:
                        h = item.shape[0] // 2
                        cross_maps = item[h:].reshape(batch_size, -1, res, res,
                                                      item.shape[-1])[select]
                    else:
                        cross_maps = item.reshape(batch_size, -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)

        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    @torch.no_grad()
    def perform_full_inference(self, guidance_scale=7.5):
        self.unet.eval()
        self.text_encoder.eval()

        latents = torch.randn((2, 4, self.mask_res, self.mask_res), device=self.accelerator.device)
        uncond_input = self.tokenizer(
            ["", ""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.accelerator.device)
        input_ids = self.tokenizer(
            [self.args.instance_prompt, self.args.instance_prompt_raw],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.accelerator.device)

        cond_embeddings = self.text_encoder(input_ids)[0]
        uncond_embeddings = self.text_encoder(uncond_input)[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        for t in self.validation_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.validation_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
            noise_pred = pred.sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.validation_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents

        images = self.vae.decode(latents.to(self.weight_dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")

        self.accelerator.trackers[0].log({
            "validation": [wandb.Image(images[0], caption=f"{self.args.instance_prompt}")],
            "validation_raw": [wandb.Image(images[1], caption=f"{self.args.instance_prompt_raw}")],
        })

        self.unet.train()
        if self.args.train_text_encoder:
            self.text_encoder.train()

        return Image.fromarray(images[0]), Image.fromarray(images[1])

    @torch.no_grad()
    def save_cross_attention_vis(self, prompt, batch_pixels, attention_maps, path):

        tokens = self.tokenizer.encode(prompt)
        images = []

        if torch.is_tensor(batch_pixels):
            image = batch_pixels.permute(1, 2, 0).numpy()
            image = ((image + 1.0) * 0.5 * 255).round().astype("uint8")
            image = np.array(Image.fromarray(image).resize((256, 256)))
        elif isinstance(batch_pixels, Image.Image):
            image = np.array(batch_pixels.resize((256, 256)))

        image = ptp_utils.text_under_image(image, "raw pixels")
        images.append(image)

        for i in range(len(tokens)):
            asset_word = self.tokenizer.decode(int(tokens[i]))
            if ("asset" in asset_word) or (asset_word in self.gpt4v_response.keys()):
                image = attention_maps[:, :, i]
                image = 255 * image / image.max()
                image = image.unsqueeze(-1).expand(*image.shape, 3)
                image = image.numpy().astype(np.uint8)
                image = np.array(Image.fromarray(image).resize((256, 256)))
                image = ptp_utils.text_under_image(image, asset_word)
                images.append(image)

        vis = ptp_utils.view_images(np.stack(images, axis=0))

        vis.save(path)

        return vis



if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(False)
    SpatialDreambooth()
