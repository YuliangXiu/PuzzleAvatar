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
import warnings

from transformers import PretrainedConfig


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default=None,
        help="The name of the project to which the model will be uploaded.",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default=None,
        help="Gender of instance data",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="a photo at the beach",
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--no_prior_preservation",
        action="store_false",
        help="Flag to add prior preservation loss.",
        dest="with_prior_preservation"
    )
    parser.add_argument(
        "--use_shape_description",
        action="store_true",
        help="Flag to use shape description loss.",
    )

    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--mask_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--syn_loss_weight",
        type=str,
        default="2.0,2.0",
        help="The weight of synthetic prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=1993, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--no_train_text_encoder",
        action="store_false",
        help=
        "Whether to train the text encoder. If set, the text encoder should be float32 precision.",
        dest="train_text_encoder"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--phase1_train_steps",
        type=int,
        default="400",
        help="Number of trainig steps for the first phase.",
    )
    parser.add_argument(
        "--phase2_train_steps",
        type=int,
        default="400",
        help="Number of trainig steps for the second phase.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=
        "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--initial_learning_rate",
        type=float,
        default=5e-4,
        help="The LR for the Textual Inversion steps.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=
        "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="piecewise_constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup", "piecewise_constant"]'
        ),
    )
    parser.add_argument(
        "--lr_step_rules",
        type=str,
        default="1:2000,0.1",
        help=(
            "A list of tuples, where each tuple contains the epoch and the learning rate to use at that epoch. The"
            " learning rate will be kept constant after the last epoch."
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="offline",
        help="The mode to use for wandb logging.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    # boft args
    parser.add_argument(
        "--use_peft",
        type=str,
        default="none",
        help="which peft to use, can be 'none', 'lora' or 'boft'",
    )
    parser.add_argument("--boft_block_num", type=int, default=4, help="The number of BOFT blocks")
    parser.add_argument("--boft_block_size", type=int, default=0, help="The size of BOFT blocks")
    parser.add_argument(
        "--boft_n_butterfly_factor", type=int, default=2, help="The number of butterfly factors"
    )
    parser.add_argument("--boft_bias_fit", action="store_true", help="Whether to use bias fit")
    parser.add_argument(
        "--boft_dropout",
        type=float,
        default=0.1,
        help="BOFT dropout, only used if use_peft is 'boft'",
    )
    parser.add_argument(
        "--boft_bias",
        type=str,
        default="none",
        help=
        "Bias type for BOFT. Can be 'none', 'all' or 'boft_only', only used if use_peft is 'boft'",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="The number of Lora rank",
    )

    parser.add_argument("--lambda_attention", type=float, default=1e-2)
    parser.add_argument("--img_log_steps", type=int, default=200)
    parser.add_argument("--num_of_assets", type=int, default=1)
    parser.add_argument("--initializer_tokens", type=str, nargs="+", default=[])
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="<asset>",
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--do_not_apply_masked_loss",
        action="store_false",
        help="Use masked loss instead of standard epsilon prediciton loss",
        dest="apply_masked_loss"
    )
    parser.add_argument(
        "--do_not_apply_masked_prior",
        action="store_false",
        help="Use masked loss instead of standard epsilon prediciton loss",
        dest="apply_masked_prior"
    )
    parser.add_argument(
        "--log_checkpoints",
        action="store_true",
        help="Indicator to log intermediate model checkpoints",
    )
    parser.add_argument(
        "--use_view_prompt",
        action="store_true",
        help="Indicator to use view prompt",
    )

    args = parser.parse_args()

    if args.train_text_encoder:
        args.mixed_precision = "no"

    with open(os.path.join(args.instance_data_dir, 'gpt4v_simple.json'), 'r') as f:
        gpt4v_response = json.load(f)
    args.gender = 'man' if gpt4v_response['gender'] in ['man', 'male'] else 'woman'
    gpt4v_classes = list(gpt4v_response.keys())

    gpt4v_classes.remove("gender")
    for key in ["eyeglasses", "sunglasses", "glasses"]:
        if key in gpt4v_classes:
            gpt4v_classes.remove(key)

    args.num_of_assets = len(gpt4v_classes)
    args.initializer_tokens = gpt4v_classes

    assert len(args.initializer_tokens) == 0 or len(args.initializer_tokens) == args.num_of_assets
    args.max_train_steps = args.phase1_train_steps + args.phase2_train_steps

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args, gpt4v_response


def clean_decode_prompt(prompt, tokenizer, remove_blank=False):

    prompt = prompt[(prompt != 0) & (prompt != 49406) & (prompt != 49407)]
    prompt = tokenizer.decode(prompt)
    if remove_blank:
        prompt = prompt.replace(" ", "")

    return prompt
