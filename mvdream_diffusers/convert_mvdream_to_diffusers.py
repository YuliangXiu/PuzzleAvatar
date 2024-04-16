# Modified from https://github.com/huggingface/diffusers/blob/bc691231360a4cbc7d19a58742ebb8ed0f05e027/scripts/convert_original_stable_diffusion_to_diffusers.py

import argparse
import torch
import sys

sys.path.insert(0, ".")

from diffusers.models import (
    AutoencoderKL,
)
from omegaconf import OmegaConf
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import logging
from typing import Any
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor

from mv_unet import MultiViewUNetModel
from pipeline_mvdream import MVDreamPipeline
import kiui

logger = logging.get_logger(__name__)


def assign_to_checkpoint(
    paths,
    checkpoint,
    old_checkpoint,
    attention_paths_to_split=None,
    additional_replacements=None,
    config=None,
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming to them. It splits
    attention layers, and takes into account additional replacements that may arise.
    Assigns the weights to the new checkpoint.
    """
    assert isinstance(
        paths, list
    ), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            assert config is not None
            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape(
                (num_heads, 3 * channels // num_heads) + old_tensor.shape[1:]
            )
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if (
            attention_paths_to_split is not None
            and new_path in attention_paths_to_split
        ):
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        is_attn_weight = "proj_attn.weight" in new_path or (
            "attentions" in new_path and "to_" in new_path
        )
        shape = old_checkpoint[path["old"]].shape
        if is_attn_weight and len(shape) == 3:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        elif is_attn_weight and len(shape) == 4:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def create_vae_diffusers_config(original_config, image_size):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """

    
    if 'imagedream' in original_config.model.target:
        vae_params = original_config.model.params.vae_config.params.ddconfig
        _ = original_config.model.params.vae_config.params.embed_dim
        vae_key = "vae_model."
    else:
        vae_params = original_config.model.params.first_stage_config.params.ddconfig
        _ = original_config.model.params.first_stage_config.params.embed_dim
        vae_key = "first_stage_model."

    block_out_channels = [vae_params.ch * mult for mult in vae_params.ch_mult]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = {
        "sample_size": image_size,
        "in_channels": vae_params.in_channels,
        "out_channels": vae_params.out_ch,
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "block_out_channels": tuple(block_out_channels),
        "latent_channels": vae_params.z_channels,
        "layers_per_block": vae_params.num_res_blocks,
    }
    return config, vae_key


def convert_ldm_vae_checkpoint(checkpoint, config, vae_key):
    # extract state dict for VAE
    vae_state_dict = {}
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict[
        "encoder.conv_out.weight"
    ]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict[
        "encoder.norm_out.weight"
    ]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict[
        "encoder.norm_out.bias"
    ]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict[
        "decoder.conv_out.weight"
    ]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict[
        "decoder.norm_out.weight"
    ]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict[
        "decoder.norm_out.bias"
    ]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len(
        {
            ".".join(layer.split(".")[:3])
            for layer in vae_state_dict
            if "encoder.down" in layer
        }
    )
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key]
        for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len(
        {
            ".".join(layer.split(".")[:3])
            for layer in vae_state_dict
            if "decoder.up" in layer
        }
    )
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key]
        for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [
            key
            for key in down_blocks[i]
            if f"down.{i}" in key and f"down.{i}.downsample" not in key
        ]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[
                f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"
            ] = vae_state_dict.pop(f"encoder.down.{i}.downsample.conv.weight")
            new_checkpoint[
                f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"
            ] = vae_state_dict.pop(f"encoder.down.{i}.downsample.conv.bias")

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        paths,
        new_checkpoint,
        vae_state_dict,
        additional_replacements=[meta_path],
        config=config,
    )
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key
            for key in up_blocks[block_id]
            if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[
                f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"
            ] = vae_state_dict[f"decoder.up.{block_id}.upsample.conv.weight"]
            new_checkpoint[
                f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"
            ] = vae_state_dict[f"decoder.up.{block_id}.upsample.conv.bias"]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        paths,
        new_checkpoint,
        vae_state_dict,
        additional_replacements=[meta_path],
        config=config,
    )
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        new_item = shave_segments(
            new_item, n_shave_prefix_segments=n_shave_prefix_segments
        )

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        new_item = new_item.replace("q.weight", "to_q.weight")
        new_item = new_item.replace("q.bias", "to_q.bias")

        new_item = new_item.replace("k.weight", "to_k.weight")
        new_item = new_item.replace("k.bias", "to_k.bias")

        new_item = new_item.replace("v.weight", "to_v.weight")
        new_item = new_item.replace("v.bias", "to_v.bias")

        new_item = new_item.replace("proj_out.weight", "to_out.0.weight")
        new_item = new_item.replace("proj_out.bias", "to_out.0.bias")

        new_item = shave_segments(
            new_item, n_shave_prefix_segments=n_shave_prefix_segments
        )

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def create_unet_config(original_config) -> Any:
    return OmegaConf.to_container(
        original_config.model.params.unet_config.params, resolve=True
    )


def convert_from_original_mvdream_ckpt(checkpoint_path, original_config_file, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # print(f"Checkpoint: {checkpoint.keys()}")
    torch.cuda.empty_cache()

    original_config = OmegaConf.load(original_config_file)
    # print(f"Original Config: {original_config}")
    prediction_type = "epsilon"
    image_size = 256
    num_train_timesteps = (
        getattr(original_config.model.params, "timesteps", None) or 1000
    )
    beta_start = getattr(original_config.model.params, "linear_start", None) or 0.02
    beta_end = getattr(original_config.model.params, "linear_end", None) or 0.085
    scheduler = DDIMScheduler(
        beta_end=beta_end,
        beta_schedule="scaled_linear",
        beta_start=beta_start,
        num_train_timesteps=num_train_timesteps,
        steps_offset=1,
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type=prediction_type,
    )
    scheduler.register_to_config(clip_sample=False)

    unet_config = create_unet_config(original_config)

    # remove unused configs
    unet_config.pop('legacy', None)
    unet_config.pop('use_linear_in_transformer', None)
    unet_config.pop('use_spatial_transformer', None)
    
    unet_config.pop('ip_mode', None)
    unet_config.pop('with_ip', None)

    unet = MultiViewUNetModel(**unet_config)
    unet.register_to_config(**unet_config)
    # print(f"Unet State Dict: {unet.state_dict().keys()}")
    unet.load_state_dict(
        {
            key.replace("model.diffusion_model.", ""): value
            for key, value in checkpoint.items()
            if key.replace("model.diffusion_model.", "") in unet.state_dict()
        }
    )
    for param_name, param in unet.state_dict().items():
        set_module_tensor_to_device(unet, param_name, device=device, value=param)

    # Convert the VAE model.
    vae_config, vae_key = create_vae_diffusers_config(original_config, image_size=image_size)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config, vae_key)

    if (
        "model" in original_config
        and "params" in original_config.model
        and "scale_factor" in original_config.model.params
    ):
        vae_scaling_factor = original_config.model.params.scale_factor
    else:
        vae_scaling_factor = 0.18215  # default SD scaling factor

    vae_config["scaling_factor"] = vae_scaling_factor

    with init_empty_weights():
        vae = AutoencoderKL(**vae_config)

    for param_name, param in converted_vae_checkpoint.items():
        set_module_tensor_to_device(vae, param_name, device=device, value=param)

    # we only supports SD 2.1 based model
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="tokenizer")
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="text_encoder").to(device=device)  # type: ignore
    
    # imagedream variant
    if unet.ip_dim > 0:
        feature_extractor: CLIPImageProcessor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        image_encoder: CLIPVisionModel = CLIPVisionModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    else:
        feature_extractor = None
        image_encoder = None

    pipe = MVDreamPipeline(
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
    )

    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        help="The YAML config file corresponding to the original architecture.",
    )
    parser.add_argument(
        "--to_safetensors",
        action="store_true",
        help="Whether to store pipeline in safetensors format or not.",
    )
    parser.add_argument(
        "--half", action="store_true", help="Save weights in half precision."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether to test inference after convertion.",
    )
    parser.add_argument(
        "--dump_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output model.",
    )
    parser.add_argument(
        "--device", type=str, help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)"
    )
    args = parser.parse_args()

    args.device = torch.device(
        args.device
        if args.device is not None
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    pipe = convert_from_original_mvdream_ckpt(
        checkpoint_path=args.checkpoint_path,
        original_config_file=args.original_config_file,
        device=args.device,
    )

    if args.half:
        pipe.to(torch_dtype=torch.float16)

    print(f"Saving pipeline to {args.dump_path}...")
    pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)

    if args.test:
        try:
            # mvdream
            if pipe.unet.ip_dim == 0:
                print(f"Testing each subcomponent of the pipeline...")
                images = pipe(
                    prompt="Head of Hatsune Miku",
                    negative_prompt="painting, bad quality, flat",
                    output_type="pil",
                    guidance_scale=7.5,
                    num_inference_steps=50,
                    device=args.device,
                )
                for i, image in enumerate(images):
                    image.save(f"test_image_{i}.png")  # type: ignore

                print(f"Testing entire pipeline...")
                loaded_pipe = MVDreamPipeline.from_pretrained(args.dump_path)  # type: ignore
                images = loaded_pipe(
                    prompt="Head of Hatsune Miku",
                    negative_prompt="painting, bad quality, flat",
                    output_type="pil",
                    guidance_scale=7.5,
                    num_inference_steps=50,
                    device=args.device,
                )
                for i, image in enumerate(images):
                    image.save(f"test_image_{i}.png")  # type: ignore
            # imagedream
            else:
                input_image = kiui.read_image('data/anya_rgba.png', mode='float')
                print(f"Testing each subcomponent of the pipeline...")
                images = pipe(
                    image=input_image,
                    prompt="",
                    negative_prompt="",
                    output_type="pil",
                    guidance_scale=5.0,
                    num_inference_steps=50,
                    device=args.device,
                )
                for i, image in enumerate(images):
                    image.save(f"test_image_{i}.png")  # type: ignore

                print(f"Testing entire pipeline...")
                loaded_pipe = MVDreamPipeline.from_pretrained(args.dump_path)  # type: ignore
                images = loaded_pipe(
                    image=input_image,
                    prompt="",
                    negative_prompt="",
                    output_type="pil",
                    guidance_scale=5.0,
                    num_inference_steps=50,
                    device=args.device,
                )
                for i, image in enumerate(images):
                    image.save(f"test_image_{i}.png")  # type: ignore
                

            print("Inference test passed!")
        except Exception as e:
            print(f"Failed to test inference: {e}")
