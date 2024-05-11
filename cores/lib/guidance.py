import os.path as osp
from jutils import image_utils
from diffusers import (
    ControlNetModel, DDIMScheduler, PNDMScheduler, UNet2DConditionModel, DiffusionPipeline
)
from transformers import CLIPTextModel, CLIPTokenizer, logging
from mvdream_diffusers.pipeline_mvdream import MVDreamPipeline

# suppress partial model loading warning
logging.set_verbosity_error()

import numpy as np
import PIL
import os
from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.cuda.amp import custom_bwd, custom_fwd
from diffusers.utils.import_utils import is_xformers_available


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        return torch.zeros([1], device=input_tensor.device,
                           dtype=input_tensor.dtype)    # dummy loss value

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        placeholders,
        use_peft,
        sd_version='2-1',
        hf_key=None,
        sd_step_range=[0.2, 0.98],
        subdiv_step=[3000],
        iters=8000,
        controlnet=None,
        lora=None,
        cfg=None,
        head_hf_key=None
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.sd_version = sd_version
        self.placeholders = placeholders
        self.use_peft = use_peft
        self.base_model_key = None
        self.res = 768
        self.sd_step_range = sd_step_range
        self.subdiv_step = subdiv_step
        self.iters = iters

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key

        self.mv_mode = False
        # import pdb; pdb.set_trace()
        if self.sd_version == '2-1':
            self.base_model_key = os.environ.get(
                'BASE_MODEL', "stabilityai/stable-diffusion-2-1-base"
            )
        elif self.sd_version == '2-0':
            self.base_model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1-5':
            self.base_model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == 'mv':
            self.base_model_key = cfg.pretrain
            self.mv_mode = True
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        if 'base' in self.base_model_key:
            self.res = 512
        elif self.mv_mode:
            self.res = 256
        else:
            self.res = 768

        if self.use_peft != 'none':

            pipe = DiffusionPipeline.from_pretrained(
                self.base_model_key,
                torch_dtype=torch.float32,
                requires_safety_checker=False,
            ).to(self.device)

            # add tokens
            num_added_tokens = pipe.tokenizer.add_tokens(self.placeholders)
            print(f"Added {num_added_tokens} tokens")
            pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

            # load peft model
            from peft import PeftModel
            self.text_encoder = PeftModel.from_pretrained(
                pipe.text_encoder, join(model_key, 'text_encoder')
            )
            self.unet = PeftModel.from_pretrained(pipe.unet, join(model_key, 'unet'))
            self.vae = pipe.vae
            self.tokenizer = pipe.tokenizer

        else:
            if self.sd_version == 'mv':
                Pipeline = MVDreamPipeline
            else:
                Pipeline = DiffusionPipeline

            pipe = Pipeline.from_pretrained(
                self.base_model_key,
                torch_dtype=torch.float32,
                requires_safety_checker=False,
                trust_remote_code=True,
            ).to(self.device)
            self.pipe = pipe

            self.text_encoder = pipe.text_encoder
            self.unet = pipe.unet
            self.vae = pipe.vae
            self.tokenizer = pipe.tokenizer

            if self.placeholders != 'sks':

                num_added_tokens = self.tokenizer.add_tokens(self.placeholders)
                print(f"Added {num_added_tokens} tokens")
                self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # enable FreeU
        # self.unet.enable_freeu(s1=0.9, s2=0.2, b1=1.4, b2=1.6)

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()

        self.text_encoder.eval()
        self.unet.eval()
        self.vae.eval()

        print(f'[INFO] loaded PEFT adapters!')

        self.use_head_model = head_hf_key is not None

        if self.use_head_model:
            self.tokenizer_head = CLIPTokenizer.from_pretrained(head_hf_key, subfolder="tokenizer")
            self.text_encoder_head = CLIPTextModel.from_pretrained(
                head_hf_key, subfolder="text_encoder"
            ).to(self.device)
            self.unet_head = UNet2DConditionModel.from_pretrained(head_hf_key,
                                                                  subfolder="unet").to(self.device)
        else:
            self.tokenizer_head = self.tokenizer
            self.text_encoder_head = self.text_encoder
            self.unet_head = self.unet

        # self.scheduler = PNDMScheduler.from_pretrained(self.base_model_key, subfolder="scheduler")
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * sd_step_range[0])
        self.max_step = int(self.num_train_timesteps * sd_step_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)    # for convenience

        if controlnet is None:
            self.controlnet = None
        else:
            self.controlnet = ControlNetModel.from_pretrained(controlnet).to(self.device)

        if lora is not None:
            self.unet.load_attn_procs(lora)

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt, is_face=False):
        # print('text prompt: [positive]', prompt, '[negative]', negative_prompt)
        if not is_face:
            tokenizer = self.tokenizer
            text_encoder = self.text_encoder
        else:
            tokenizer = self.tokenizer_head
            text_encoder = self.text_encoder_head

        # Tokenize text and get embeddings
        text_input = tokenizer(
            prompt,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        negative_input = tokenizer(
            negative_prompt,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            return_tensors='pt'
        )

        uncond_input = tokenizer(
            "", padding='max_length', max_length=tokenizer.model_max_length, return_tensors='pt'
        )

        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(self.device))[0]
            negative_embeddings = text_encoder(negative_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([negative_embeddings, text_embeddings, uncond_embeddings])
        return text_embeddings

    def train_step(
        self,
        text_embeddings,
        pred_rgb,
        guidance_scale=7.5,
        controlnet_hint=None,
        controlnet_conditioning_scale=1.0,
        is_face=False,
        cur_epoch=0,
        stage="geometry",
        camera=None,
        debug=False,
        **kwargs
    ):
        if is_face:
            unet = self.unet_head
        else:
            unet = self.unet

        if controlnet_hint is not None:
            controlnet_hint = self.controlnet_hint_conversion(controlnet_hint, self.res, self.res)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device
        )
        # print(t)

        # encode image into latents with vae, requires grad!
        # pred_img = F.interpolate(pred_rgb, (self.res, self.res), mode='bicubic', align_corners=True)
        pred_img = F.adaptive_avg_pool2d(pred_rgb, self.res)

        latents = self.encode_imgs(pred_img)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            # print('add noise', latents.shape)
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 3)
            if controlnet_hint is not None:
                print('controlnet_hint')
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=controlnet_hint,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False
                )
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            else:
                # print('nohint', latent_model_input.shape, t.shape, text_embeddings.shape)
                if self.mv_mode:
                    noise_pred = self.forward_mv_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        camera=camera,
                    )
                else:
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                    ).sample

        # perform guidance (high scale from paper!)
        if self.scheduler.config.prediction_type == "v_prediction":
            alphas_cumprod = self.scheduler.alphas_cumprod.to(
                device=latents_noisy.device, dtype=latents_noisy.dtype
            )
            alpha_t = alphas_cumprod[t]**0.5
            sigma_t = (1 - alphas_cumprod[t])**0.5

            noise_pred = latent_model_input * torch.cat([sigma_t] * 3, dim=0).view(
                -1, 1, 1, 1
            ) + noise_pred * torch.cat([alpha_t] * 3, dim=0).view(-1, 1, 1, 1)

        noise_pred_neg, noise_pred_text, noise_pred_null = noise_pred.chunk(3)

        # vanilla sds: w(t), sigma_t^2
        w = (1 - self.alphas[t])

        # # fantasia3d
        # if stage == "geometry":
        #     if cur_epoch <= 1000:
        #         w = (1 - self.alphas[t])
        #     else:
        #         w = (self.alphas[t]**0.5 * (1 - self.alphas[t]))
        # elif stage == "texture":
        #     if cur_epoch <= 1000:
        #         w = (self.alphas[t]**0.5 * (1 - self.alphas[t]))
        #     else:
        #         w = 1 / (1 - self.alphas[t])

        # original version from DreamFusion (https://dreamfusion3d.github.io/)
        # noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_null)

        # updated version inspired by HumanGuassian (https://arxiv.org/abs/2311.17061) and NFSD (https://arxiv.org/abs/2310.17590)

        classifier_pred = guidance_scale * (noise_pred_text - noise_pred_null)

        mask = (t < 200).int().view(-1, 1, 1, 1)
        negative_pred = mask * noise_pred_null + (1 - mask) * (noise_pred_null - noise_pred_neg)

        grad = w.view(-1, 1, 1, 1) * (classifier_pred + negative_pred)

        grad_norm = torch.norm(grad, dim=-1, keepdim=True) + 1e-8
        grad = grad_norm.clamp(max=0.1) * grad / grad_norm

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)

        rtn_extra = {}
        if debug:
            with torch.no_grad():
                # debug to save signal
                # if eval_guidance:
                # print('debug', noise_pred.shape, latents.shape)
                self.scheduler.set_timesteps(self.num_train_timesteps)
                latents = self.scheduler.step(noise_pred_text, t, latents)['pred_original_sample']
                imgs = self.decode_latents(latents)
                rtn_extra['1-step'] = imgs
                rtn_extra['orig'] = pred_img
                # num_inference_steps = 20

                # self.scheduler.set_timesteps(num_inference_steps)
                # for i, t in enumerate(self.scheduler.timesteps):
                #     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                #     # predict the noise residual
                #     with torch.no_grad():
                #         noise_pred = self.forward_mv_unet(
                #             latent_model_input, t, encoder_hidden_states=text_embeddings,
                #             camera=camera,
                #         )

                #     # perform guidance
                #     # noise_pred_null, noise_pred_text = noise_pred.chunk(2)
                #     noise_pred_neg, noise_pred_text, noise_pred_null = noise_pred.chunk(3)

                #     noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_null)

                #     # compute the previous noisy sample x_t -> x_t-1
                #     latent_model_input = self.scheduler.step(noise_pred, t, latents)['prev_sample']
                # imgs_multi = self.decode_latents(latent_model_input)
                # rtn_extra['multi-step'] = imgs_multi
                # print(imgs_multi.shape, rtn_extra['1-step'].shape, rtn_extra['orig'].shape)

        return loss, rtn_extra

    def produce_latents(
        self,
        text_embeddings,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        camera=None,
    ):

        if latents is None:
            latents = torch.randn(
                (text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                device=self.device
            )

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    if self.mv_mode:
                        # timestep = t.expand(latent_model_input.shape[0])
                        # print(latent_model_input.shape, t.shape, timestep.shape, text_embeddings.shape, camera.shape)
                        noise_pred = self.forward_mv_unet(
                            latent_model_input,
                            t,
                            text_embeddings,
                            camera=camera,
                        )
                        # print(noise_pred.shape)
                    else:
                        noise_pred = self.unet(
                            latent_model_input, t, encoder_hidden_states=text_embeddings
                        )['sample']

                # perform guidance
                noise_pred_null, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_null)
                # print('noise_pred', noise_pred.shape, latents.shape)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(
        self,
        prompts,
        negative_prompts='',
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        camera=None,
    ):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)    # [2, 77, 768]
        # text_embeddings = torch.cat([negative_embeddings, text_embeddings, uncond_embeddings])
        text_embeds = torch.stack([text_embeds[2], text_embeds[1]], 0)    # (2, 77, 768)
        text_embeds = text_embeds.unsqueeze(1).repeat(1, 4, 1, 1).reshape(
            -1, *text_embeds.shape[1:]
        )    # (8, 77, 768)
        latents_orig = latents.clone() if latents is not None else None
        # Text embeds -> img latents
        latents = self.produce_latents(
            text_embeds,
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            camera=camera
        )    # [1, 4, 64, 64]
        # print('produce latents', latents.shape)

        # Img latents -> imgs
        imgs = self.decode_latents(latents)    # [1, 3, 512, 512]
        img_pt = imgs

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs, img_pt, {'text_embeds': text_embeds, 'latents': latents_orig}

    def controlnet_hint_conversion(self, controlnet_hint, height, width, num_images_per_prompt=1):
        channels = 3
        if isinstance(controlnet_hint, torch.Tensor):
            # torch.Tensor: acceptble shape are any of chw, bchw(b==1) or bchw(b==num_images_per_prompt)
            shape_chw = (channels, height, width)
            shape_bchw = (1, channels, height, width)
            shape_nchw = (num_images_per_prompt, channels, height, width)
            if controlnet_hint.shape in [shape_chw, shape_bchw, shape_nchw]:
                controlnet_hint = controlnet_hint.to(
                    dtype=self.controlnet.dtype, device=self.controlnet.device
                )
                if controlnet_hint.shape != shape_nchw:
                    controlnet_hint = controlnet_hint.repeat(num_images_per_prompt, 1, 1, 1)
                return controlnet_hint
            else:
                raise ValueError(
                    f"Acceptble shape of `controlnet_hint` are any of ({channels}, {height}, {width}),"
                    + f" (1, {channels}, {height}, {width}) or ({num_images_per_prompt}, " +
                    f"{channels}, {height}, {width}) but is {controlnet_hint.shape}"
                )
        elif isinstance(controlnet_hint, np.ndarray):
            # np.ndarray: acceptable shape is any of hw, hwc, bhwc(b==1) or bhwc(b==num_images_per_promot)
            # hwc is opencv compatible image format. Color channel must be BGR Format.
            if controlnet_hint.shape == (height, width):
                controlnet_hint = np.repeat(
                    controlnet_hint[:, :, np.newaxis], channels, axis=2
                )    # hw -> hwc(c==3)
            shape_hwc = (height, width, channels)
            shape_bhwc = (1, height, width, channels)
            shape_nhwc = (num_images_per_prompt, height, width, channels)
            if controlnet_hint.shape in [shape_hwc, shape_bhwc, shape_nhwc]:
                controlnet_hint = torch.from_numpy(controlnet_hint.copy())
                controlnet_hint = controlnet_hint.to(
                    dtype=self.controlnet.dtype, device=self.controlnet.device
                )
                controlnet_hint /= 255.0
                if controlnet_hint.shape != shape_nhwc:
                    controlnet_hint = controlnet_hint.repeat(num_images_per_prompt, 1, 1, 1)
                controlnet_hint = controlnet_hint.permute(0, 3, 1, 2)    # b h w c -> b c h w
                return controlnet_hint
            else:
                raise ValueError(
                    f"Acceptble shape of `controlnet_hint` are any of ({width}, {channels}), " +
                    f"({height}, {width}, {channels}), " +
                    f"(1, {height}, {width}, {channels}) or " +
                    f"({num_images_per_prompt}, {channels}, {height}, {width}) but is {controlnet_hint.shape}"
                )
        elif isinstance(controlnet_hint, PIL.Image.Image):
            if controlnet_hint.size == (width, height):
                controlnet_hint = controlnet_hint.convert("RGB")    # make sure 3 channel RGB format
                controlnet_hint = np.array(controlnet_hint)    # to numpy
                controlnet_hint = controlnet_hint[:, :, ::-1]    # RGB -> BGR
                return self.controlnet_hint_conversion(
                    controlnet_hint, height, width, num_images_per_prompt
                )
            else:
                raise ValueError(
                    f"Acceptable image size of `controlnet_hint` is ({width}, {height}) but is {controlnet_hint.size}"
                )
        else:
            raise ValueError(
                f"Acceptable type of `controlnet_hint` are any of torch.Tensor, np.ndarray, PIL.Image.Image but is {type(controlnet_hint)}"
            )

    def forward_mv_unet(self, latents, t, encoder_hidden_states, camera, get_inp=False):
        """interface, hijack multiple frames.

        :param latents: (BF, C, H, W)
        :param t: (1, )
        :param encoder_hidden_states: (BF, L, D)
        :param camera: (F, 16)
        """
        #  torch.Size([12, 4, 32, 32]) torch.Size([1]) torch.Size([12, 77, 1024]) torch.Size([4, 16])
        # print('forward_mv_unet', latents.shape, t.shape, encoder_hidden_states.shape, camera.shape)
        device = latents.device
        # expand all inputs to have batch BF
        BF = encoder_hidden_states.shape[0]
        # import pickle
        # with open("camera_list_mv_unet.pkl", "wb") as f:
        #     pickle.dump(camera.reshape(-1, 4, 4), f)
        # assert False

        if camera is None:
            F = 1
        else:
            F = camera.shape[0]
            # if latents.shape[0] == B:
            #     latents = latents.unsqueeze(1).expand(B, F, -1, -1, -1).reshape(-1, *latents.shape[1:])

            # encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(B, F, -1, -1).reshape(-1, *encoder_hidden_states.shape[1:])
            B = BF // F
            # print(B, F)
            camera = camera.unsqueeze(0).expand(B, F, -1).reshape(-1, *camera.shape[1:])

        # print('camera', camera.shape)
        # print('latents num', latents[...,0, 0, 0], t) # c, h, w
        t = t.expand(BF).to(device)
        if get_inp:
            return {
                'x': latents,
                'timesteps': t,
                'context': encoder_hidden_states,
                'camera': camera,
                'num_frames': F,
            }
        # print('after expand', latents.shape, t.shape, encoder_hidden_states.shape, camera.shape)
        # forward
        noise_pred = self.unet(
            latents,
            t,
            encoder_hidden_states,
            camera=camera,
            num_frames=F,
        )    # (BF, C, H, W)

        # only use output F=0 # nah!!!
        # noise_pred = noise_pred.reshape(B, F, *noise_pred.shape[1:])[:, 0] # (B, C, H, W)
        return noise_pred


if __name__ == '__main__':

    import argparse

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument(
        '--sd_version',
        type=str,
        default='2.1',
        choices=['1.5', '2.0', '2.1'],
        help="stable diffusion version"
    )
    parser.add_argument(
        '--hf_key', type=str, default=None, help="hugging face Stable diffusion model key"
    )
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    from cores.main_mc import load_config
    from cores.lib.provider import ViewDataset
    cfg = load_config(opt.config, default_path="configs/default.yaml")

    seed_everything(opt.seed)

    device = torch.device('cuda')

    test_loader = ViewDataset(
        cfg, device=device, type='test', H=cfg.test.H, W=cfg.test.W, size=100, render_head=True
    ).dataloader()
    sd = StableDiffusion(device, opt.sd_version, opt.hf_key)
    # visualize image

    plt.show()
    # for test_data in test_loader:

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    plt.imshow(imgs[0])
