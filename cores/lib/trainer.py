import glob
import os
import random
import time
import pprint

import cv2
import imageio
import numpy as np
import tensorboardX
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import tqdm
from PIL import Image
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from thirdparties.lpips import LPIPS

from .annotators import Cannydetector, HEDdetector
from .camera_utils import *
from .loss_utils import *


def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Trainer(object):
    def __init__(
        self,
        name,    # name of this experiment
        cfg,    # extra conf
        model,    # network 
        guidance,    # guidance network
        criterion=None,    # loss function, if None, assume inline implementation in train_step
        optimizer=None,    # optimizer
        ema_decay=None,    # if use EMA, set the decay
        lr_scheduler=None,    # scheduler
        metrics=[],    # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
        local_rank=0,    # which GPU am I
        world_size=1,    # total num of GPUs
        device=None,    # device to use, usually setting to None is OK. (auto choose device)
        mute=False,    # whether to mute all print
        fp16=False,    # amp optimize level
        eval_interval=1,    # eval once every $ epoch
        max_keep_ckpt=2,    # max num of saved ckpts in disk
        workspace='workspace',    # workspace to save logs & ckpts
        best_mode='min',    # the smaller/larger result, the better
        use_loss_as_metric=True,    # use loss as the first metric
        report_metric_at_train=False,    # also report metrics at training
        use_checkpoint="latest",    # which ckpt to use at init time
        pretrained=None,
        subdiv_steps=[5000],
        use_tensorboardX=True,    # whether to use tensorboard for logging
        scheduler_update_every_step=False,    # whether to call scheduler.step() after every train step
    ):

        self.name = name
        self.cfg = cfg
        self.stage = self.cfg.stage
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.subdiv_steps = subdiv_steps
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        )
        self.console = Console()

        model.to(self.device)
        model.mesh.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model
        self._load_input_image()

        # guide model
        self.guidance = guidance

        # text prompt
        if self.guidance is not None:

            self.prepare_text_embeddings()

        else:
            self.text_z = None

        # try out torch 2.0
        if torch.__version__[0] == '2' and torch.cuda.get_device_capability(self.device)[0] >= 7:
            self.model = torch.compile(self.model)
            self.guidance = torch.compile(self.guidance)

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        self.optimizer_fn = optimizer
        self.lr_scheduler_fn = lr_scheduler

        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=0.001, weight_decay=5e-4
            )    # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1
            )    # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],    # metrics[0], or valid_loss
            "checkpoints": [],    # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}'
        )
        self.log(
            f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}'
        )

        self.load_pretrained(pretrained=pretrained)

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:    # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        if self.cfg.train.lambda_recon > 0 or self.cfg.train.lambda_normal > 0.:
            self.lpips_model = LPIPS(net='vgg').cuda()
            for param in self.lpips_model.parameters():
                param.requires_grad = False

        if self.cfg.guidance.controlnet_guidance_geometry:
            if self.cfg.guidance.controlnet_guidance_geometry == 'hed':
                self.controlnet_annotator = HEDdetector()
            elif self.cfg.guidance.controlnet_guidance_geometry == 'canny':
                self.controlnet_annotator = Cannydetector(100, 200)
            else:
                raise NotImplementedError

        self.render_openpose_training = self.cfg.guidance.controlnet_openpose_guidance
        self.render_openpose = self.cfg.guidance.controlnet_openpose_guidance

    def _load_input_image(self):
        if self.cfg.data.img is not None:
            self.input_image = Image.open(self.cfg.data.img)
            if self.input_image.width > 2048 or self.input_image.height > 2048:
                self.input_image = self.input_image.resize((2048, 2048))
            self.input_image = np.array(self.input_image) / 255
            self.input_mask = torch.tensor(self.input_image[..., 3],
                                           dtype=torch.float).to(self.device).unsqueeze(0)
            self.input_mask_edt = get_edt(self.input_mask.unsqueeze(0))[0]
            self.input_image = torch.tensor(self.input_image[..., :3],
                                            dtype=torch.float).to(self.device).permute(2, 0, 1)
            self.input_image = self.input_image * self.input_mask
            self.model.input_image = self.input_image
            self.model.input_mask = self.input_mask
        else:
            self.input_image = None
            self.input_mask = None
            self.input_mask_edt = None

        if self.cfg.data.front_normal_img is not None:
            self.normal_image = np.array(Image.open(self.cfg.data.front_normal_img)) / 255
            self.normal_mask = torch.tensor(self.normal_image[..., 3],
                                            dtype=torch.float).to(self.device).unsqueeze(0)
            self.normal_mask_edt = get_edt(self.normal_mask.unsqueeze(0))[0]
            self.normal_image = torch.tensor(self.normal_image[..., :3],
                                             dtype=torch.float).to(self.device).permute(2, 0, 1)
            self.normal_image = self.normal_image * self.normal_mask
        else:
            self.normal_mask = None
            self.normal_image = None

        if self.cfg.data.back_normal_img is not None:
            self.back_normal_image = np.array(Image.open(self.cfg.data.back_normal_img)) / 255
            self.back_normal_mask = torch.tensor(self.back_normal_image[..., 3],
                                                 dtype=torch.float).to(self.device).unsqueeze(0)
            self.back_normal_mask_edt = get_edt(self.back_normal_mask.unsqueeze(0))[0]
            self.back_normal_image = torch.tensor(
                self.back_normal_image[..., :3], dtype=torch.float
            ).to(self.device).permute(2, 0, 1)
            self.back_normal_image = self.back_normal_image * self.back_normal_mask
        else:
            self.back_normal_mask = None
            self.back_normal_image = None

        if self.cfg.data.loss_mask is not None:
            self.loss_mask = np.array(
                Image.open(self.cfg.data.loss_mask).resize(self.input_image.shape[1:])
            )[..., -1] / 255
            self.loss_mask_norm = np.array(Image.open(self.cfg.data.loss_mask).resize((512, 512))
                                          )[..., -1] / 255
            self.loss_mask = torch.tensor(self.loss_mask, dtype=torch.float).to(
                self.device
            ).unsqueeze(0) * self.input_mask
            self.loss_mask_norm = torch.tensor(self.loss_mask_norm, dtype=torch.float).to(
                self.device
            ).unsqueeze(0) * self.normal_mask
        elif self.cfg.data.occ_mask is not None:
            self.loss_mask = torch.tensor(
                self.get_loss_mask(self.cfg.data.occ_mask, None, self.input_image.shape[-2:]),
                dtype=torch.float
            ).to(self.device) * self.input_mask
            self.loss_mask_norm = torch.tensor(
                self.get_loss_mask(self.cfg.data.occ_mask, None, self.normal_mask.shape[-2:]),
                dtype=torch.float
            ).to(self.device) * self.normal_mask
        else:
            self.loss_mask = None
            self.loss_mask_norm = None

        if self.cfg.train.loss_mask_erosion is not None:
            input_size = max(self.input_mask.shape[0], self.input_mask.shape[1])
            normal_input_size = max(self.normal_mask.shape[0], self.normal_mask.shape[1])
            image_kernel_size = int(input_size * self.cfg.train.loss_mask_erosion)
            norm_kernel_size = int(normal_input_size * self.cfg.train.loss_mask_erosion)
            kernel = np.ones((image_kernel_size, image_kernel_size), np.float32)
            self.erosion_mask = torch.tensor(
                cv2.erode(self.input_mask.cpu().numpy()[0], kernel, cv2.BORDER_REFLECT)
            ).to(self.device).unsqueeze(0)
            norm_kernel = np.ones((norm_kernel_size, norm_kernel_size), np.float32)
            if self.normal_mask is not None:
                self.erosion_normal_mask = torch.tensor(
                    cv2.erode(self.normal_mask.cpu().numpy()[0], norm_kernel, cv2.BORDER_REFLECT)
                ).to(self.device).unsqueeze(0)
            else:
                self.erosion_normal_mask = None
            if self.back_normal_mask is not None:
                self.erosion_back_normal_mask = torch.tensor(
                    cv2.erode(
                        self.back_normal_mask.cpu().numpy()[0], norm_kernel, cv2.BORDER_REFLECT
                    )
                ).to(self.device).unsqueeze(0)
            else:
                self.erosion_back_normal_mask = None
        else:
            self.erosion_mask = None
            self.erosion_normal_mask = None
            self.erosion_back_normal_mask = None

        self.input_can_pos_map = None

    def get_loss_mask(self, occ_map_path, seg_path, img_size):
        occ_map = np.array(Image.open(occ_map_path).resize(img_size)) / 255
        if len(occ_map.shape) == 3:
            occ_map = occ_map[..., -1]
        if seg_path is not None:
            seg_map = np.array(Image.open(seg_path).resize(img_size)) / 255
            if len(seg_map.shape) == 3:
                seg_map = seg_map[..., -1]
            occ_map = (occ_map > 0) and (seg_map == 0)
        return occ_map

    # calculate the text embs.
    def prepare_text_embeddings(self):

        if self.cfg.guidance.text is None:
            self.log(f"[WARN] text prompt is not provided.")
            self.text_z = None
            return

        if not self.cfg.guidance.use_view_prompt:
            self.text_z = self.guidance.get_text_embeds([self.cfg.guidance.text],
                                                        [self.cfg.guidance.negative])
        else:
            print('get rgb text prompt')
            self.text_z_novd = self.guidance.get_text_embeds([self.cfg.guidance.text],
                                                             [self.cfg.guidance.negative])
            self.text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                # construct dir-encoded text
                text = f"{self.cfg.guidance.text}, {d} view, {self.cfg.guidance.text_extra}"

                negative_text = f"{self.cfg.guidance.negative}"

                # explicit negative dir-encoded text
                text_z = self.guidance.get_text_embeds([text], [negative_text])
                self.text_z.append(text_z)

            if self.cfg.train.face_sample_ratio > 0.:
                self.face_text_z_novd = self.guidance.get_text_embeds([
                    f"the headshot of {self.cfg.guidance.text_head}, {self.cfg.guidance.text_extra}"
                ], [self.cfg.guidance.negative])
                self.face_text_z = []
                prompt = self.cfg.guidance.text_head if (
                    self.cfg.guidance.text_head is not None
                ) and (len(self.cfg.guidance.text_head) > 0) else self.cfg.guidance.text
                for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                    # construct dir-encoded text
                    text = f"the headshot of {prompt}, {d} view, {self.cfg.guidance.text_extra}"

                    negative_text = f"{self.cfg.guidance.negative_normal}"

                    # explicit negative dir-encoded text
                    text_z = self.guidance.get_text_embeds([text], [negative_text], is_face=True)
                    self.face_text_z.append(text_z)

            if (self.cfg.guidance.normal_text
                is not None) and (len(self.cfg.guidance.normal_text) > 0):
                print('get normal text prompt')
                basic_prompt = self.cfg.guidance.text if (self.cfg.guidance.text_geo is None) or (
                    len(self.cfg.guidance.text_geo) == 0
                ) else self.cfg.guidance.text_geo
                self.normal_text_z_novd = self.guidance.get_text_embeds([
                    f"{self.cfg.guidance.normal_text} of {basic_prompt}, {self.cfg.guidance.normal_text_extra}"
                ], [self.cfg.guidance.negative_normal])
                self.normal_text_z = []
                for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                    # construct dir-encoded text
                    text = f"{self.cfg.guidance.normal_text} of {basic_prompt}, {d} view, {self.cfg.guidance.normal_text_extra}"

                    negative_text = f"{self.cfg.guidance.negative_normal}"

                    # explicit negative dir-encoded text
                    text_z = self.guidance.get_text_embeds([text], [negative_text])
                    self.normal_text_z.append(text_z)

                basic_prompt = self.cfg.guidance.text_head if (
                    self.cfg.guidance.text_head is not None
                ) and (len(self.cfg.guidance.text_head) > 0) else basic_prompt

                self.face_normal_text_z_novd = self.guidance.get_text_embeds([
                    f"{self.cfg.guidance.normal_text} of the headshot of {basic_prompt}, {self.cfg.guidance.normal_text_extra}"
                ], [self.cfg.guidance.negative_normal])
                self.face_normal_text_z = []
                for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                    # construct dir-encoded text
                    text = f"{self.cfg.guidance.normal_text} of the headshot of {basic_prompt}, {d} view, {self.cfg.guidance.normal_text_extra}"

                    negative_text = f"{self.cfg.guidance.negative_normal}"

                    # explicit negative dir-encoded text
                    text_z = self.guidance.get_text_embeds([text], [negative_text])
                    self.face_normal_text_z.append(text_z)
            if (self.cfg.guidance.textureless_text
                is not None) and (len(self.cfg.guidance.textureless_text)) > 0:
                print('get textureless text prompt')
                self.textureless_text_z_novd = self.guidance.get_text_embeds([
                    f"{self.cfg.guidance.textureless_text} of {self.cfg.guidance.text}, {self.cfg.guidance.textureless_text_extra}"
                ], [self.cfg.guidance.negative_textureless])
                self.textureless_text_z = []
                for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                    # construct dir-encoded text
                    text = f"{self.cfg.guidance.textureless_text} of {self.cfg.guidance.text}, {d} view, {self.cfg.guidance.textureless_text_extra}"

                    negative_text = f"{self.cfg.guidance.negative_textureless}"

                    # explicit negative dir-encoded text
                    text_z = self.guidance.get_text_embeds([text], [negative_text])
                    self.textureless_text_z.append(text_z)

                self.face_textureless_text_z_novd = self.guidance.get_text_embeds([
                    f"{self.cfg.guidance.textureless_text} of the headshot of {self.cfg.guidance.text_head}, {self.cfg.guidance.textureless_text_extra}"
                ], [self.cfg.guidance.negative_textureless])
                self.face_textureless_text_z = []
                for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                    # construct dir-encoded text
                    text = f"{self.cfg.guidance.textureless_text} of the headshot of {self.cfg.guidance.text_head}, {d} view, {self.cfg.guidance.textureless_text_extra}"

                    negative_text = f"{self.cfg.guidance.negative_textureless}"

                    # explicit negative dir-encoded text
                    text_z = self.guidance.get_text_embeds([text], [negative_text])
                    self.face_textureless_text_z.append(text_z)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()    # write immediately to file

    ### ------------------------------

    def train_step(self, data, verbose=False):

        if self.cfg.stage == 'geometry':
            shading = 'normal'
            ambient_ratio = 0.1
        elif self.cfg.stage == 'texture':
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            print(f"Unknown stage: {self.cfg.stage}")

        loss = 0
        step_mesh = None

        loss_dict = {}

        mvp = data['mvp']
        poses = data['poses']
        H, W = data['H'], data['W']

        rays = get_rays(data['poses'], data['intrinsics'], H, W, -1)
        rays_o = rays['rays_o']    # [B, N, 3]
        rays_d = rays['rays_d']    # [B, N, 3]

        outputs = self.model(
            rays_o,
            rays_d,
            mvp,
            H,
            W,
            poses=poses,
            ambient_ratio=ambient_ratio,
            shading=shading,
            return_openpose_map=self.render_openpose_training,
            global_step=self.global_step,
            can_pose=data['can_pose'],
            mesh=step_mesh
        )
        # [1, 3, H, W]
        pred_rgb = outputs['image'].reshape(1, H, W, 3).permute(0, 3, 1, 2).contiguous()
        pred_depth = outputs['depth'].reshape(1, H, W)

        if step_mesh is None:
            step_mesh = outputs['mesh']

        # text embeddings
        if self.cfg.guidance.use_view_prompt:
            dirs = data['dir']    # [B,]
            is_face = data['is_face']
            if (self.cfg.guidance.normal_text
                is not None) and len(self.cfg.guidance.normal_text) > 0 and shading == 'normal':
                if is_face:
                    text_z_novd = self.face_normal_text_z_novd
                    text_z = self.face_normal_text_z[dirs]
                else:
                    text_z_novd = self.normal_text_z_novd
                    text_z = self.normal_text_z[dirs]
            elif (self.cfg.guidance.textureless_text is not None) and len(
                self.cfg.guidance.textureless_text
            ) > 0 and shading == 'textureless':
                if is_face:
                    text_z_novd = self.face_textureless_text_z_novd
                    text_z = self.face_textureless_text_z[dirs]
                else:
                    text_z_novd = self.textureless_text_z_novd
                    text_z = self.textureless_text_z[dirs]
            else:
                if is_face:
                    text_z_novd = self.face_text_z_novd
                    text_z = self.face_text_z[dirs]
                else:
                    text_z_novd = self.text_z_novd
                    text_z = self.text_z[dirs]
        else:
            text_z = self.text_z
            text_z_novd = self.text_z

        guidance_loss = self.guidance.train_step(
            text_z,
            pred_rgb,
            guidance_scale=self.cfg.guidance.guidance_scale,
            controlnet_hint=pred_rgb if (self.cfg.stage == 'texture') and
            (self.cfg.guidance.controlnet != None) else None,
            poses=data['poses'],
            text_embedding_novd=text_z_novd,
            is_face=is_face,
            cur_epoch=self.epoch,
            stage=self.stage,
        )

        loss += guidance_loss
        loss_dict['guidance_loss'] = f"{guidance_loss.item():.5f}"

        if self.cfg.stage == 'geometry':

            # smoothness regularizer
            mesh = outputs['mesh']
            lap_loss = 0
            eikonal_loss = 0

            if self.cfg.train.lambda_lap > 0 and self.global_step >= self.subdiv_steps[0]:
                lap_loss = laplacian_loss(mesh.v, mesh.f.long())
                loss += self.cfg.train.lambda_lap * lap_loss
                loss_dict['lap_loss'] = f"{lap_loss.item() * self.cfg.train.lambda_lap:.5f}"
            if self.cfg.train.lambda_eik > 0:
                eikonal_loss = (outputs['geo_reg_loss'].norm(dim=-1) - 1).abs().mean()
                loss += self.cfg.train.lambda_eik * eikonal_loss
                loss_dict['depth_loss'] = f"{eikonal_loss.item() * self.cfg.train.lambda_eik:.5f}"

        if verbose:
            pprint.pprint(loss_dict)

        return pred_rgb, pred_depth, loss

    def eval_step(self, data, no_resize=True):

        is_face = data['is_face']
        mvp = data['mvp']
        if no_resize and not is_face:
            mvp = mvp @ self.model.mesh.resize_matrix_inv
        poses = data['poses']
        H, W = data['H'], data['W']

        rays = get_rays(data['poses'], data['intrinsics'], H, W, -1)
        rays_o = rays['rays_o']    # [B, N, 3]
        rays_d = rays['rays_d']    # [B, N, 3]

        if self.cfg.train.normal_sample_ratio >= 1.:
            shading = 'normal'
            ambient_ratio = 0.1
        elif self.cfg.train.textureless_sample_ratio >= 1.:
            shading = 'textureless'
            ambient_ratio = 0.1
        else:
            shading = data['shading'] if 'shading' in data else 'albedo'
            ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
            
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model(
            rays_o,
            rays_d,
            mvp,
            H,
            W,
            poses=poses,
            light_d=light_d,
            ambient_ratio=ambient_ratio,
            shading=shading,
            global_step=self.global_step
        )
        pred_rgb = outputs['image'].reshape(1, H, W, 3)
        pred_depth = outputs['depth'].reshape(1, H, W)
        
        outputs_normal = self.model(
            rays_o,
            rays_d,
            mvp,
            H,
            W,
            poses=poses,
            light_d=light_d,
            ambient_ratio=0.1,
            shading='normal',
            global_step=self.global_step
        )
        pred_norm = outputs_normal['image'].reshape(1, H, W, 3)
        pred_alpha = outputs['alpha'].reshape(1, H, W, 1)
        # dummy
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)

        return pred_rgb, pred_depth, pred_norm, pred_alpha, loss

    def test_step(
        self, data, bg_color=None, perturb=False, mesh=None, can_pose=False, no_resize=False
    ):
        is_face = data['is_face']
        mvp = data['mvp']
        if no_resize and not is_face:
            mvp = mvp @ self.model.mesh.resize_matrix_inv
        poses = data['poses']
        H, W = data['H'], data['W']

        rays = get_rays(data['poses'], data['intrinsics'], H, W, -1)
        rays_o = rays['rays_o']    # [B, N, 3]
        rays_d = rays['rays_d']    # [B, N, 3]

        if self.cfg.train.normal_sample_ratio >= 1:
            shading = 'normal'
            ambient_ratio = 0.1
        elif self.cfg.train.textureless_sample_ratio >= 1:
            shading = 'textureless'
            ambient_ratio = 0.1
        else:
            shading = data['shading'] if 'shading' in data else 'albedo'
            ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model(
            rays_o,
            rays_d,
            mvp,
            H,
            W,
            poses=poses,
            light_d=light_d,
            ambient_ratio=ambient_ratio,
            shading=shading,
            return_openpose_map=self.render_openpose,
            global_step=self.global_step,
            mesh=mesh,
            can_pose=can_pose
        )

        outputs_normal = self.model(
            rays_o,
            rays_d,
            mvp,
            H,
            W,
            poses=poses,
            light_d=light_d,
            ambient_ratio=0.1,
            shading='normal',
            global_step=self.global_step,
            mesh=mesh,
            can_pose=can_pose
        )
        pred_norm = outputs_normal['image'].reshape(1, H, W, 3)    #[:, :, W//4: W//4 + W//2]

        pred_rgb = outputs['image'].reshape(1, H, W, 3)    #[:, :, W//4: W//4 + W//2]
        pred_depth = outputs['depth'].reshape(1, H, W)    #[:, :, W//4: W//4 + W//2]
        pred_alpha = outputs['alpha'].reshape(1, H, W, 1)    #[:, :, W//4: W//4 + W//2]
        pred_mesh = outputs.get('mesh', None)
        if self.render_openpose:
            openpose_map = outputs['openpose_map'].reshape(1, H, W, 3)    #[:, :, W//4: W//4 + W//2]
        else:
            openpose_map = None

        return pred_rgb, pred_depth, pred_norm, pred_alpha, openpose_map, mesh

    def save_mesh(self, save_path=None):

        name = f'{self.cfg.sub_name}_{self.cfg.stage}'

        if save_path is None:
            save_path = os.path.join(self.cfg.exp_root, 'obj')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path, name=name, export_uv=self.cfg.test.save_uv)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):

        assert self.text_z is not None, 'Training must provide a text prompt!'

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()

        if self.epoch % self.eval_interval == 0:
            self.evaluate_one_epoch(valid_loader)
            self.save_checkpoint(full=False, best=True)
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)
            torch.cuda.empty_cache()
            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(
        self,
        loader,
        save_path=None,
        name=None,
        write_video=True,
        can_pose=False,
        write_image=True
    ):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'visualize')

        if name is None:
            name = f'{self.workspace.split("/")[-1]}_ep{self.epoch:04d}'
            if can_pose:
                name = name + '_can_pose'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        self.model.eval()

        if write_video:
            all_preds = []
            # all_preds_depth = []
            all_preds_norm = []
            all_openpose_map = []

        # save front image for comparison
        if "PuzzleIOI" in save_path:
            from shutil import copyfile
            start_idx = save_path.split("/").index("puzzle_capture")
            subject, outfit = save_path.split("/")[start_idx+1:start_idx+3]
            src_front_image = os.path.join(
                "data", "PuzzleIOI", "fitting", subject, outfit, "images", "07_C.jpg"
            )
            tgt_front_image = os.path.join(save_path, "full-body.jpg")
            copyfile(src_front_image, tgt_front_image)

            self.log(f"Copyfile {src_front_image} ==> {tgt_front_image}")

        with torch.no_grad():
            mesh = None
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_norm, preds_alpha, openpose_map, pred_mesh = self.test_step(
                        data, mesh=mesh, can_pose=can_pose, no_resize=not can_pose
                    )
                if mesh is None:
                    mesh = pred_mesh
                preds_alpha = preds_alpha[0].detach().cpu().numpy()

                pred = preds[0].detach().cpu().numpy()
                #pred = (pred * 255).astype(np.uint8)
                pred = ((pred * preds_alpha + (1 - preds_alpha)) * 255).astype(np.uint8)

                pred_norm = preds_norm[0].detach().cpu().numpy()
                pred_norm = ((pred_norm * preds_alpha + (1 - preds_alpha)) * 255).astype(np.uint8)

                # pred_depth = preds_depth[0].detach().cpu().numpy()
                # pred_depth = (pred_depth -
                #               pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                # pred_depth = (pred_depth * 255).astype(np.uint8)

                if self.render_openpose:

                    openpose_map = (openpose_map[0].detach().cpu().numpy() * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    # all_preds_depth.append(pred_depth)
                    all_preds_norm.append(pred_norm)
                    if self.render_openpose:
                        all_openpose_map.append(openpose_map)

                if write_image and i % 10 == 0:
                    if isinstance(preds_alpha, torch.Tensor):
                        preds_alpha = preds_alpha[0].detach().cpu().numpy()
                    preds_alpha = (preds_alpha * 255).astype(np.uint8)
                    pred = np.concatenate([pred, preds_alpha], axis=-1)
                    pred_norm = np.concatenate([pred_norm, preds_alpha], axis=-1)
                    cv2.imwrite(
                        os.path.join(save_path, f'{name}_{i:04d}_rgb.png'),
                        cv2.cvtColor(pred, cv2.COLOR_RGBA2BGRA)
                    )
                    # cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)
                    cv2.imwrite(
                        os.path.join(save_path, f'{name}_{i:04d}_norm.png'),
                        cv2.cvtColor(pred_norm, cv2.COLOR_RGBA2BGRA)
                    )

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            # all_preds_depth = np.stack(all_preds_depth, axis=0)
            all_preds_norm = np.stack(all_preds_norm, axis=0)
            # all_preds_norm[100:] = 255 - all_preds_norm[100:]

            frame_size = 100
            all_preds_full = np.concatenate([
                np.concatenate([all_preds[:frame_size], all_preds_norm[:frame_size]], axis=2),
                np.concatenate([all_preds[frame_size:], all_preds_norm[frame_size:]], axis=2),
            ],
                                            axis=1)
            if self.cfg.stage == 'texture':
                imageio.mimwrite(
                    os.path.join(save_path, f'{name}_rgb.mp4'),
                    all_preds[:frame_size],
                    fps=25,
                    quality=8,
                    macro_block_size=1
                )
            imageio.mimwrite(
                os.path.join(save_path, f'{name}_full.mp4'),
                all_preds_full,
                fps=25,
                quality=8,
                macro_block_size=1
            )
            imageio.mimwrite(
                os.path.join(save_path, f'{name}_head.mp4'),
                all_preds_norm[frame_size:],
                fps=25,
                quality=8,
                macro_block_size=1
            )
            # imageio.mimwrite(
            #     os.path.join(save_path, f'{name}_depth.mp4'),
            #     all_preds_depth[:100],
            #     fps=25,
            #     quality=8,
            #     macro_block_size=1
            # )
            imageio.mimwrite(
                os.path.join(save_path, f'{name}_norm.mp4'),
                all_preds_norm[:frame_size],
                fps=25,
                quality=8,
                macro_block_size=1
            )
            if self.render_openpose:
                all_openpose_map = np.stack(all_openpose_map, axis=0)
                imageio.mimwrite(
                    os.path.join(save_path, f'{name}_openpose.mp4'),
                    all_openpose_map,
                    fps=25,
                    quality=8,
                    macro_block_size=1
                )

        self.log(f"==> Finished Test.")

    def train_one_epoch(self, loader):
        self.log(
            f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ..."
        )

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format=
                '{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

        self.local_step = 0

        for data in loader:

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_depths, loss = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar(
                        "train/lr", self.optimizer.param_groups[0]['lr'], self.global_step
                    )

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format=
                '{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

        with torch.no_grad():
            self.local_step = 0

            save_geo_lst = []
            save_tex_lst = []

            save_path_geo = os.path.join(self.workspace, 'validation', f'{name}_geo.png')
            save_path_tex = os.path.join(self.workspace, 'validation', f'{name}_tex.png')

            os.makedirs(os.path.dirname(save_path_geo), exist_ok=True)
            os.makedirs(os.path.dirname(save_path_tex), exist_ok=True)

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_normal, preds_alpha, loss = self.eval_step(data)

                # add alpha channel
                preds = preds * preds_alpha + (1 - preds_alpha)
                preds_normal = preds_normal * preds_alpha + (1 - preds_alpha)
                preds_depth = preds_depth * preds_alpha + (1 - preds_alpha)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss /= self.world_size

                    preds_list = [
                        torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)
                    ]    # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [
                        torch.zeros_like(preds_depth).to(self.device)
                        for _ in range(self.world_size)
                    ]    # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    preds_normal_list = [
                        torch.zeros_like(preds_normal).to(self.device)
                        for _ in range(self.world_size)
                    ]    # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_normal_list, preds_normal)
                    preds_normal = torch.cat(preds_normal_list, dim=0)
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth -
                                  pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    pred_normal = preds_normal[0].detach().cpu().numpy()
                    pred_normal = (pred_normal * 255).astype(np.uint8)

                    save_geo_lst.append(cv2.cvtColor(pred_normal, cv2.COLOR_RGB2BGR))
                    if self.stage == "texture":
                        save_tex_lst.append(cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)

            if len(save_geo_lst) % 2 == 0:
                col_size = len(save_geo_lst) // 2
                save_geo_arr = np.concatenate([
                    np.concatenate(save_geo_lst[:col_size], 1),
                    np.concatenate(save_geo_lst[col_size:], 1)
                ], 0)
                if self.stage == "texture":
                    save_tex_arr = np.concatenate([
                        np.concatenate(save_tex_lst[:col_size], 1),
                        np.concatenate(save_tex_lst[col_size:], 1)
                    ], 0)
            else:
                save_geo_arr = np.concatenate(save_geo_lst, 1)
                if self.stage == "texture":
                    save_tex_arr = np.concatenate(save_tex_lst, 1)

            cv2.imwrite(save_path_geo, save_geo_arr)
            if self.stage == "texture":
                cv2.imwrite(save_path_tex, save_tex_arr)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == 'min' else -result
                )    # if max mode, use -result
            else:
                self.stats["results"].append(
                    average_loss
                )    # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_pretrained(self, pretrained=None):
        if pretrained is None:
            return
        else:
            self.log("[INFO] loading pretrained model from {}".format(pretrained))
            checkpoint_dict = torch.load(pretrained, map_location=self.device)
            if 'model' in checkpoint_dict:
                checkpoint_dict = checkpoint_dict['model']
            if 'v_offsets' in checkpoint_dict:
                checkpoint_dict.pop('v_offsets')
            if 'vn_offsets' in checkpoint_dict:
                checkpoint_dict.pop('vn_offsets')
            if 'sdf' in checkpoint_dict:
                checkpoint_dict.pop('sdf')
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint_dict, strict=False
            )
            self.log("[INFO] loaded model.")
            if len(missing_keys) > 0:
                self.log(f"[WARN] missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                self.log(f"[WARN] unexpected keys: {unexpected_keys}")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict['model'], strict=False
        )
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
