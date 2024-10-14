#import nvdiffrast.torch as dr
import argparse
import json
import os

import torch
import trimesh
import logging
from yacs.config import CfgNode as CN

from cores.lib.provider import ViewDataset
from cores.lib.renderer import Renderer
from cores.lib.trainer import *
from utils.body_utils.lib import smplx
from utils.body_utils.lib.dataset.mesh_util import SMPLX

torch.set_float32_matmul_precision('high')
torch._dynamo.config.verbose = False
logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)


def load_config(path, default_path=None):
    cfg = CN(new_allowed=True)
    if default_path is not None:
        cfg.merge_from_file(default_path)
    cfg.merge_from_file(path)

    return cfg


def dict_to_prompt(d, use_shape=False, multi_mode=False, so_name=None):

    classes = list(d.keys())
    gender = "man" if d['gender'] == "male" else "woman"
    # gender = "man"
    classes.remove("gender")

    prompt_head = f"a high-resolution DSLR colored image of a {gender}"

    placeholders = []

    if not multi_mode:

        for key in ["eyeglasses", "sunglasses", "glasses"]:
            if key in classes:
                classes.remove(key)

        tokens = [f"<asset{i}>" for i in range(len(classes))]
        descs = [d[key] for key in classes]
        placeholders = tokens
    else:
        from glob import glob
        dict_path = glob(f"./results/multi/*{so_name}*/multi_dict.npy")[0]
        multi_dict = np.load(dict_path, allow_pickle=True).item()
        for key in multi_dict.keys():
            placeholders += multi_dict[key]["tokens"]

        if False:
            # create the novel ID
            # new_multi_dict = {
            #     'human_yuliang': {
            #         'classes': ['face', 'haircut', 'dress', 'shoes'],
            #         'tokens': ['<asset0>', '<asset1>', '<asset7>', '<asset4>'],
            #         'descs': ['oval', 'short', 'sleeveless midi', 'sneakers']
            #     }, 'human_kexin': {
            #         'classes': ['face', 'haircut', 'jacket', 'pants', 'shoes'],
            #         'tokens': ['<asset0>', '<asset1>', '<asset12>', '<asset13>', '<asset4>'],
            #         'descs': ['oval', 'short', 'long hooded', 'ankle-length', 'sneakers']
            #     }, 'human_yifei': {
            #         'classes': ['face', 'haircut', 'shirt', 'pants', 'shoes'],
            #         'tokens': ['<asset9>', '<asset10>', '<asset2>', '<asset13>', '<asset14>'],
            #         'descs': ['oval', 'medium-length', 'crew neck and short', 'ankle-length', 'open sandals']
            #     }
            # }

            new_multi_dict = {
                'human_yuliang': {
                    'classes': ['face', 'haircut', 'shirt', 'pants', 'shoes'], 'tokens':
                    ['<asset0>', '<asset1>', '<asset7>', '<asset8>',
                     '<asset9>'], 'descs': ['', '', 'short', '', '']
                }, 'human_yamei': {
                    'classes': ['face', 'haircut', 'shirt', 'pants', 'shoes'], 'tokens':
                    ['<asset0>', '<asset1>', '<asset7>', '<asset3>', '<asset4>'], 'descs':
                    ['oval', 'long ponytail', 'long loose checked', 'fitted', 'casual sneakers']
                }, 'human_zhen': {
                    'classes': ['face', 'haircut', 'shirt', 'pants', 'shoes'], 'tokens':
                    ['<asset0>', '<asset1>', '<asset12>', '<asset13>', '<asset14>'], 'descs':
                    ['round', 'short', 'crew-neck t-shirt', 'fitted jeans', 'casual sneakers']
                }
            }

            multi_dict = new_multi_dict

        so_data = multi_dict[so_name]
        classes = so_data["classes"]
        tokens = so_data["tokens"]
        descs = so_data["descs"]

    facial_classes = ['face', 'haircut', 'hair']
    with_classes = [cls for cls in classes if cls in facial_classes]
    wear_classes = [cls for cls in classes if cls not in facial_classes]

    prompt = f"{prompt_head}, "
    with_prompt = f"{prompt_head}, "

    for class_token in with_classes:
        idx = classes.index(class_token)

        if len(wear_classes) == 0 and with_classes.index(class_token) == len(with_classes) - 1:
            ending = "."
        else:
            ending = ", "

        if use_shape:
            prompt += f"{tokens[idx]} {descs[idx]} {class_token}{ending}"
            with_prompt += f"{tokens[idx]} {descs[idx]} {class_token}{ending}"
        else:
            prompt += f"{tokens[idx]} {class_token}{ending}"
            with_prompt += f"{tokens[idx]} {class_token}{ending}"

    if len(wear_classes) > 0:
        prompt += "wearing "

        for class_token in wear_classes:

            idx = classes.index(class_token)

            if wear_classes.index(class_token) < len(wear_classes) - 2:
                ending = ", "
            elif wear_classes.index(class_token) == len(wear_classes) - 2:
                ending = ", and "
            else:
                ending = "."
            if use_shape:
                prompt += f"{tokens[idx]} {descs[idx]} {class_token}{ending}"
            else:
                prompt += f"{tokens[idx]} {class_token}{ending}"

    return d['gender'], prompt, prompt, placeholders


# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="config file")
    parser.add_argument('--exp_dir', type=str, required=True, help="experiment dir")
    parser.add_argument('--sub_name', type=str, required=True, help="subject name")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--use_peft', type=str, default="none", help="none/lora/boft")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--use_shape_description', action="store_true")
    parser.add_argument('--use_sds', action="store_true")

    opt = parser.parse_args()
    cfg = load_config(opt.config, default_path="configs/default.yaml")

    cfg.test.test = opt.test
    cfg.workspace = os.path.join(opt.exp_dir, cfg.stage)
    cfg.exp_root = opt.exp_dir
    cfg.sub_name = opt.sub_name
    cfg.use_peft = opt.use_peft
    cfg.use_sds = opt.use_sds

    multi_mode = True if "_subjects" in opt.exp_dir else False

    if cfg.guidance.use_dreambooth:
        if multi_mode:
            if "human" in cfg.exp_root:
                subjects_lst = np.loadtxt("./clusters/lst/group_char.txt", dtype=str)
            else:
                subjects_lst = np.loadtxt("./clusters/lst/group_5.txt", dtype=str)
            if subjects_lst.shape == ():
                real_hf_key = subjects_lst.item()
            else:
                real_hf_key = [name for name in subjects_lst if opt.sub_name in name][0]
            cfg.guidance.hf_key = os.path.join("./results/multi", real_hf_key)
        else:
            cfg.guidance.hf_key = opt.exp_dir

    if cfg.guidance.text is None:
        json_path = os.path.join(
            "./data", "/".join(opt.exp_dir.split("/")[1:]), "gpt4v_simple.json"
        )

        with open(json_path, 'r') as f:
            gpt4v_response = json.load(f)
            gender, cfg.guidance.text, cfg.guidance.text_head, placeholders = dict_to_prompt(
                gpt4v_response, opt.use_shape_description, multi_mode, cfg.sub_name
            )

            print(f"Using prompt: {cfg.guidance.text}")
            print(f"Using head prompt: {cfg.guidance.text_head}")

    # create smplx base meshes wrt gender

    smplx_path = os.path.join("./data", "/".join(opt.exp_dir.split("/")[1:]), f"smplx_{gender}.obj")
    keypoint_path = os.path.join(
        "./data", "/".join(opt.exp_dir.split("/")[1:]), f"smplx_{gender}.npy"
    )

    cfg.data.last_model = smplx_path
    cfg.data.keypoints_path = keypoint_path

    if cfg.data.load_result_mesh:
        cfg.data.last_model = os.path.join(opt.exp_dir, 'obj', f"{opt.sub_name}_geometry_final.obj")

    if not os.path.exists(smplx_path) or not os.path.exists(keypoint_path):

        use_puzzle = True if "PuzzleIOI" in opt.exp_dir else False

        smplx_container = SMPLX()
        smplx_model = smplx.create(
            smplx_container.model_dir,
            model_type='smplx',
            gender=gender,
            age="adult",
            use_face_contour=False,
            use_pca=True,
            num_betas=10,
            num_expression_coeffs=10,
            flat_hand_mean=not use_puzzle,
            ext='pkl'
        )

        if use_puzzle:

            smpl_param_path = os.path.join(
                "./data", "/".join(opt.exp_dir.split("/")[2:]).replace("puzzle_cam", "fitting"),
                "smplx/smplx.pkl"
            )
            print(f"SMPL pkl path: {smpl_param_path}")

            smpl_param = np.load(smpl_param_path, allow_pickle=True)

            for key in ["gender", "keypoints_3d"]:
                smpl_param.pop(key, None)

            for key in smpl_param.keys():
                smpl_param[key] = torch.as_tensor(smpl_param[key]).float()

            smplx_obj = smplx_model(**smpl_param)

        else:

            smplx_obj = smplx_model(pose_type="a-pose")

        smplx_verts = smplx_obj.vertices.detach()[0].numpy()
        smplx_joints = smplx_obj.joints.detach()[0].numpy()
        pelvis_y = 0.5 * (smplx_verts[:, 1].min() + smplx_verts[:, 1].max())
        smplx_verts[:, 1] -= pelvis_y
        smplx_joints[:, 1] -= pelvis_y
        smplx_faces = smplx_model.faces
        trimesh.Trimesh(smplx_verts, smplx_faces).export(smplx_path)
        np.save(keypoint_path, {"joints": smplx_joints, "pelvis_y": pelvis_y}, allow_pickle=True)

    seed_everything(opt.seed)
    model = Renderer(cfg)

    if model.keypoints is not None:
        # SMPL-X head joint is 15
        cfg.train.head_position = model.keypoints[0][15].cpu().numpy().tolist()
    else:
        cfg.train.head_position = np.array([0., 0.4, 0.], dtype=np.float32).tolist()
    cfg.train.canpose_head_position = np.array([0., 0.4, 0.], dtype=np.float32).tolist()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.test.test:
        guidance = None    # no need to load guidance model at test
        trainer = Trainer(
            'df',
            cfg,
            model,
            guidance,
            device=device,
            workspace=cfg.workspace,
            fp16=cfg.fp16,
            use_checkpoint=cfg.train.ckpt,
            pretrained=cfg.train.pretrained
        )

        if not cfg.test.not_test_video:
            test_loader = ViewDataset(
                cfg,
                device=device,
                type='test',
                H=cfg.test.H,
                W=cfg.test.W,
                size=100,
                render_head=True
            ).dataloader()
            trainer.test(test_loader, write_image=cfg.test.write_image)
            if cfg.data.can_pose_folder is not None:
                trainer.test(test_loader, write_image=cfg.test.write_image, can_pose=True)
        if cfg.test.save_mesh:
            trainer.save_mesh()
    else:

        train_loader = ViewDataset(
            cfg, device=device, type='train', H=cfg.train.h, W=cfg.train.w, size=100
        ).dataloader()

        params_list = list()
        if cfg.guidance.type == 'stable-diffusion':
            from cores.lib.guidance import StableDiffusion
            guidance = StableDiffusion(
                device,
                placeholders,
                cfg.use_peft,
                cfg.guidance.sd_version,
                cfg.guidance.hf_key,
                cfg.guidance.step_range,
                cfg.train.tet_subdiv_steps,
                cfg.train.iters,
                controlnet=cfg.guidance.controlnet,
                lora=cfg.guidance.lora,
                cfg=cfg,
                head_hf_key=cfg.guidance.head_hf_key,
                use_sds=cfg.use_sds,
            )
            for p in guidance.parameters():
                p.requires_grad = False
        else:
            raise NotImplementedError(f'--guidance {cfg.guidance.type} is not implemented.')

        if cfg.train.optim == 'adan':
            from cores.lib.optimizer import Adan

            # Adan usually requires a larger LR
            params_list.extend(model.get_params(5 * cfg.train.lr))
            optimizer = lambda model: Adan(
                params_list, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False
            )
        else:    # adam
            params_list.extend(model.get_params(cfg.train.lr))
            optimizer = lambda model: torch.optim.AdamW(params_list, betas=(0.9, 0.99), eps=1e-15)

        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1**min(iter / cfg.train.iters, 1)
        )

        trainer = Trainer(
            'df',
            cfg,
            model,
            guidance,
            device=device,
            workspace=cfg.workspace,
            optimizer=optimizer,
            ema_decay=None,
            fp16=cfg.train.fp16,
            lr_scheduler=scheduler,
            use_checkpoint=cfg.train.ckpt,
            eval_interval=cfg.train.eval_interval,
            scheduler_update_every_step=True,
            pretrained=cfg.train.pretrained,
            subdiv_steps=cfg.train.tet_subdiv_steps,
        )

        valid_loader = ViewDataset(
            cfg, device=device, type='val', H=cfg.test.H, W=cfg.test.W, size=5, render_head=True
        ).dataloader()

        max_epoch = np.ceil(cfg.train.iters / len(train_loader)).astype(np.int32)
        if cfg.profile:
            import cProfile
            with cProfile.Profile() as pr:
                trainer.train(train_loader, valid_loader, max_epoch)
                pr.dump_stats(os.path.join(cfg.workspace, 'profile.dmp'))
                pr.print_stats()
        else:
            trainer.train(train_loader, valid_loader, max_epoch)

        test_loader = ViewDataset(
            cfg, device=device, type='test', H=cfg.test.H, W=cfg.test.W, size=100, render_head=True
        ).dataloader()
        trainer.test(test_loader, write_image=cfg.test.write_image)

        if cfg.test.save_mesh:
            trainer.save_mesh()
