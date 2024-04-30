import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .camera_utils import *
from mvdream_diffusers.mv_unet import get_camera_condition

def get_view_direction(thetas, phis, overhead, front, phi_diff=0):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    # TODO
    phis = phis.clone()
    phis += phi_diff / 180 * np.pi
    phis[phis >= 2 * np.pi] -= np.pi * 2
    phis[phis < 0] += np.pi * 2
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    res[(phis < front)] = 0
    res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi + front))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def rand_poses(
    size,
    device,
    radius_range=[1, 1.5],
    theta_range=[0, 120],
    phi_range=[0, 360],
    height_range=[0, 0],
    return_dirs=False,
    angle_overhead=30,
    angle_front=60,
    jitter=False,
    uniform_sphere_rate=0.5,
    phi_diff=0,
    center_offset=0.,
    num_frame=1,
):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size*num_frame, 4, 4]
    '''
    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size, device=device) - 0.5) * 2.0,
                torch.rand(size, device=device),
                (torch.rand(size, device=device) - 0.5) * 2.0,
            ],
                        dim=-1),
            p=2,
            dim=1
        )
        thetas = torch.acos(unit_centers[:, 1])
        phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
        phis[phis < 0] += 2 * np.pi

        thetas, phis = expand_to_4poses(thetas, phis, num_frames=num_frame)
        centers = unit_centers * radius.unsqueeze(-1)
        centers = centers + centers.new_tensor(center_offset)
    else:
        heights = torch.rand(size,
                             device=device) * (height_range[1] - height_range[0]) + height_range[0]
        thetas = torch.rand(size,
                            device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        thetas, phis = expand_to_4poses(thetas, phis, num_frames=num_frame)
        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas) + heights,
            radius * torch.sin(thetas) * torch.cos(phis),
        ],
                              dim=-1)    # [B, 3]
        centers = centers + centers.new_tensor(center_offset)

    targets = torch.zeros_like(centers) + centers.new_tensor(center_offset)
    targets[:, 1] += heights

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size*num_frame, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front, phi_diff=phi_diff)
    else:
        dirs = None

    return poses, dirs, radius, thetas, phis


def expand_to_4poses(thetas, phis, num_frames=4):
    device = thetas.device
    thetas = thetas.repeat(num_frames)
    # (0, top, pi bottom) --> (-pi/2 top, pi/2 bottoem)
    
    # maybe need to -= phi_diff
    # phis = phis + phi_diff / 180 * np.pi
    phis = phis.repeat(num_frames)
    delta_phis = torch.linspace(0, 2 * np.pi, num_frames+1)[:num_frames].to(device)
    phis = phis + delta_phis

    return thetas, phis


def get_mv_pose(thetas, phis, radius, phi_diff=0, blender_coord=True):
    """

    :param thetas: (F, ) in radius
    :param phis: (F, ) -- azimuth??
    :param radius: (1, )
    :param num_frames: _description_, defaults to 4
    :return poses: (num_frames, 4, 4)
    """
    device = thetas.device
    num_frames = thetas.shape[0]
    # thetas = thetas.repeat(num_frames)
    # (0, top, pi bottom) --> (-pi/2 top, pi/2 bottoem)
    thetas = thetas - np.pi / 2
    thetas = -thetas
    # thetas = thetas - np.pi

    # maybe need to -= phi_diff
    phis = np.pi - phis - np.pi / 2
    # phis = phis + np.pi / 2
    # phis = np.pi

    if isinstance(radius, float):
        radius = torch.tensor([radius] * num_frames, device=device)
    radius = radius.expand([num_frames])
    pose = get_camera_condition(phis, thetas, radius, blender_coord=blender_coord)  # # kiui's elevation is negated, [4, 4] (4, 4)
    return pose

def circle_poses(
    device,
    radius=1.25,
    theta=60,
    phi=0,
    return_dirs=False,
    angle_overhead=30,
    angle_front=60,
    phi_diff=0,
    height=0,
    center_offset=0.,
    num_frame=1,
):

    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    thetas, phis = expand_to_4poses(thetas, phis, num_frames=num_frame)
    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas) + height,
        radius * torch.sin(thetas) * torch.cos(phis),
    ],
                          dim=-1)    # [B, 3]

    centers = centers + centers.new_tensor(center_offset)

    # lookat
    targets = torch.zeros_like(centers) + centers.new_tensor(center_offset)
    targets[:, 1] += height
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(num_frame, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front, phi_diff=phi_diff)
    else:
        dirs = None

    return poses, dirs, radius, thetas, phis


class ViewDataset:
    def __init__(
        self,
        cfg,
        device,
        type='train',
        H=256,
        W=256,
        size=100,
        render_head=False,
        render_canpose=False
    ):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.type = type    # train, val, test

        self.H = H
        self.W = W
        self.size = size
        self.num_frames = size
        if render_head:
            self.size = self.num_frames * 2

        self.training = self.type in ['train', 'all']

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.near = self.cfg.model.min_near
        self.far = 1000    # infinite

        self.aspect = self.W / self.H
        self.global_step = 0

    def get_phi_range(self):
        return self.cfg.train.phi_range

    def update_global_step(self, global_step):
        self.global_step = global_step

    def collate(self, index):

        B = len(index)    # always 1
        is_face = False
        can_pose = False
        if self.training:
            if self.cfg.data.can_pose_folder is not None:
                can_pose = random.random() < self.cfg.train.can_pose_sample_ratio
            # random pose on the fly
            if random.random() < self.cfg.train.face_sample_ratio:
                poses, dirs, radius,thetas, phis = rand_poses(
                    B,
                    self.device,
                    radius_range=self.cfg.train.face_radius_range,
                    return_dirs=self.cfg.guidance.use_view_prompt,
                    angle_overhead=self.cfg.train.angle_overhead,
                    angle_front=self.cfg.train.angle_front,
                    jitter=False,
                    uniform_sphere_rate=0.,
                    phi_diff=self.cfg.train.face_phi_diff,
                    theta_range=self.cfg.train.face_theta_range,
                    phi_range=self.cfg.train.face_phi_range,
                    height_range=self.cfg.train.face_height_range,
                    center_offset=np.array(
                        self.cfg.train.head_position if not can_pose else self.cfg.train.
                        canpose_head_position
                    ),
                    num_frame=4,
                )
                is_face = True
            else:
                poses, dirs, radius,thetas, phis = rand_poses(
                    B,
                    self.device,
                    radius_range=self.cfg.train.radius_range,
                    return_dirs=self.cfg.guidance.use_view_prompt,
                    angle_overhead=self.cfg.train.angle_overhead,
                    angle_front=self.cfg.train.angle_front,
                    jitter=self.cfg.train.jitter_pose,
                    uniform_sphere_rate=0.,
                    phi_diff=self.cfg.train.phi_diff,
                    theta_range=self.cfg.train.theta_range,
                    phi_range=self.get_phi_range(),
                    height_range=self.cfg.train.height_range,
                    num_frame=4,
                )
            # random focal
            fov = random.random() * (self.cfg.train.fovy_range[1] -
                                     self.cfg.train.fovy_range[0]) + self.cfg.train.fovy_range[0]
        else:
            # circle pose
            phi = ((index[0] / self.num_frames) * 360) % 360
            if index[0] < self.num_frames:
                poses, dirs, radius, thetas, phis = circle_poses(
                    self.device,
                    radius=self.cfg.train.radius_range[1] * 0.9,
                    theta=90,
                    phi=phi,
                    return_dirs=self.cfg.guidance.use_view_prompt,
                    angle_overhead=self.cfg.train.angle_overhead,
                    angle_front=self.cfg.train.angle_front,
                    phi_diff=self.cfg.train.phi_diff,
                    num_frame=4,
                )
            else:
                is_face = True
                poses, dirs, radius,thetas, phis = circle_poses(
                    self.device,
                    radius=self.cfg.train.face_radius_range[0],
                    height=self.cfg.train.face_height_range[0],
                    theta=90,
                    phi=phi,
                    return_dirs=self.cfg.guidance.use_view_prompt,
                    angle_overhead=self.cfg.train.angle_overhead,
                    angle_front=self.cfg.train.angle_front,
                    phi_diff=self.cfg.train.phi_diff,
                    center_offset=np.array(self.cfg.train.head_position),
                    num_frame=4,
                )

            # fixed focal
            fov = (self.cfg.train.fovy_range[1] + self.cfg.train.fovy_range[0]) / 2

        cameras = get_mv_pose(thetas, phis, radius, phi_diff=self.cfg.train.phi_diff,
                               blender_coord=True)
        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])
        F = cameras.shape[0]
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(F, 1)

        projection = torch.tensor([[2 * focal / self.W, 0, 0, 0], [0, -2 * focal / self.H, 0, 0],
                                   [
                                       0, 0, -(self.far + self.near) / (self.far - self.near),
                                       -(2 * self.far * self.near) / (self.far - self.near)
                                   ], [0, 0, -1, 0]],
                                  dtype=torch.float32,
                                  device=self.device).unsqueeze(0)    # yapf: disabl
        mvp = projection @ torch.inverse(poses.cpu()).to(self.device)
        if not self.training:
            if is_face or can_pose:
                mvp = projection @ torch.inverse(poses.cpu()).to(self.device)
            else:
                mvp = torch.inverse(poses.cpu()).to(self.device)
                mvp[0, 2, 3] = 0.
                TO_WORLD = np.eye(
                    4,
                    dtype=np.float32,
                )
                TO_WORLD[2, 2] = -1
                TO_WORLD[1, 1] = -1
                TO_WORLD = mvp.new_tensor(TO_WORLD)
                mvp = TO_WORLD @ mvp

        data = {
            'H': self.H,
            'W': self.W,
            # 'mvp': mvp[0],    # [4, 4]
            'mvp': mvp,    # [4, 4]
            'poses': poses,    # [1, 4, 4]
            'intrinsics': intrinsics,
            'dir': dirs,
            'camera': cameras,  # cameras for mvfusion [F, 4, 4]
            'near_far': [self.near, self.far],
            'is_face': is_face,
            'radius': radius,
            'can_pose': can_pose
        }

        return data

    def dataloader(self):
        loader = DataLoader(
            list(range(self.size)),
            batch_size=1,
            collate_fn=self.collate,
            shuffle=self.training,
            num_workers=0
        )
        return loader
