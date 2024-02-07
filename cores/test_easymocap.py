#import nvdiffrast.torch as dr
import json
import numpy as np
import trimesh
import torch
from utils.body_utils.lib import smplx
from utils.body_utils.lib.smplx.lbs import batch_rodrigues
from utils.body_utils.lib.dataset.mesh_util import SMPLX

smpl_path = "./data/PuzzleIOI/fitting/00145/apose/output/smplx/smpl/000000.json"
vert_path = "./data/PuzzleIOI/fitting/00145/apose/output/smplx/vertices/000000.json"
smpl_param = json.load(open(smpl_path, "r"))[0]
for key in smpl_param.keys():
    smpl_param[key] = torch.as_tensor(np.array(smpl_param[key])).float()
verts = np.array(json.load(open(vert_path, "r"))[0]["vertices"])

smplx_container = SMPLX()
smplx_model = smplx.create(
    smplx_container.model_dir,
    model_type='smplx',
    gender="male",
    age="adult",
    use_face_contour=False,
    use_pca=True,
    num_betas=10,
    num_expression_coeffs=10,
    flat_hand_mean=True,
    ext='pkl'
)

smplx_obj = smplx_model(
    betas=smpl_param['shapes'],
    global_orient=smpl_param['poses'][:, :3],
    body_pose=smpl_param["poses"][:, 3:22 * 3],
    expression=smpl_param['expression'],
    jaw_pose=smpl_param['poses'][:, 22 * 3:23 * 3],
    leye_pose=smpl_param['poses'][:, 23 * 3:24 * 3],
    reye_pose=smpl_param['poses'][:, 24 * 3:25 * 3],
    left_hand_pose=smpl_param['poses'][:, 25 * 3:27 * 3],
    right_hand_pose=smpl_param['poses'][:, 27 * 3:],
    return_verts=True,
    return_full_pose=True,
    return_joint_transformation=True,
    return_vertex_transformation=True,
)
    
smplx_verts = smplx_obj.vertices.detach()[0]
# smplx_joints = smplx_obj.joints.detach()[0].numpy()
# pelvis_y = 0.5 * (smplx_verts[:, 1].min() + smplx_verts[:, 1].max())
# smplx_verts[:, 1] -= pelvis_y
# smplx_joints[:, 1] -= pelvis_y
smplx_faces = smplx_model.faces

smplx_verts = torch.matmul(smplx_verts,
                           batch_rodrigues(smpl_param['Rh']).transpose(1, 2)) + smpl_param['Th']
trimesh.Trimesh(smplx_verts[0], smplx_faces).export("smplx_infer.obj")
trimesh.Trimesh(verts, smplx_faces).export("smplx_gt.obj")
