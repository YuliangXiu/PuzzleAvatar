#import nvdiffrast.torch as dr
import json
import numpy as np
import trimesh
import torch
from utils.body_utils.lib import smplx
from utils.body_utils.lib.smplx.lbs import batch_rodrigues
from utils.body_utils.lib.dataset.mesh_util import SMPLX

smpl_path = "./data/PuzzleIOI/fitting/00145/apose/smplx/smplx.pkl"
vert_path = "./data/PuzzleIOI/fitting/00145/apose/smplx/smplx.obj"

smpl_param = np.load(smpl_path, allow_pickle=True)
for key in smpl_param.keys():
    smpl_param[key] = torch.as_tensor(smpl_param[key]).float()
verts = trimesh.load(vert_path).vertices

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
    flat_hand_mean=False,
    ext='pkl'
)

smplx_obj = smplx_model(**smpl_param)
    
smplx_verts = smplx_obj.vertices.detach()[0]
# smplx_joints = smplx_obj.joints.detach()[0].numpy()
# pelvis_y = 0.5 * (smplx_verts[:, 1].min() + smplx_verts[:, 1].max())
# smplx_verts[:, 1] -= pelvis_y
# smplx_joints[:, 1] -= pelvis_y
smplx_faces = smplx_model.faces


trimesh.Trimesh(smplx_verts, smplx_faces).export("smplx_infer.obj")
trimesh.Trimesh(verts, smplx_faces).export("smplx_gt.obj")
