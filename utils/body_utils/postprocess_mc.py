import argparse
import numpy as np
import torch
import trimesh
from glob import glob
from .lib.dataset.mesh_util import SMPLX, mesh_simplify, keep_largest, poisson
from scipy.spatial import cKDTree

# loading cfg file
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default="exp/demo/teaser/obj/")
parser.add_argument("--data_dir", type=str, default="exp/demo/teaser/obj/")

parser.add_argument("-n", "--name", type=str, default="")
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-t", "--type", type=str, default="smplx")
parser.add_argument("-f", "--face", action='store_true', default=False)
args = parser.parse_args()

smplx_container = SMPLX()
device = torch.device(f"cuda:{args.gpu}")

# load smplx and TeCH objs
smplx_path = glob(f"{args.data_dir}/smplx_*.obj")[0]
tech_path = f"{args.dir}/obj/{args.name}_geometry.obj"
smplx_obj = trimesh.load(smplx_path, maintain_orders=True, process=False)
tech_obj = trimesh.load(tech_path, maintain_orders=True, process=False)

smpl_tree = cKDTree(smplx_obj.vertices)
dist, idx = smpl_tree.query(tech_obj.vertices, k=5)

# remove hands from TeCH
tech_body = tech_obj.copy()
mano_mask = ~np.isin(idx[:, 0], smplx_container.smplx_mano_vid)
tech_body.update_faces(mano_mask[tech_obj.faces].all(axis=1))
tech_body.remove_unreferenced_vertices()
tech_body = keep_largest(tech_body)

# keep hands from smplx
smplx_hand = smplx_obj.copy()
smplx_hand.update_faces(
    smplx_container.smplx_mano_vertex_mask.numpy()[smplx_hand.faces].any(axis=1)
)
smplx_hand.remove_unreferenced_vertices()

# combine Tech's body and SMPL-X's hands
tech_new = sum([tech_body, smplx_hand])
tech_new_obj = poisson(tech_new, tech_path.replace("geometry", "geometry_final"), depth=10)
mesh_simplify(tech_path.replace("geometry", "geometry_final"))