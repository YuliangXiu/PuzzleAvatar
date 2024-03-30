import argparse
import os, sys
import cv2
from glob import glob
import os.path as osp
import numpy as np
import trimesh
import random
import math
import random
from tqdm import tqdm

# multi-thread
from functools import partial
from multiprocessing import Pool
import multiprocessing as mp

# to remove warning from numba
# "Numba: Attempted to fork from a non-main thread, the TBB library may be in an invalid state in the child process.""
import numba

numba.config.THREADING_LAYER = 'workqueue'

sys.path.append(os.path.join(os.getcwd()))


def render_subject(subject, dataset, save_folder, rotation, size, render_types, egl):

    initialize_GL_context(width=size, height=size, egl=egl)

    scale = 100.0
    up_axis = 1
    
    mesh_file = glob(f"{subject}/*.obj")[0]
    tex_file = glob(f"{subject}/*.jpg")[0]
    
    vertices, faces, normals, faces_normals, textures, face_textures = load_scan(
        mesh_file, with_normal=True, with_texture=True
    )
    vertices -= vertices.mean(axis=0)
    
    # mesh = trimesh.load(mesh_file, process=False, maintain_order=True)
    # mesh = trimesh.intersections.slice_mesh_plane(mesh, [0, 1, 0], [0, -580.0, 0])
    # vertices = mesh.vertices
    # faces = mesh.faces
    # normals = mesh.vertex_normals
    # faces_normals = faces
    # textures = mesh.visual.uv
    # face_textures = faces
    
    
    # center
    scan_scale = 1.8 / (vertices.max(0)[up_axis] - vertices.min(0)[up_axis])
    vertices *= scale

    vmin = vertices.min(0)
    vmax = vertices.max(0)
    vmed = 0.5 * (vmax + vmin)
    # vmed[[0,2]] *= 0.

    prt, face_prt = prt_util.computePRT(mesh_file, scale, 10, 2)
    rndr = PRTRender(width=size, height=size, ms_rate=16, egl=egl)

    # texture
    texture_image = cv2.cvtColor(cv2.imread(tex_file), cv2.COLOR_BGR2RGB)

    tan, bitan = compute_tangent(normals)
    rndr.set_norm_mat(scan_scale, vmed)
    rndr.set_mesh(
        vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan,
        np.zeros((vertices.shape[0], 3))
    )
    rndr.set_albedo(texture_image)

    # camera
    cam = Camera(width=size, height=size)

    cam.near = -100
    cam.far = 100

    for y in range(0, 360, 360 // rotation):

        cam.ortho_ratio = (512 / size) * 0.4

        R = opengl_util.make_rotate(0, math.radians(y), 0)

        rndr.rot_matrix = R
        cam.sanity_check()
        rndr.set_camera(cam)

        # ==================================================================

        rndr.display()

        rgb_path = os.path.join(save_folder, "/".join(subject.split("/")[-4:]), 'render', f'{y:03d}.png')
        norm_path = os.path.join(save_folder, "/".join(subject.split("/")[-4:]), 'normal', f'{y:03d}.png')

        # if not os.path.exists(rgb_path):
        opengl_util.render_result(rndr, 0, rgb_path)
        # if not os.path.exists(norm_path):
        opengl_util.render_result(rndr, 1, norm_path)

    # ==================================================================


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', type=str, default="PuzzleIOI", help='dataset name')
    parser.add_argument('-out_dir', '--out_dir', type=str, default="./tmp", help='output dir')
    parser.add_argument('-num_views', '--num_views', type=int, default=4, help='number of views')
    parser.add_argument('-size', '--size', type=int, default=512, help='render size')
    parser.add_argument(
        '-debug', '--debug', action="store_true", help='debug mode, only render one subject'
    )
    parser.add_argument(
        '-headless', '--headless', action="store_true", help='headless rendering with EGL'
    )
    args = parser.parse_args()

    # rendering setup
    if args.headless:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    else:
        os.environ["PYOPENGL_PLATFORM"] = ""

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # shoud be put after PYOPENGL_PLATFORM
    import render.libs.opengl_util as opengl_util
    from render.libs.mesh import load_scan, compute_tangent
    import render.libs.prt_util as prt_util
    from render.libs.gl.init_gl import initialize_GL_context
    from render.libs.gl.prt_render import PRTRender
    from render.libs.camera import Camera

    print(
        f"Start Rendering {args.dataset} with {args.num_views} views, {args.size}x{args.size} size."
    )

    current_out_dir = f"{args.out_dir}/{args.dataset}_{args.num_views}views"
    os.makedirs(current_out_dir, exist_ok=True)
    print(f"Output dir: {current_out_dir}")

    # subjects = glob(f"./data/{args.dataset}/fitting/*/outfit*/")
    
    subjects = ["./data/PuzzleIOI/fitting/03539/outfit17/"]

    if args.debug:
        subjects = subjects[:2]
        render_types = ["normal"]
    else:
        random.shuffle(subjects)
        render_types = ["normal"]

    print(f"Rendering types: {render_types}")

    with Pool(processes=mp.cpu_count(), maxtasksperchild=1) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(
                    render_subject,
                    dataset=args.dataset,
                    save_folder=current_out_dir,
                    rotation=args.num_views,
                    size=args.size,
                    egl=args.headless,
                    render_types=render_types,
                ),
                subjects,
            ),
            total=len(subjects)
        ):
            pass

    pool.close()
    pool.join()

    print('Finish Rendering.')
