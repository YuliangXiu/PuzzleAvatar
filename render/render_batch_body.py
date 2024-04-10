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


def render_subject(subject, save_folder, rotation, size, egl):

    initialize_GL_context(width=size, height=size, egl=egl)

    scale = 100.0
    up_axis = 1

    mesh_file = glob(f"{subject}/*.obj")[0]

    scan = trimesh.load_mesh(mesh_file, process=False, maintain_orders=True)
    vertices = scan.vertices
    vertices -= vertices.mean(axis=0)

    smplx_file = f"{subject}/smplx/smplx.obj"
    smplx_mesh = trimesh.load_mesh(smplx_file, process=False, maintain_orders=True)
    
    # center
    scan_scale = 1.8 / (vertices.max(0)[up_axis] - vertices.min(0)[up_axis])
    vertices *= scale

    vmin = vertices.min(0)
    vmax = vertices.max(0)
    vmed = 0.5 * (vmax + vmin)

    rndr = ColorRender(width=size, height=size, egl=egl)
    rndr.set_mesh(
        smplx_mesh.vertices * scale * 1000.0, smplx_mesh.faces, smplx_mesh.vertices,
        smplx_mesh.vertex_normals
    )
    rndr.set_norm_mat(scan_scale, vmed)

    # camera
    cam = Camera(width=size, height=size)

    for y in range(0, 360, 360 // rotation):

        cam.ortho_ratio = (512 / size) * 0.4

        R = opengl_util.make_rotate(0, math.radians(y), 0)

        rndr.rot_matrix = R
        cam.near = -100
        cam.far = 100
        cam.sanity_check()
        rndr.set_camera(cam)
        rndr.display()

        norm_path = os.path.join(
            save_folder, "/".join(subject.split("/")[-4:]), 'body', f'{y:03d}_F.png'
        )
        opengl_util.render_result(rndr, 1, norm_path)

        # back render
        cam.near = 100
        cam.far = -100
        cam.sanity_check()
        rndr.set_camera(cam)
        rndr.display()

        norm_path = os.path.join(
            save_folder, "/".join(subject.split("/")[-4:]), 'body', f'{y:03d}_B.png'
        )
        opengl_util.render_result(rndr, 1, norm_path, front=False)
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
    from render.libs.gl.init_gl import initialize_GL_context
    from render.libs.gl.color_render import ColorRender
    from render.libs.camera import Camera

    print(
        f"Start Rendering {args.dataset} with {args.num_views} views, {args.size}x{args.size} size."
    )

    current_out_dir = f"{args.out_dir}/{args.dataset}_{args.num_views}views"
    os.makedirs(current_out_dir, exist_ok=True)
    print(f"Output dir: {current_out_dir}")

    subjects = glob(f"./data/{args.dataset}/fitting/*/outfit*/")

    if args.debug:
        subjects = subjects[:2]
    else:
        random.shuffle(subjects)

    with Pool(processes=mp.cpu_count(), maxtasksperchild=1) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(
                    render_subject,
                    save_folder=current_out_dir,
                    rotation=args.num_views,
                    size=args.size,
                    egl=args.headless,
                ),
                subjects,
            ),
            total=len(subjects)
        ):
            pass

    pool.close()
    pool.join()

    print('Finish Rendering.')
