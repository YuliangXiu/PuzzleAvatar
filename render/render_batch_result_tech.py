import argparse
import os, sys
import cv2
from glob import glob
import numpy as np
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


def render_subject(subject, save_folder, rotation, size, egl, overwrite, head):

    initialize_GL_context(width=size, height=size, egl=egl)

    scale = 100.0
    up_axis = 1

    try:
        mesh_file = glob(f"{subject}/obj/*texture.obj")[0]
        tex_file = glob(f"{subject}/obj/*albedo.png")[0]
        smpl_file = glob(f"{subject}/obj/*smpl.npy")[0]
    except:
        with open("./data/PuzzleIOI/error_eval_tech.txt", "a") as f:
            head = "/".join(subject.split("/")[2:-1])
            f.write(f"{head} {' '.join(head.split('/')[-2:])}\n")

        return

    [person, outfit] = mesh_file.split("/")[-4:-2]
    scan_file = f"./data/PuzzleIOI/fitting/{person}/{outfit}/scan.obj"

    vertices, faces, normals, faces_normals, textures, face_textures = load_scan(
        mesh_file, with_normal=True, with_texture=True
    )

    vertices_scan, _ = load_scan(scan_file, with_normal=False, with_texture=False)
    scan_center = vertices_scan.mean(axis=0)
    vertices_scan -= scan_center

    smpl_data = np.load(smpl_file, allow_pickle=True).item()
    smpl_scale = smpl_data["scale"].cpu().numpy()
    smpl_trans = smpl_data["transl"].cpu().numpy()
    smpl_model_trans = smpl_trans - np.array([-0.06, -0.40, 0.0]) - scan_center / 1000.0

    vertices /= smpl_scale
    vertices -= smpl_trans
    vertices += smpl_model_trans
    vertices *= 1000.0

    # center
    scan_scale = 1.8 / (vertices_scan.max(0)[up_axis] - vertices_scan.min(0)[up_axis])
    vertices *= scale
    vertices_scan *= scale

    vmin = vertices_scan.min(0)
    vmax = vertices_scan.max(0)
    vmed = 0.5 * (vmax + vmin)
    # vmed[[0, 2]] *= 0.

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

        if not head:
            cam.ortho_ratio = (512 / size) * 0.4
        else:
            cam.ortho_ratio = (512 / size) * 0.4 * 0.4
            cam.center[up_axis] = 0.8 * scale

        R = opengl_util.make_rotate(0, math.radians(y), 0)

        rndr.rot_matrix = R
        cam.sanity_check()
        rndr.set_camera(cam)

        # ==================================================================

        rndr.display()

        rgb_path = os.path.join(
            save_folder, "/".join(subject.split("/")[4:]).replace("fitting", "tech_full"),
            'render_head' if head else 'render', f'{y:03d}.png'
        )
        norm_path = os.path.join(
            save_folder, "/".join(subject.split("/")[4:]).replace("fitting", "tech_full"),
            'normal_head' if head else 'normal', f'{y:03d}.png'
        )

        if overwrite or (not os.path.exists(rgb_path)):
            opengl_util.render_result(rndr, 0, rgb_path)
        if overwrite or (not os.path.exists(norm_path)):
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
    parser.add_argument(
        '-overwrite', '--overwrite', action="store_true", help='overwrite existing files'
    )
    parser.add_argument('-head', '--head', action="store_true", help='head rendering mode')
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

    subjects = np.loadtxt("data/PuzzleIOI/subjects_all.txt", dtype=str, delimiter=" ")[:, 0]
    subjects = [f"./results/full/{outfit.replace('puzzle_capture', 'tech')}" for outfit in subjects]
    # subjects = [item for item in subjects if "03619" in item or "03633" in item]
    
    

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
                    overwrite=args.overwrite,
                    head=args.head,
                ),
                subjects,
            ),
            total=len(subjects)
        ):
            pass

    pool.close()
    pool.join()

    print('Finish Rendering.')
