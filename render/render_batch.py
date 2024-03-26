import argparse
import os, sys
import cv2
import os.path as osp
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


def grid_save(dir):

    from PIL import Image

    ROW = 2
    COL = 6

    # Define the dimensions of the final big image
    res = 256    # Resolution of each image

    # Create a new blank image for the final big image
    big_image = Image.new('RGB', (ROW * res, COL * res))

    for mode in ['body', 'head']:

        # Loop through each image and paste it into the big image
        for i in range(ROW):
            for j in range(COL):
                # Open the PNG image
                image_path = f"{mode}_{(i*COL+j)*(int(360/ROW/COL)):03d}.png"
                img = Image.open(osp.join(dir, image_path)).resize((res, res))

                # Paste the current image into the big image
                big_image.paste(img, (i * res, j * res))

        # Save the final big image
        big_image.save(osp.join(dir, f"final_{mode}.png"))


def render_subject(subject, dataset, save_folder, rotation, size, render_types, egl):

    initialize_GL_context(width=size, height=size, egl=egl)

    scale = 100.0
    up_axis = 1

    mesh_file = os.path.join(f'./data/{dataset}/scans/{subject}', f'{subject}.obj')
    tex_file = f'./data/{dataset}/scans/{subject}/material0.jpeg'

    vertices, faces, normals, faces_normals, textures, face_textures = load_scan(
        mesh_file, with_normal=True, with_texture=True
    )

    # center
    scan_scale = 1.8 / (vertices.max(0)[up_axis] - vertices.min(0)[up_axis])

    vertices *= scale
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    vmed = 0.5 * (vmax + vmin)

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
    shs = np.load('./render/env_sh.npy')

    # camera
    cam = Camera(width=size, height=size)

    cam.near = -100
    cam.far = 100

    for mode in ['head', 'body']:

        for y in range(0, 360, 360 // rotation):

            cam.ortho_ratio = (512 / size) * 0.4

            if mode == 'body':
                cam.ortho_ratio = (512 / size) * 0.4 * np.random.uniform(0.5, 1.3)
                cam.center[up_axis] = np.random.uniform(-0.5, 0.5) * scale
                # cam.center[up_axis] = 0.0
                cam.up = np.dot(
                    np.array([0., 1., 0.]),
                    opengl_util.make_rotate(math.radians(np.random.uniform(-60, 60)), 0, 0)
                )
            else:
                cam.ortho_ratio = (512 / size) * 0.4 * np.random.uniform(0.3, 0.4)
                cam.center[up_axis] = 0.8 * scale
                cam.up = np.dot(
                    np.array([0., 1., 0.]),
                    opengl_util.make_rotate(math.radians(np.random.uniform(-30, 30)), 0, 0)
                )

            R = opengl_util.make_rotate(0, math.radians(y), 0)

            rndr.rot_matrix = R
            cam.sanity_check()
            rndr.set_camera(cam)

            dic = {'ortho_ratio': cam.ortho_ratio, 'scale': scan_scale, 'center': vmed, 'R': R}

            if "light" in render_types:

                # random light
                sh_id = random.randint(0, shs.shape[0] - 1)
                sh = shs[sh_id]
                sh_angle = 0.2 * np.pi * (random.random() - 0.5)
                sh = opengl_util.rotateSH(sh, opengl_util.make_rotate(0, sh_angle, 0).T)
                dic.update({"sh": sh})

                rndr.set_sh(sh)
                rndr.analytic = False
                rndr.use_inverse_depth = False

            # ==================================================================

            rndr.display()

            rgb_path = os.path.join(save_folder, subject, 'render', f'{mode}_{y:03d}.png')
            norm_path = os.path.join(save_folder, subject, 'normal', f'{mode}_{y:03d}.png')

            if not os.path.exists(rgb_path):
                opengl_util.render_result(rndr, 0, rgb_path)
            if not os.path.exists(norm_path):
                opengl_util.render_result(rndr, 1, norm_path)

    # grid_save(os.path.join(save_folder, subject, 'normal'))

    # ==================================================================


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', type=str, default="thuman2", help='dataset name')
    parser.add_argument('-out_dir', '--out_dir', type=str, default="./tmp", help='output dir')
    parser.add_argument('-num_views', '--num_views', type=int, default=36, help='number of views')
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

    subjects = np.loadtxt(f"./data/{args.dataset}/all.txt", dtype=str)

    if args.debug:
        subjects = subjects[:2]
        render_types = ["light", "normal"]
    else:
        random.shuffle(subjects)
        render_types = ["light", "normal"]

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
