import os
import sys
import trimesh
import numpy as np
from tqdm import tqdm
from glob import glob
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import pyrender


def read_camera_cali(file, ref_img_file, camera_id):

    with open(file) as f:
        lines = [line.rstrip() for line in f]

    camera_cali = {}

    # read reference image:
    ref_img = plt.imread(ref_img_file)

    RENDER_RESOLUTION = [ref_img.shape[1], ref_img.shape[0]]

    line_id = None
    for i in range(len(lines)):
        if lines[i].split()[0] == camera_id:
            line_id = i
            break
    if line_id is None:
        print("Wrong camera id!")
        exit()

    camera_info = lines[line_id].split()

    Rxyz = np.array(camera_info[1:4]).astype(np.float32)
    t = np.array(camera_info[4:7]).astype(np.float32) / 1000.0

    R = Rotation.from_rotvec(np.array([Rxyz[0], Rxyz[1], Rxyz[2]]))
    R = R.as_matrix()

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    T = np.linalg.inv(T)
    T_openGL = np.eye(4, dtype=np.float32)
    T_openGL[:3, :3] = Rotation.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()

    T = np.dot(T, T_openGL)
    camera_cali['extrinsic'] = T

    # intrinsics camera
    camera_cali['fx'] = float(camera_info[7])
    camera_cali['fy'] = float(camera_info[8])
    camera_cali['c_x'] = float(camera_info[9]) + 0.5
    camera_cali['c_y'] = float(camera_info[10]) + 0.5

    camera_cali['c_x'] *= RENDER_RESOLUTION[0]
    camera_cali['c_y'] *= RENDER_RESOLUTION[1]
    camera_cali['fx'] *= RENDER_RESOLUTION[0]
    camera_cali['fy'] *= RENDER_RESOLUTION[0]

    camera = pyrender.camera.IntrinsicsCamera(
        camera_cali['fx'], camera_cali['fy'], camera_cali['c_x'], camera_cali['c_y']
    )

    camera_cali['intrinsic'] = camera.get_projection_matrix(
        width=RENDER_RESOLUTION[0], height=RENDER_RESOLUTION[1]
    )
    return camera_cali


scene = pyrender.Scene()
light = pyrender.SpotLight(
    color=np.ones(3), intensity=50.0, innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0
)


def render(scan_file, ref_img_file, camera_cali):

    # read reference image:
    ref_img = plt.imread(ref_img_file)

    RENDER_RESOLUTION = [ref_img.shape[1], ref_img.shape[0]]
    camera = pyrender.camera.IntrinsicsCamera(
        camera_cali['fx'], camera_cali['fy'], camera_cali['c_x'], camera_cali['c_y']
    )
    camera_pose = camera_cali['extrinsic']
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)

    # Add mesh:
    scan_mesh = trimesh.load(scan_file, process=False)
    scan_mesh = trimesh.intersections.slice_mesh_plane(scan_mesh, [0, 1, 0], [0, -580.0, 0])
    scan_mesh.vertices /= 1000.0

    mesh = pyrender.Mesh.from_trimesh(scan_mesh)
    scene.add(mesh)

    # Render
    r = pyrender.OffscreenRenderer(RENDER_RESOLUTION[0], RENDER_RESOLUTION[1])
    color, _ = r.render(scene)
    mask = (color == color[0, 0]).sum(axis=2, keepdims=True) != 3
    masked_img = ref_img * mask

    scene.clear()

    return masked_img


cameras = {}
person_id = "00145"
cam_cali_file = f"/ps/scratch/ps_shared/yxiu/PuzzleIOI/fitting/{person_id}/outfit5/camera.csd"
