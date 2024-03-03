import os
import sys
import trimesh
import numpy as np
from tqdm import tqdm
from glob import glob
from .puzzle_utils import read_camera_cali, render

data_root = "./data/PuzzleIOI/fitting"
result_root = "./results/PuzzleIOI/puzzle"
pbar = tqdm(os.listdir(data_root))


cameras = {}
person_id = "00145"
cam_cali_file = f"/ps/scratch/ps_shared/yxiu/PuzzleIOI/fitting/{person_id}/outfit5/camera.csd"

for i in range(1, 23, 1):

    camera_id = f"{i:02d}_C"
    ref_img_file = f"/ps/scratch/ps_shared/yxiu/PuzzleIOI/fitting/{person_id}/outfit5/images/{camera_id}.jpg"
    cameras[camera_id] = read_camera_cali(cam_cali_file, ref_img_file, camera_id)

for subject in pbar:
    for motion in os.listdir(os.path.join(data_root, subject)):
        pbar.set_description(f"Processing {subject}/{motion}")
        scan_path = os.path.join(data_root, subject, motion, "scan.obj")
        smplx_path = os.path.join(data_root, subject, motion, "smplx/smplx.obj")
        smplx_param_path = os.path.join(data_root, subject, motion, "smplx/smplx.pkl")
        recon_name = f"{subject}_{motion}_geometry_final"
        recon_path = os.path.join(result_root, subject, motion, f"obj/{recon_name}.obj")
        pelvis_path = glob(os.path.join(data_root.replace("fitting", "puzzle"), subject, motion)+"/smplx_*.npy")
        
        if os.path.exists(recon_path) and len(pelvis_path) > 0:
            pelvis_y = np.load(pelvis_path[0], allow_pickle=True).item()["pelvis_y"]
            smplx_obj = trimesh.load(smplx_path)
            
            scan_obj = trimesh.load(scan_path)
            scan_obj_mean = scan_obj.vertices.mean(axis=0)
            scan_obj = trimesh.intersections.slice_mesh_plane(scan_obj, [0, 1, 0], [0, -580.0, 0])
            scan_obj.vertices -= scan_obj_mean
            scan_obj.vertices /= 1000.0
            
            recon_path = "./tmp/recon.obj"
            recon_obj = trimesh.load(recon_path)
            # recon_obj = recon_obj.simplify_quadric_decimation(10000)
            # recon_obj.export("./tmp/recon.obj")
            recon_obj.vertices[:,1] += pelvis_y
            
            (scan_obj+recon_obj).export("./tmp/scan_recon.obj")
                        
            sys.exit()
            
            
            
                