import os
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import pyrender
from torchvision.utils import make_grid
from glob import glob

import torch
from typing import Tuple
from pyrender.primitive import Primitive
from pyrender.constants import GLTF
from pyrender.mesh import Mesh

from pytorch3d import _C
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.structures import Pointclouds
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from pytorch3d.structures import Meshes

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3


# PointFaceDistance
class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """
    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_points: Scalar equal to maximum number of points in the batch
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.

            `dists[p]` is
            `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`

        """
        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists, idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


def _rand_barycentric_coords(
    size1, size2, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to generate random barycentric coordinates which are uniformly
    distributed over a triangle.

    Args:
        size1, size2: The number of coordinates generated will be size1*size2.
                      Output tensors will each be of shape (size1, size2).
        dtype: Datatype to generate.
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
            coordinates
    """
    uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    w = torch.cat([w0[..., None], w1[..., None], w2[..., None]], dim=2)

    return w


def sample_points_from_meshes(meshes, num_samples: int = 10000):
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: Integer giving the number of point samples per mesh.
        return_normals: If True, return normals for the sampled points.
        return_textures: If True, return textures for the sampled points.

    Returns:
        3-element tuple containing

        - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.
        - **normals**: FloatTensor of shape (N, num_samples, 3) giving a normal vector
          to each sampled point. Only returned if return_normals is True.
          For empty meshes the corresponding row in the normals array will
          be filled with 0.
        - **textures**: FloatTensor of shape (N, num_samples, C) giving a C-dimensional
          texture vector to each sampled point. Only returned if return_textures is True.
          For empty meshes the corresponding row in the textures array will
          be filled with 0.

        Note that in a future releases, we will replace the 3-element tuple output
        with a `Pointclouds` datastructure, as follows

        .. code-block:: python

            Pointclouds(samples, normals=normals, features=textures)
    """
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)    # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(verts, faces)    # Face areas can be zero.
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(areas, mesh_to_face[meshes.valid], max_faces)    # (N, F)

        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        samples_face_idxs = areas_padded.multinomial(
            num_samples, replacement=True
        )    # (N, num_samples)
        samples_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    # Randomly generate barycentric coords.
    # w                 (N, num_samples, 3)
    # sample_face_idxs  (N, num_samples)
    # samples_verts     (N, num_samples, 3, 3)

    samples_bw = _rand_barycentric_coords(num_valid_meshes, num_samples, verts.dtype, verts.device)
    sample_verts = verts[faces][samples_face_idxs]
    samples[meshes.valid] = (sample_verts * samples_bw[..., None]).sum(dim=-2)

    return samples, samples_face_idxs, samples_bw


def point_mesh_distance(meshes, pcls, weighted=True):

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")

    # packed representation for pointclouds
    points = pcls.points_packed()    # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]    # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face, idxs = _PointFaceDistance.apply(
        points, points_first_idx, tris, tris_first_idx, max_points, 5e-3
    )

    if weighted:
        # weight each example by the inverse of number of points in the example
        point_to_cloud_idx = pcls.packed_to_cloud_idx()    # (sum(P_i),)
        num_points_per_cloud = pcls.num_points_per_cloud()    # (N,)
        weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
        weights_p = 1.0 / weights_p.float()
        point_to_face = torch.sqrt(point_to_face) * weights_p

    return point_to_face, idxs


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


cameras = {}
person_id = "00145"
cam_cali_file = f"/ps/scratch/ps_shared/yxiu/PuzzleIOI/fitting/{person_id}/outfit5/camera.csd"

for i in range(1, 23, 1):

    camera_id = f"{i:02d}_C"
    ref_img_file = f"/ps/scratch/ps_shared/yxiu/PuzzleIOI/fitting/{person_id}/outfit5/images/{camera_id}.jpg"
    cameras[camera_id] = read_camera_cali(cam_cali_file, ref_img_file, camera_id)

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure


class Evaluation:
    def __init__(self, data_root, result_root, device):
        self.data_root = data_root
        self.result_root = result_root
        self.cameras = cameras
        self.results = {}

        self.scene = pyrender.Scene()
        self.light = pyrender.SpotLight(
            color=np.ones(3),
            intensity=50.0,
            innerConeAngle=np.pi / 16.0,
            outerConeAngle=np.pi / 6.0
        )

        self.scan_file = None
        self.recon_file = None
        self.pelvis_file = None
        self.scan = None
        self.scan_center = None
        self.recon = None

        self.ref_img = None

        self.subject = None
        self.outfit = None

        self.device = device

        # metrics
        self.psnr = PeakSignalNoiseRatio()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def load_assets(self, subject, outfit):
        self.scan_file = os.path.join(self.data_root, subject, outfit, "scan.obj")
        recon_name = f"{subject}_{outfit}_texture"
        self.recon_file = os.path.join(self.result_root, subject, outfit, f"obj/{recon_name}.obj")
        self.pelvis_file = glob(
            os.path.join(self.data_root.replace("fitting", "puzzle_cam"), subject, outfit) +
            "/smplx_*.npy"
        )
        self.smplx_path = os.path.join(self.data_root, subject, outfit, "smplx/smplx.obj")

        self.pelvis_y = np.load(self.pelvis_file[0], allow_pickle=True).item()["pelvis_y"]

        self.scan = self.load_mesh(self.scan_file, is_scan=True, use_pyrender=True)
        self.recon = self.load_mesh(self.recon_file, use_pyrender=True)

        self.subject = subject
        self.outfit = outfit

    def load_mesh(self, mesh_file, is_scan=False, use_pyrender=False):

        mesh = trimesh.load(mesh_file, process=False)

        if is_scan:
            scan_center = mesh.vertices.mean(axis=0)
            mesh = trimesh.intersections.slice_mesh_plane(mesh, [0, 1, 0], [0, -580.0, 0])
            if not use_pyrender:
                mesh.vertices -= scan_center
            mesh.vertices /= 1000.0
        else:
            mesh.vertices[:, 1] += self.pelvis_y
            if use_pyrender:
                mesh.vertices *= 1000.0
                mesh.vertices += scan_center
                mesh.vertices /= 1000.0

        if use_pyrender:
            mesh = pyrender.Mesh.from_trimesh(mesh)
        return mesh

    def calculate_visual_similarity(self, cam_id):

        assert isinstance(self.scan, pyrender.Mesh) and isinstance(self.recon, pyrender.Mesh)

        ref_img = plt.imread(
            os.path.join(self.data_root, self.subject, self.outfit, f"{cam_id}.jpg")
        )
        camera_cali = self.cameras[cam_id]
        r = pyrender.OffscreenRenderer(ref_img.shape[1], ref_img.shape[0])
        camera = pyrender.camera.IntrinsicsCamera(
            camera_cali['fx'], camera_cali['fy'], camera_cali['c_x'], camera_cali['c_y']
        )
        camera_pose = camera_cali['extrinsic']
        self.scene.add(camera, pose=camera_pose)
        self.scene.add(self.light, pose=camera_pose)

        self.scene.add(self.scan, name="scan")
        scan_color, _ = r.render(self.scene)
        scan_mask = (scan_color == scan_color[0, 0]).sum(axis=2, keepdims=True) != 3
        self.scene.remove_node(self.scene.get_node("scan"))

        self.scene.add(self.recon, name="recon")
        recon_color, _ = r.render(self.scene)
        self.scene.clear()

        render_dict = {
            "scan_color": scan_color * scan_mask,
            "recon_color": recon_color * scan_mask,
        }

        metrics = self.similarity(render_dict)

        return metrics

    def similarity(self, render_dict):
        psnr_diff = self.psnr(
            torch.tensor(render_dict["scan_color"]).permute(2, 0, 1).unsqueeze(0),
            torch.tensor(render_dict["recon_color"]).permute(2, 0, 1).unsqueeze(0)
        )
        ssim_diff = self.ssim(
            torch.tensor(render_dict["scan_color"]).permute(2, 0, 1).unsqueeze(0),
            torch.tensor(render_dict["recon_color"]).permute(2, 0, 1).unsqueeze(0)
        )
        lpips_diff = self.lpips(
            torch.tensor(render_dict["scan_color"]).permute(2, 0, 1).unsqueeze(0),
            torch.tensor(render_dict["recon_color"]).permute(2, 0, 1).unsqueeze(0)
        )

        return {"psnr": psnr_diff, "ssim": ssim_diff, "lpips": lpips_diff}

    def calculate_p2s(self):

        # reload scan and mesh
        scan = self.load_mesh(self.scan_file, is_scan=True, use_pyrender=False)
        recon = self.load_mesh(self.recon_file, use_pyrender=False)

        tgt_mesh = Meshes(
            verts=[torch.tensor(scan.vertices).float()], faces=[torch.tensor(scan.faces).long()]
        ).to(self.device)
        src_mesh = Meshes(
            verts=[torch.tensor(recon.vertices).float()], faces=[torch.tensor(recon.faces).long()]
        ).to(self.device)

        tgt_points = Pointclouds(tgt_mesh.verts_packed().unsqueeze(0))
        p2s_dist1 = point_mesh_distance(src_mesh, tgt_points)[0].sum() * 100.0

        samples_src, _, _ = sample_points_from_meshes(src_mesh, 100000)
        src_points = Pointclouds(samples_src)
        p2s_dist2 = point_mesh_distance(tgt_mesh, src_points)[0].sum() * 100.0

        chamfer_dist = 0.5 * (p2s_dist1 + p2s_dist2)
        return p2s_dist1, chamfer_dist


class Evaluation_EASY:
    def __init__(self, data_root, result_geo_root, result_img_root, device):
        self.data_root = data_root
        self.result_geo_root = result_geo_root
        self.result_img_root = result_img_root
        self.results = {}

        self.scan_file = None
        self.recon_file = None
        self.pelvis_file = None
        self.smplx_file = None

        self.scan = None
        self.scan_center = None
        self.recon = None
        self.ref_img = None
        self.pelvis_y = None

        self.subject = None
        self.outfit = None

        self.device = device

        # metrics
        self.psnr = PeakSignalNoiseRatio()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def load_paths(self, subject, outfit):

        # geo path
        self.scan_file = os.path.join(self.data_root, subject, outfit, "scan.obj")
        recon_name = f"{subject}_{outfit}_texture"
        self.recon_file = os.path.join(
            self.result_geo_root, subject, outfit, f"obj/{recon_name}.obj"
        )
        self.pelvis_file = glob(
            os.path.join(self.data_root.replace("fitting", "puzzle_cam"), subject, outfit) +
            "/smplx_*.npy"
        )
        self.smplx_file = os.path.join(self.data_root, subject, outfit, "smplx/smplx.obj")

        # tex path
        self.render_gt_dir = os.path.join(self.result_img_root, "fitting", subject, outfit)
        self.render_recon_dir = os.path.join(self.result_img_root, "puzzle_cam", subject, outfit)

        self.subject = subject
        self.outfit = outfit

    def load_assets(self):

        # geo data
        self.pelvis_y = np.load(self.pelvis_file[0], allow_pickle=True).item()["pelvis_y"]
        self.scan = self.load_mesh(self.scan_file, is_scan=True)
        self.recon = self.load_mesh(self.recon_file)
        self.smplx = trimesh.load(self.smplx_file, process=False)

        # tex data
        self.src_imgs = {}
        self.tgt_imgs = {}
        for mode in ["normal", "render"]:
            self.src_imgs[mode] = [
                torch.as_tensor(plt.imread(img_file))
                for img_file in glob(f"{self.render_gt_dir}/{mode}/*.png")
            ]
            self.tgt_imgs[mode] = [
                torch.as_tensor(plt.imread(img_file))
                for img_file in glob(f"{self.render_recon_dir}/{mode}/*.png")
            ]

    def load_mesh(self, mesh_file, is_scan=False):

        mesh = trimesh.load(mesh_file, process=False)

        if is_scan:
            scan_center = mesh.vertices.mean(axis=0)
            mesh = trimesh.intersections.slice_mesh_plane(mesh, [0, 1, 0], [0, -580.0, 0])
            mesh.vertices -= scan_center
            mesh.vertices /= 1000.0
        else:
            mesh.vertices[:, 1] += self.pelvis_y

        return mesh

    def calculate_visual_similarity(self):

        tgt_normal_arr = make_grid(torch.cat(self.tgt_imgs["normal"], dim=0), nrow=4)
        tgt_render_arr = make_grid(torch.cat(self.tgt_imgs["render"], dim=0), nrow=4)
        src_normal_arr = make_grid(torch.cat(self.src_imgs["normal"], dim=0), nrow=4)
        src_render_arr = make_grid(torch.cat(self.src_imgs["render"], dim=0), nrow=4)

        mask_arr = tgt_normal_arr[:, :, [-1]] * (tgt_normal_arr[:, :, [2]] > 0.5)
        # plt.imsave(f"./tmp/{self.subject}_{self.outfit}_mask.jpg", mask_arr[:,:,0])

        metrics = {}
        metrics["Normal"] = (((((src_normal_arr[..., :3] - tgt_normal_arr[..., :3]) * mask_arr)**
                               2).sum(dim=2).mean()) * 4.0).item()

        render_dict = {
            "scan_color": tgt_render_arr[..., :3] * mask_arr,
            "recon_color": src_render_arr[..., :3] * mask_arr,
        }

        metrics.update(self.similarity(render_dict))

        return metrics

    def similarity(self, render_dict):
        psnr_diff = self.psnr(
            render_dict["scan_color"].permute(2, 0, 1).unsqueeze(0),
            render_dict["recon_color"].permute(2, 0, 1).unsqueeze(0)
        )
        ssim_diff = self.ssim(
            render_dict["scan_color"].permute(2, 0, 1).unsqueeze(0),
            render_dict["recon_color"].permute(2, 0, 1).unsqueeze(0)
        )
        lpips_diff = self.lpips(
            render_dict["scan_color"].permute(2, 0, 1).unsqueeze(0),
            render_dict["recon_color"].permute(2, 0, 1).unsqueeze(0)
        )

        return {"PSNR": psnr_diff.item(), "SSIM": ssim_diff.item(), "LPIPS": lpips_diff.item()}

    def calculate_p2s(self):

        # reload scan and mesh

        tgt_mesh = Meshes(
            verts=[torch.tensor(self.scan.vertices).float()],
            faces=[torch.tensor(self.scan.faces).long()]
        ).to(self.device)
        src_mesh = Meshes(
            verts=[torch.tensor(self.recon.vertices).float()],
            faces=[torch.tensor(self.recon.faces).long()]
        ).to(self.device)

        tgt_points = Pointclouds(tgt_mesh.verts_packed().unsqueeze(0))
        p2s_dist1 = point_mesh_distance(src_mesh, tgt_points)[0].sum() * 100.0

        samples_src, _, _ = sample_points_from_meshes(src_mesh, 100000)
        src_points = Pointclouds(samples_src)
        p2s_dist2 = point_mesh_distance(tgt_mesh, src_points)[0].sum() * 100.0

        chamfer_dist = 0.5 * (p2s_dist1 + p2s_dist2)

        return {"P2S": p2s_dist1.item(), "Chamfer": chamfer_dist.item()}


class PyRenderer:
    def __init__(self, data_root, device):
        self.data_root = data_root
        self.cameras = np.load("./multi_concepts/camera.npy", allow_pickle=True).item()
        self.results = {}

        self.scene = pyrender.Scene(ambient_light=np.ones(3))
        self.material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1., 1., 1., 1.],
            metallicFactor=0.0,
            roughnessFactor=0.0,
            smooth=True,
            alphaMode='BLEND'
        )

        self.scan_file = None
        self.pelvis_file = None

        self.scan = None
        self.ref_img = None

        self.subject = None
        self.outfit = None

        self.device = device

    def load_assets(self, subject, outfit):
        self.scan_file = os.path.join(self.data_root, subject, outfit, "scan.obj")
        self.pelvis_file = glob(
            os.path.join(self.data_root.replace("fitting", "puzzle_cam"), subject, outfit) +
            "/smplx_*.npy"
        )
        self.smplx_path = os.path.join(self.data_root, subject, outfit, "smplx/smplx.obj")

        self.scan = self.load_mesh(self.scan_file)
        self.smplx = trimesh.load(self.smplx_path, process=False)
        self.pelvis_y = 0.5 * (self.smplx.vertices[:, 1].min() + self.smplx.vertices[:, 1].max())

        self.subject = subject
        self.outfit = outfit

    def load_mesh(self, mesh_file):

        mesh = trimesh.load(mesh_file, process=False)
        mesh = trimesh.intersections.slice_mesh_plane(mesh, [0, 1, 0], [0, -580.0, 0])

        return mesh

    def load_material(self, mesh, camera_mat):

        primitive = Primitive(
            positions=mesh.vertices / 1000.0,
            normals=mesh.vertex_normals,
            texcoord_0=None,
            color_0=(np.matmul(mesh.vertex_normals,
                               np.linalg.inv(camera_mat).T) + 1.0) * 0.5,
            indices=mesh.faces,
            material=self.material,
            mode=GLTF.TRIANGLES
        )

        return Mesh(primitives=[primitive], is_visible=True)

    def render_normal(self, cam_id):

        ref_img = plt.imread(
            os.path.join(self.data_root, self.subject, self.outfit, "images", f"{cam_id}.jpg")
        )
        camera_cali = self.cameras[cam_id]
        r = pyrender.OffscreenRenderer(ref_img.shape[1], ref_img.shape[0])
        camera = pyrender.camera.IntrinsicsCamera(
            camera_cali['fx'], camera_cali['fy'], camera_cali['c_x'], camera_cali['c_y']
        )
        camera_pose = camera_cali['extrinsic']
        self.scene.add(camera, pose=camera_pose)

        tex_scan = self.load_material(self.scan, camera_cali['extrinsic'][:3, :3])
        self.scene.add(tex_scan, name="scan")

        scan_color, _ = r.render(self.scene, flags=pyrender.constants.RenderFlags.FLAT)
        self.scene.clear()
        mask_arr = (scan_color.sum(2)[...,None] != 255 * 3) * (scan_color[:, :, [2]] > 0.5 * 255.0)

        return np.concatenate([scan_color, (mask_arr * 255).astype(np.uint8)], axis=2)
