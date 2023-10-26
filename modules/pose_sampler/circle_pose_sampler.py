from .pose_sampler import PoseSampler

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.camera_utils import *
from scipy.ndimage import minimum_filter1d, gaussian_filter1d


@torch.no_grad()
def _resample_uniformly(pts):
    n = len(pts)
    pts = F.interpolate(pts[None].permute(0, 2, 1), size=n * 128, mode='linear')[0].permute(1, 0)
    cat_pts = torch.cat([pts, pts[:1]], dim=0)
    bias = cat_pts[1:] - cat_pts[:-1]
    bias_len = torch.linalg.norm(bias, 2, -1, keepdim=False)
    bias_len = torch.cumsum(bias_len, dim=0)
    bias_len = bias_len / bias_len[-1]
    idx = torch.searchsorted(bias_len, torch.linspace(0., 1. - 1./n, n))
    return pts[idx]


@torch.no_grad()
def _get_trajectory_normals(pts):
    n_pts = len(pts)
    sigma = float(n_pts) / 32. * 2. + 1.
    ext_pts = torch.cat([pts, pts[:1]], dim=0)
    right_vec = (ext_pts[1:] - ext_pts[:-1])
    right_vec = right_vec / torch.linalg.norm(right_vec, 2, -1, True)
    up_vec = torch.zeros_like(right_vec)
    up_vec[:, 2] = 1
    to_vec = torch.cross(up_vec, right_vec)
    to_vec = to_vec / torch.linalg.norm(to_vec, 2, -1, True)
    to_vec = to_vec.cpu().numpy()
    for i in range(3):
        to_vec[:, i] = gaussian_filter1d(to_vec[:, i], sigma=sigma, mode='wrap')
    to_vec = torch.from_numpy(to_vec).to(pts.device)
    to_vec = to_vec / torch.linalg.norm(to_vec, 2, -1, True)
    return -to_vec


class CirclePoseSampler(PoseSampler):
    def __init__(self, distance_map, traverse_ratios, n_anchors_per_ratio, test_z_min_max=(0., 0.), **kwargs):
        super().__init__()
        if torch.is_tensor(distance_map):
            distance_map = distance_map.cpu().numpy()
        distance_map = distance_map.squeeze()

        height, width = distance_map.shape
        pano_coords = img_to_pano_coord(img_coord_from_hw(height, width))

        plane_dis = distance_map * np.cos(pano_coords[:, :, 0].cpu().numpy())
        h_height = height // 2
        plane_dis = plane_dis[h_height - 10: h_height + 10]
        plane_dis[np.where(plane_dis < 1e-5)] = 1e9
        plane_dis = np.min(plane_dis, axis=0)

        for i in range(1, width):
            if plane_dis[i] > 1e8:
                plane_dis[i] = plane_dis[i - 1]

        for i in range(1, width):
            if plane_dis[width - i - 1] > 1e8:
                plane_dis[width - i - 1] = plane_dis[width - i]

        pool_size = (width // 16) // 2 * 2 + 1
        filtered_plane_dis = minimum_filter1d(plane_dis, size=pool_size, mode='wrap')
        smooth_size = (width // 8) // 2 * 2 + 1
        smoothed_plane_dis = gaussian_filter1d(filtered_plane_dis, sigma=smooth_size, mode='wrap')
        blur_size = (width // 64) // 2 * 2 + 1
        filtered_plane_dis = gaussian_filter1d(filtered_plane_dis, sigma=blur_size, mode='wrap')
        plane_coords = torch.stack([torch.ones([width]) * .5,
                                    torch.linspace(.5 / width, 1. - .5 / width, width)], -1)

        plane_pts = img_coord_to_pano_direction(plane_coords).cpu().numpy()
        self.plane_pts_raw = torch.from_numpy(plane_pts * plane_dis[:, None]).cuda()
        self.plane_pts_filter = torch.from_numpy(plane_pts * filtered_plane_dis[:, None]).cuda()
        self.plane_pts_smooth = torch.from_numpy(plane_pts * smoothed_plane_dis[:, None]).cuda()
        filtered_plane_dis = torch.from_numpy(filtered_plane_dis).cuda()
        smoothed_plane_dis = torch.from_numpy(smoothed_plane_dis).cuda()

        # Get anchors
        n_anchors_per_ratio = n_anchors_per_ratio
        plane_coords = torch.stack([torch.ones([width]) * .5,
                                    torch.linspace(.5 / width, 1. - .5 / width, width)], -1)

        circle_pts = img_coord_to_pano_direction(plane_coords)

        anchor_pts = []
        test_z_min, test_z_max = test_z_min_max
        for i, traverse_ratio in enumerate(traverse_ratios):
            traverse_pts = circle_pts * filtered_plane_dis[:, None] * traverse_ratio
            traverse_pts = _resample_uniformly(traverse_pts)
            n = n_anchors_per_ratio[i]
            bias = 0. if i % 2 == 0 else .5 / n
            anchor_idx = torch.linspace(.5 / n, 1. - .5 / n, n) + bias
            anchor_idx = (anchor_idx * width).to(torch.long).clip(0, width - 1)
            cur_pts = traverse_pts[anchor_idx].clone()
            for j in range(len(cur_pts)):
                cur_pts[j, 2] = test_z_min if (i + j) % 2 == 0 else test_z_max
            anchor_pts.append(cur_pts)

            traverse_pts[..., 2] += (test_z_min + test_z_max) * .5

        self.anchor_pts = torch.cat(anchor_pts, dim=0)
        self.traverse_pts = _resample_uniformly(circle_pts * smoothed_plane_dis[:, None] * .3)
        self.traverse_normals = _get_trajectory_normals(self.traverse_pts)

        self.n_anchors = len(self.anchor_pts)
        self.n_poses = self.n_anchors

    @torch.no_grad()
    def sample_pose(self, idx):
        pose = torch.eye(4)
        pose[:3, 3] = self.anchor_pts[idx]
        return pose


