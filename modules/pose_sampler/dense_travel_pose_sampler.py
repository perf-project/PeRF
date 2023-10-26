import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .pose_sampler import PoseSampler
from utils.debug_utils import printarr
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

 
def _get_travel_indices(positions):
    print('get travel indices')
    n = len(positions)
    indices = torch.arange(0, n, dtype=torch.int64)
    dis = 1e8
    n_steps = 10000
    for iter_step in tqdm(range(n_steps)):
        a = np.random.randint(n)
        b = np.random.randint(n)
        new_indices = indices.clone()
        new_indices[a] = indices[b]
        new_indices[b] = indices[a]
        prev_indices = new_indices[:-1]
        next_indices = new_indices[1:]

        shitfs = positions[prev_indices] - positions[next_indices]
        new_dis = torch.linalg.norm(shitfs, 2, -1).sum()
        ratio = (1. - (iter_step / n_steps)) ** 5
        if new_dis < dis or np.random.rand() < ratio:
            indices = new_indices
            dis = new_dis

    return indices


class DenseTravelPoseSampler(PoseSampler):
    def __init__(self, sparse_pose_sampler: PoseSampler, n_dense_poses, dir_bias_ratio=-1):
        super().__init__()
        sparse_poses = []
        for i in range(sparse_pose_sampler.n_poses):
            sparse_poses.append(sparse_pose_sampler.sample_pose(i))

        sparse_poses = torch.stack(sparse_poses, 0)
        sparse_positions = sparse_poses[:, :3, 3]
        travel_indices = _get_travel_indices(sparse_positions)
        # travel_indices = travel_indices[:int(len(travel_indices) * 0.8)]
        travel_sparse_poses = sparse_poses[travel_indices]

        N = n_dense_poses * 50
        sparse_pts = travel_sparse_poses[:, :3, 3]
        shifts = sparse_pts[1:] - sparse_pts[:-1]
        sec_lens = torch.linalg.norm(shifts, 2, -1, True)
        sec_ratios = sec_lens / sec_lens.sum()
        sec_n_poses = torch.round(N * sec_ratios).to(torch.int64)

        pts = []
        for i in range(len(sec_n_poses)):
            a = sparse_pts[i]
            b = sparse_pts[i + 1]
            cur_n = sec_n_poses[i].item()
            t = torch.linspace(.5 / cur_n, 1. - .5 / cur_n, cur_n)
            cur_pts = a[None, :] * (1. - t)[:, None] + b[None, :] * t[:, None]
            pts.append(cur_pts)

        pts = torch.concatenate(pts, 0)
        pts = _resample_uniformly(pts)
        pts = pts[::50]

        pts = pts.cpu().numpy()
        for i in range(3):
            pts[:, i] = gaussian_filter1d(pts[:, i], sigma=20)

        pts = torch.from_numpy(pts).cuda()
        self.sample_poses = torch.eye(4)[None].repeat(len(pts), 1, 1)
        self.sample_poses[:, :3, 3] = pts

        self.n_poses = len(self.sample_poses)

        to_vecs = pts.clone()
        to_vecs[:-1] = pts[1:] - pts[:-1]
        to_vecs[-1] = to_vecs[-2]

        for i in range(3):
            filtered = gaussian_filter1d(to_vecs[:, i].cpu().numpy(), sigma=30)
            to_vecs[:, i] = torch.from_numpy(filtered).to(to_vecs.device)

        to_vecs = to_vecs / torch.linalg.norm(to_vecs, 2, -1, True)
        up_vecs = torch.zeros_like(to_vecs)
        up_vecs[..., 2] = 1
        left_vecs = torch.cross(up_vecs, to_vecs)

        left_vecs /= torch.linalg.norm(left_vecs, 2, -1, True)
        to_vecs /= torch.linalg.norm(to_vecs, 2, -1, True)
        to_vecs = to_vecs + dir_bias_ratio * left_vecs
        to_vecs /= torch.linalg.norm(to_vecs, 2, -1, True)

        self.sample_poses[:, :3, :3] = look_at(to_vecs)

    @torch.no_grad()
    def sample_pose(self, idx):
        return self.sample_poses[idx]
