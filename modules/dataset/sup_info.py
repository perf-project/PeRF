import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import kornia
from kornia.morphology import erosion, dilation

from utils.camera_utils import *
from utils.utils import write_image
from icecream import ic
from tqdm import tqdm


class SupInfo(nn.Module):
    # Support:
    # Supervision of color and distance.
    # Output occlusion info given points
    def __init__(self):
        super().__init__()

    def get_occulsion_mask(self, pts):
        raise NotImplementedError


class PanoSupInfo(SupInfo):
    def __init__(self, pose, mask, color_map, distance_map, normal_map=None, factor=1):
        '''
        :param pose: [4, 4]
        :param mask: [H, W, 1] or [H, W]
        :param color_map: [H, W, 3]
        :param distance_map: [H, W, 1] or [H, W]
        :param normal_map: [H, W, 3]
        :param factor:
        '''
        super().__init__()
        device = color_map.device
        height, width, _ = color_map.shape

        if distance_map is None:
            distance_map = torch.ones([height, width, 1], device=color_map.device)
        else:
            distance_map = distance_map.squeeze()[..., None]
        mask = mask.squeeze()[..., None]

        has_normal_map = True
        if normal_map is None:
            has_normal_map = False
            normal_map = torch.zeros([height, width, 3], device=color_map.device)

        assert color_map.shape[-1] == 3 and distance_map.shape[-1] == 1
        self.register_buffer('pose', pose)

        if factor != 1:
            factor = int(factor)
            height = height // factor
            width = width // factor

            color_map = cv.resize(color_map.cpu().numpy(), (width, height), interpolation=cv.INTER_AREA)
            distance_map = cv.resize(distance_map.cpu().numpy(), (width, height), interpolation=cv.INTER_AREA)
            normal_map = cv.resize(normal_map.cpu().numpy(), (width, height), interpolation=cv.INTER_AREA)

            color_map = torch.from_numpy(color_map).to(device)
            distance_map = torch.from_numpy(distance_map).to(device)
            normal_map = torch.from_numpy(normal_map).to(device)

        self.height, self.width = height, width
        if mask is None:
            mask = torch.ones_like(distance_map, dtype=torch.bool)
        else:
            mask = (mask > .5)

        mask = mask & (distance_map > 1e-5)
        self.register_buffer('mask_raw', mask.clone())

        x_laplacian = kornia.filters.laplacian(distance_map[None].permute(0, 3, 1, 2), kernel_size=3)
        edge_mask = (x_laplacian.abs() < 0.01).float()
        edge_mask = erosion(edge_mask, kernel=torch.ones(3, 3))
        edge_mask = dilation(edge_mask, kernel=torch.ones(3, 3))

        mask = mask & (edge_mask[0] > .5).permute(1, 2, 0)

        if has_normal_map:
            pano_dirs = -img_coord_to_pano_direction(img_coord_from_hw(height, width))
            normal_cos = (pano_dirs * normal_map).sum(-1, True).clip(0., 1.)
            mask = mask & (normal_cos > 0.15)

        self.register_buffer('color_map', color_map)
        self.register_buffer('distance_map', distance_map)
        self.register_buffer('normal_map', normal_map)
        self.register_buffer('mask', mask)

        # self.sup_colors = None
        # self.sup_distances = None
        # self.sup_normals = None
        # self.sup_rays = None
        self.update_sup_info()

    def update_sup_info(self):
        pose, height, width = self.pose, self.height, self.width
        img_coords = torch.meshgrid(torch.linspace(.5 / height, 1. - .5 / height, height),
                                    torch.linspace(.5 / width,  1. - .5 / width,  width),
                                    indexing='ij')
        dirs = img_coord_to_pano_direction(torch.stack(img_coords, -1))
        dirs = apply_rot(dirs, self.pose[:3, :3])

        positions = self.pose[None, None, :3, 3].repeat(height, width, 1)

        sup_indices = torch.where(self.mask[..., 0] > 0.5)
        sup_colors, sup_distances, sup_normals =\
            self.color_map[sup_indices], self.distance_map[sup_indices], self.normal_map[sup_indices]
        sup_dirs, sup_positions = dirs[sup_indices], positions[sup_indices]

        self.register_buffer('sup_colors', sup_colors)
        self.register_buffer('sup_distances', sup_distances)
        self.register_buffer('sup_normals', sup_normals)
        self.register_buffer('sup_dirs', sup_dirs)
        self.register_buffer('sup_positions', sup_positions)

        self.sup_rays = Rays(sup_positions, sup_dirs)

    def get_pers_patch_data(self, res, fov, from_masked_region=True):
        local_pers_rays_d = cam_rays_cam_space(res, res, fovy=fov)
        if not from_masked_region:
            to_vec = torch.randn(3)
            to_vec /= torch.linalg.norm(to_vec, 2, -1, True)
        else:
            coords = torch.stack([
                (self.sup_indices[0] + .5) / self.height,
                (self.sup_indices[1] + .5) / self.width
            ], dim=-1)
            dirs = img_coord_to_pano_direction(coords)
            idx = np.random.randint(0, len(dirs))
            to_vec = dirs[idx]
            to_vec /= torch.linalg.norm(to_vec, 2, -1, True)

        local_rots = look_at(to_vec[None])[0]
        # local_pers_rays_d = torch.matmul(local_rots[None, None, :, :], local_pers_rays_d[:, :, :, None])[..., 0]
        local_pers_rays_d = apply_rot(local_pers_rays_d, local_rots)
        img_coords = direction_to_img_coord(local_pers_rays_d)
        sampled_coords = img_coord_to_sample_coord(img_coords)
        colors = F.grid_sample(self.color_map[None].permute(0, 3, 1, 2), sampled_coords[None])[0].permute(1, 2, 0)
        rays = Rays(torch.zeros_like(local_pers_rays_d) + self.pose[:3, 3], apply_rot(local_pers_rays_d, self.pose[:3, :3]))
        return { 'colors': colors, 'rays': rays }

    def set_after_reload(self):
        self.sup_rays = Rays(self.sup_positions, self.sup_dirs)


class SupInfoPool:
    def __init__(self):
        super().__init__()
        self.sup_infos = list()

        self.all_sup_colors = None
        self.all_sup_rays = None
        self.all_sup_distances = None
        self.all_sup_normals = None

    def register_sup_info(self, pose, mask, rgb, distance, normal=None):
        self.sup_infos.append(PanoSupInfo(pose=pose, mask=mask, color_map=rgb, distance_map=distance, normal_map=normal))
        if self.all_sup_colors is None:
            self.all_sup_colors = self.sup_infos[0].sup_colors
            self.all_sup_rays = self.sup_infos[0].sup_rays
            self.all_sup_distances = self.sup_infos[0].sup_distances
            self.all_sup_normals = self.sup_infos[0].sup_normals
        else:
            self.all_sup_colors = torch.cat([self.all_sup_colors, self.sup_infos[-1].sup_colors], 0)
            self.all_sup_rays = cat_rays([self.all_sup_rays, self.sup_infos[-1].sup_rays])
            self.all_sup_distances = torch.cat([self.all_sup_distances, self.sup_infos[-1].sup_distances], 0)
            self.all_sup_normals = torch.cat([self.all_sup_normals, self.sup_infos[-1].sup_normals], 0)

    def register_sup_info_by_pts(self, pose, colors, pts):
        H, W, _ = pts.shape
        pts = pts - pose[:3, 3][None, None, :]
        pts = apply_rot(pts, torch.linalg.inv(pose[:3, :3]))
        new_d = torch.linalg.norm(pts, 2, -1, False)

        # normalize: /depth
        new_dirs = pts / new_d.reshape(H, W, 1)
        new_depth = torch.zeros(new_d.shape)
        img = torch.zeros(pts.shape)

        # backward: 3d coordinate to pano image
        # [x, y, z] = new_coord[..., 0], new_coord[..., 1], new_coord[..., 2]

        idx = torch.where(new_d > 0)

        # theta: horizontal angle, phi: vertical angle
        # theta = torch.zeros(y.shape)
        # phi = np.zeros(y.shape)
        # x1 = np.zeros(z.shape)
        # y1 = np.zeros(z.shape)

        img_coord = direction_to_img_coord(new_dirs)
        x = torch.floor(img_coord[..., 0] * H).to(torch.int64)
        y = torch.floor(img_coord[..., 1] * W).to(torch.int64)

        # Mask out
        mask = (new_d > 0) & (H > x) & (x > 0) & (W > y) & (y > 0)
        x = x[torch.where(mask)]
        y = y[torch.where(mask)]
        new_d = new_d[mask]
        colors = colors[mask]
        reorder = torch.argsort(-new_d)
        x = x[reorder]
        y = y[reorder]
        new_d = new_d[reorder]
        colors = colors[reorder]
        # Assign
        new_depth[x, y] = new_d
        img[x, y] = colors

        depth_margin = 4
        for i in tqdm(range(depth_margin, H, 2)):
            for j in range(depth_margin, W, 2):
                x_l = max(0, i - depth_margin)
                x_r = min(H, i + depth_margin)
                y_l, y_r = max(0, j - depth_margin), min(W, j + depth_margin)

                index = torch.where(new_depth[x_l:x_r, y_l:y_r] > 0)
                if len(index[0]) == 0: continue
                mean = torch.median(new_depth[x_l:x_r, y_l:y_r][index])  # median
                target_index = torch.where(new_depth[x_l:x_r, y_l:y_r] > mean * 1.3)

                if len(target_index[0]) > depth_margin ** 2 // 2:
                    # reduce block size
                    img[x_l:x_r, y_l:y_r][target_index] = 0  # np.array([255.0, 0.0, 0.0])
                    new_depth[x_l:x_r, y_l:y_r][target_index] = 0

        mask = (new_depth != 0).float()

        self.register_sup_info(pose, mask, img, distance=None, normal=None)


    def rand_ray_color_data(self, batch_size, pano_idx=-1, rand_mode='by_all_pixels'):
        assert rand_mode in ['by_all_pixels', 'only_first', 'only_last']

        if rand_mode == 'by_all_pixels':
            sup_colors = self.all_sup_colors
            sup_rays = self.all_sup_rays
            sup_distances = self.all_sup_distances
            sup_normals = self.all_sup_normals
            assert len(sup_colors) == len(sup_rays) == len(sup_distances)
        elif rand_mode == 'only_first':
            sup_colors = self.sup_infos[0].sup_colors
            sup_rays = self.sup_infos[0].sup_rays
            sup_distances = self.sup_infos[0].sup_distances
            sup_normals = self.sup_infos[0].sup_normals
        else:
            sup_colors = self.sup_infos[-1].sup_colors
            sup_rays = self.sup_infos[-1].sup_rays
            sup_distances = self.sup_infos[-1].sup_distances
            sup_normals = self.sup_infos[-1].sup_normals

        max_ray_idx = len(sup_colors)
        indices = torch.randint(0, max_ray_idx, (batch_size,))

        return sup_rays[indices], sup_colors[indices], sup_distances[indices], sup_normals[indices]

    def geo_check(self, rays, distances):
        '''
        :param rays:
        :param distances:
        :return: mask, 1 -> OK! 0 -> conflict!
        '''
        pts = rays.o + rays.d * distances.squeeze()[..., None]
        height, width = pts.shape[:2]
        mask = torch.ones([height, width, 1])

        for pano_idx in range(len(self.sup_infos)):
            sup_info = self.sup_infos[pano_idx]
            sup_distance_map = sup_info.distance_map * sup_info.mask.float()

            new_dirs = apply_rot(pts - sup_info.pose[:3, 3], sup_info.pose[:3, :3].T)
            new_distances = torch.linalg.norm(new_dirs, 2, -1, True)
            new_dirs /= new_distances
            proj_coords = direction_to_img_coord(new_dirs)
            sample_coords = img_coord_to_sample_coord(proj_coords)
            proj_distances = F.grid_sample(sup_distance_map[None].permute(0, 3, 1, 2), sample_coords[None],
                                           padding_mode='border')
            proj_distances = proj_distances[0].permute(1, 2, 0)
            # bias = (proj_distances - new_distances).clip(0., None) / (new_distances / 256.0).clip(2.5e-3, None)
            # bias = torch.exp(-bias * bias * .5)
            # bias = ((proj_distances - new_distances).clip(0., None) < (1. / 512.)).float()
            bias = (proj_distances < new_distances).float()
            mask.clamp_(min=None, max=bias)

        l_size = (9, 9)
        s_size = (3, 3)

        kernel_l = cv.getStructuringElement(cv.MORPH_ELLIPSE, l_size)
        kernel_s = cv.getStructuringElement(cv.MORPH_ELLIPSE, s_size)
        kernel_l = torch.from_numpy(kernel_l).to(torch.float32).to(mask.device)
        kernel_s = torch.from_numpy(kernel_s).to(torch.float32).to(mask.device)

        mask = (mask[None, :, :, :] > 0.5).float()
        mask = mask.permute(0, 3, 1, 2)
        mask = dilation(mask, kernel=kernel_s)
        mask = erosion(mask, kernel=kernel_l)

        return mask.permute(0, 2, 3, 1).contiguous().squeeze()

    def gen_occ_grid(self, res):
        rays_o, rays_d = self.all_sup_rays.collapse()
        dis = self.all_sup_distances
        pts = rays_o + rays_d * dis.squeeze()[..., None]
        occ_grid = torch.zeros([res * res * res], dtype=torch.uint8)
        shift = 1. / res
        xx, yy, zz = torch.meshgrid(
            torch.linspace(-shift, shift, 3),
            torch.linspace(-shift, shift, 3),
            torch.linspace(-shift, shift, 3)
        )
        shift_xyzs = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
        for shift_xyz in shift_xyzs:
            shifted = shift_xyz[None, :] + pts
            shifted = ((shifted.clip(-0.999, 0.999) * .5 + .5) * res).to(torch.int64)
            shifted_idx = shifted[..., 0] * res * res + shifted[..., 1] * res + shifted[..., 2]
            assert shifted_idx.max().item() < res * res * res and shifted_idx.min().item() >= 0
            occ_grid[shifted_idx] = 1

        valid_idx = torch.where(occ_grid > 0)[0]
        valid_x = valid_idx // (res * res)
        valid_y = (valid_idx // res) % res
        valid_z = valid_idx % res
        valid_pts = torch.stack([valid_x, valid_y, valid_z], -1)
        valid_pts = (valid_pts / float(res) - .5) * 2.

        return occ_grid, valid_pts

    def state_dict(self):
        ret = dict()
        ret['n_sup_infos'] = len(self.sup_infos)
        for i in range(len(self.sup_infos)):
            ret['sup_info_{}_height'] = self.sup_infos[i].height
            ret['sup_info_{}_width'] = self.sup_infos[i].width
            ret['sup_info_{}'.format(i)] = self.sup_infos[i].state_dict()

        return ret

    def load_state_dict(self, state_dict):
        n_sup_infos = state_dict['n_sup_infos']
        for i in range(n_sup_infos):
            height = state_dict['sup_info_{}_height']
            width  = state_dict['sup_info_{}_width']
            sup_info = PanoSupInfo(pose=torch.eye(4),
                                   mask=torch.ones([height, width, 1]),
                                   color_map=torch.ones([height, width, 3]),
                                   distance_map=torch.ones([height, width, 1]),
                                   normal_map=torch.ones([height, width, 3]))

            sup_info.set_after_reload()
            self.sup_infos.append(sup_info)

        self.all_sup_colors = torch.cat([info.sup_colors for info in self.sup_infos], 0)
        self.all_sup_rays = cat_rays([info.sup_rays for info in self.sup_infos])
        self.all_sup_distances = torch.cat([info.sup_distances for info in self.sup_infos], 0)
        self.all_sup_normals = torch.cat([info.sup_normals for info in self.sup_infos], 0)

