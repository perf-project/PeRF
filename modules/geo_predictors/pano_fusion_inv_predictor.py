import torch
import torch.nn.functional as F
import cv2 as cv
from tqdm import tqdm
from kornia.filters import laplacian, sobel, gaussian_blur2d

import numpy as np
from trimesh.creation import icosphere as IcoSphere
from scipy.spatial.transform import Rotation

from .geo_predictor import GeoPredictor
from .omnidata_predictor import OmnidataPredictor
# from .omnidata_normal_predictor import OmnidataNormalPredictor
# from .iron_predictor import IronPredictor

from utils.geo_utils import panorama_to_pers_directions
from utils.camera_utils import img_coord_to_sample_coord, img_to_pano_coord,\
    direction_to_img_coord, img_coord_to_pano_direction, direction_to_pers_img_coord, pers_depth_to_normal
from utils.utils import write_image
from utils.debug_utils import printarr


def scale_unit(x):
    return (x - x.min()) / (x.max() - x.min())


class PanoFusionInvPredictor(GeoPredictor):
    def __init__(self):
        super().__init__()
        # self.depth_predictor = BiNormalPredictor()
        self.depth_predictor = OmnidataPredictor()
        # self.depth_predictor = IronPredictor()

    def inpaint_distance(self, img, ref_distance, mask, gen_res=384):
        '''
        Do not support batch operation - can only inpaint one image at a time.
        :param img: [H, W, 3]
        :param ref_distance: [H, W] or [H, W, 1]
        :param mask: [H, W] or [H, W, 1], value 1 if need to be inpainted otherwise 0
        :return: inpainted distance [H, W]
        '''

        device = img.device
        img = img.clone().squeeze().permute(2, 0, 1)     # [3, H, W]
        mask = mask.clone().squeeze()[None]    # [1, H, W]
        ref_distance = ref_distance.clone().squeeze()[None]   # [1, H, W]

        pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = panorama_to_pers_directions(gen_res=gen_res, ratio=1.0)
        fx = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(right_vecs, 2, -1, True) * gen_res * .5
        fy = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(down_vecs, 2, -1, True) * gen_res * .5
        cx = torch.ones_like(fx) * gen_res * .5
        cy = torch.ones_like(fy) * gen_res * .5

        pers_dirs = pers_dirs.to(device)
        pers_ratios = pers_ratios.to(device)
        to_vecs = to_vecs.to(device)
        down_vecs = down_vecs.to(device)
        right_vecs = right_vecs.to(device)

        n_pers = len(pers_dirs)
        img_coords = direction_to_img_coord(pers_dirs)
        sample_coords = img_coord_to_sample_coord(img_coords)

        _, pano_height, pano_width = img.shape
        pano_img_coords = torch.meshgrid(torch.linspace(.5 / pano_height, 1. - .5 / pano_height, pano_height),
                                         torch.linspace(.5 / pano_width, 1. - .5 / pano_width, pano_width),
                                         indexing='ij')
        pano_img_coords = torch.stack(list(pano_img_coords), dim=-1)
        pano_coord = img_to_pano_coord(pano_img_coords)
        distortion_weights = torch.cos(pano_coord[:, :, 0])
        pano_dirs = img_coord_to_pano_direction(pano_img_coords)
        pers_imgs = F.grid_sample(img[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') # [n_pers, 3, gen_res, gen_res]
        pred_depths_raw = []

        for i in range(n_pers):
            with torch.no_grad():
                intri = {
                    'fx': fx[i].item(),
                    'fy': fy[i].item(),
                    'cx': cx[i].item(),
                    'cy': cy[i].item()
                }
                pred_depth = self.depth_predictor.predict_depth(pers_imgs[i: i+1], intri=intri).cuda().clip(0., None)
                # pred_depth = pred_depth * pers_ratios[i].permute(2, 0, 1)
                pred_depth = pred_depth / (pred_depth.mean() + 1e-5)
                pred_depths_raw.append(pred_depth)

        pred_depths_raw = torch.cat(pred_depths_raw, dim=0)

        proj_coords = []
        proj_masks = []

        for i in range(n_pers):
            proj_coord, proj_mask = direction_to_pers_img_coord(pano_dirs, to_vecs[i], down_vecs[i], right_vecs[i])
            proj_coord = img_coord_to_sample_coord(proj_coord)
            proj_coords.append(proj_coord[None])
            proj_masks.append(proj_mask[None])

        proj_coords = torch.cat(proj_coords, dim=0)   # [n_pers, pano_height, pano_width, 2]
        proj_masks = torch.cat(proj_masks, dim=0)     # [n_pers, pano_height, pano_width, 1]
        proj_coords = proj_coords.clip(-1., 1.) # not necessary
        proj_masks = proj_masks.permute(0, 3, 1, 2)

        scale_params = torch.zeros([n_pers], requires_grad=True)
        bias_params_global = torch.zeros([n_pers, 3], requires_grad=True)
        bias_params = torch.zeros([n_pers, 1, gen_res, gen_res], requires_grad=True)
        pano_distance_params = torch.zeros([1, pano_height, pano_width], requires_grad=True)

        all_iter_steps = 1000
        lr_alpha = 1e-2
        init_lr = 1e-1

        optimizer_a = torch.optim.Adam([scale_params, pano_distance_params], lr=init_lr)
        optimizer_b = torch.optim.Adam([scale_params, bias_params, pano_distance_params], lr=init_lr)
        def param_to_distance(x):
            return F.softplus(x) + 1e-3

        # for iter_step in tqdm(range(all_iter_steps)):
        for iter_step in tqdm(range(all_iter_steps)):
            phase = 'scale_only' if iter_step * 2 < all_iter_steps else 'all'
            optimizer = optimizer_a if phase == 'scale_only' else optimizer_b

            progress = iter_step / all_iter_steps
            lr_ratio = (np.cos(progress * np.pi) + 1.) * (1. - lr_alpha) + lr_alpha
            for g in optimizer.param_groups:
                g['lr'] = init_lr * lr_ratio

            ii, jj = torch.meshgrid(torch.linspace(-1., 1., gen_res),
                                    torch.linspace(-1., 1., gen_res))

            pano_distance = param_to_distance(pano_distance_params)
            pano_distance = pano_distance * mask + ref_distance * (1. - mask)
            scales = F.softplus(scale_params)
            # bias = ii[None, None, :, :] * bias_params_global[:, 0:1, None, None] +\
            #        jj[None, None, :, :] * bias_params_global[:, 1:2, None, None] +\
            #        bias_params_global[:, 2:3, None, None] + bias_params
            if phase == 'scale_only':
                bias = 0.
            else:
                bias = bias_params
            pred_distances = (pred_depths_raw + bias) * pers_ratios.permute(0, 3, 1, 2) * scales[:, None, None, None]
            pred_distances = pred_distances.clip(1e-5, None)
            pred_distances = F.grid_sample(pred_distances, proj_coords, padding_mode='border')
            align_error = (pred_distances - pano_distance[None]) * proj_masks
            # align_loss = F.smooth_l1_loss(align_error, torch.zeros_like(align_error), beta=1e-2, reduction='sum') / proj_masks.sum()
            align_error = F.smooth_l1_loss(align_error, torch.zeros_like(align_error), beta=1e-1, reduction='none')
            weight_masks = proj_masks * distortion_weights[None, None]
            align_loss = (align_error * weight_masks).sum() / weight_masks.sum()

            bias_tv_loss = F.smooth_l1_loss(bias_params[:, :, 1:, :], bias_params[:, :, :-1, :], beta=1e-1) + \
                           F.smooth_l1_loss(bias_params[:, :, :, 1:], bias_params[:, :, :, :-1], beta=1e-1)


            reg_loss = (scales.mean() - 1.)**2
            loss = align_loss + bias_tv_loss * 5 + reg_loss * 1e-2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pano_distance = param_to_distance(pano_distance_params).detach()
        pano_distance = pano_distance * mask + ref_distance * (1. - mask)
        return pano_distance.squeeze()
