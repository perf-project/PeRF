import os
import itertools
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2 as cv
from kornia.morphology import erosion, dilation

from .scene import Scene
from .nerf_renderer import NeRFPropRenderer, NeRFOCCRenderer
from modules.fields.ngp_nerf import NGPNeRF
from modules.fields.ngp_nerf import NGPDensityField


from utils.camera_utils import *
from modules.dataset.sup_info import SupInfoPool

from torch_efficient_distloss import flatten_eff_distloss, eff_distloss
from nerfacc.estimators.prop_net import PropNetEstimator
from nerfacc.estimators.occ_grid import OccGridEstimator


class NeRFScene(Scene):
    def __init__(self,
                 base_exp_dir,
                 train_conf,
                 estimator_type,
                 renderer_conf):
        super().__init__()
        self.aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        self.base_exp_dir = base_exp_dir
        self.writer = SummaryWriter(log_dir=pjoin(base_exp_dir, 'ts_log'))
        self.train_conf = train_conf
        self.nerf = NGPNeRF(aabb=self.aabb)
        # self.optimizer = torch.optim.Adam(self.nerf.parameters(), lr=self.train_conf.optimizer.init_lr)

        if estimator_type == 'prop':
            # proposal networks
            self.prop_networks = [
                NGPDensityField(
                    aabb=self.aabb,
                    unbounded=False,
                    n_levels=5,
                    max_resolution=128,
                ),
                NGPDensityField(
                    aabb=self.aabb,
                    unbounded=False,
                    n_levels=5,
                    max_resolution=256,
                ),
            ]
            self.prop_optimizer = torch.optim.Adam(
                itertools.chain(*[p.parameters() for p in self.prop_networks]),
                lr=self.train_conf.prop_optimizer.init_lr,
                eps=1e-15,
                betas=(0.9, 0.99),
                weight_decay=1e-6
            )
            self.estimator = PropNetEstimator(self.prop_optimizer, None).cuda()
            self.renderer = NeRFPropRenderer(**renderer_conf)
        else:
            self.estimator = OccGridEstimator(roi_aabb=self.aabb, resolution=256, levels=1).cuda()
            self.renderer = NeRFOCCRenderer(**renderer_conf)

        self.global_iter_step_geo = 0
        self.global_iter_step_app = 0

    @torch.no_grad()
    def render(self, rays: Rays, query_keys=('rgb',), sampling_requires_grad=False):
        last_train = self.nerf.training
        self.set_eval()
        rays_o, rays_d = rays.collapse()
        pre_shape = list(rays_o.shape[:-1])
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        ret = dict()
        for query_key in query_keys:
            ret[query_key] = []

        batch_size = 32768
        rays_o_batches = rays_o.split(batch_size)
        rays_d_batches = rays_d.split(batch_size)
        for rays_o_batch, rays_d_batch in zip(rays_o_batches, rays_d_batches):
            cur_render_result = self.render_once(Rays(rays_o_batch, rays_d_batch), query_keys, sampling_requires_grad=sampling_requires_grad)
            for query_key in query_keys:
                ret[query_key] += cur_render_result[query_key]

        for query_key in query_keys:
            ret[query_key] = torch.concatenate(ret[query_key], dim=0).reshape(pre_shape + [-1])

        if last_train:
            self.set_train()
        return ret

    def render_once(self, rays: Rays, query_keys=('rgb',), sampling_requires_grad=False, geo_inference=False, app_inference=False):
        rays = self.to_bounded_rays(rays)
        rays_o, rays_d, near, far = rays.collapse()
        assert len(rays_o.shape) == 2

        if isinstance(self.renderer, NeRFPropRenderer):
            render_result = self.renderer.render(self.nerf, self.prop_networks, self.estimator,
                                                 rays_o, rays_d, near, far,
                                                 sampling_requires_grad=sampling_requires_grad)
        else:
            render_result = self.renderer.render(self.nerf, self.estimator,
                                                 rays_o, rays_d, near, far,
                                                 geo_inference=geo_inference,
                                                 app_inference=app_inference)
        if (render_result is None) or (not render_result['is_valid']):
            return render_result

        ret_dict = dict()
        query_keys_ex = list(query_keys) + ['is_valid']
        for query_key in query_keys_ex:
            ret_dict[query_key] = render_result[query_key]

        return ret_dict

    def fit(self, sup_pool: SupInfoPool):
        if len(sup_pool.sup_infos) == 1:
            self.train_one_episode(sup_pool,
                                   self.train_conf.raw_phase_iter_geo,
                                   self.train_conf.raw_phase_iter_app,
                                   pixel_sup_rand_mode='by_all_pixels')
        else:
            self.train_one_episode(sup_pool,
                                   self.train_conf.raw_phase_iter_geo,
                                   self.train_conf.raw_phase_iter_app,
                                   pixel_sup_rand_mode='by_all_pixels')

    def train_one_episode(self, sup_pool: SupInfoPool, geo_res_iters, app_res_iters, pixel_sup_rand_mode):
        self.set_train()
        grad_scaler = torch.cuda.amp.GradScaler(2**7)
        pre_grid = None
        occ_res = 256

        if isinstance(self.estimator, OccGridEstimator):
            self.estimator = OccGridEstimator(roi_aabb=self.aabb, resolution=256, levels=1).cuda()
            pre_grid, occ_pts = sup_pool.gen_occ_grid(res=occ_res)

        def occ_eval_fn(x):
            if pre_grid is None:
                density = self.nerf.query_density(x)
                return density * 5e-3
            else:
                x = (x.clip(-0.999, 0.999) * .5 + .5) * occ_res
                x = x.to(torch.int64)
                idx = x[..., 0] * occ_res * occ_res +\
                      x[..., 1] * occ_res +\
                      x[..., 2]
                return pre_grid[idx].float()

        if pre_grid is not None:
            for i in tqdm(range(256)):
                self.estimator.update_every_n_steps(
                    step=i,
                    occ_eval_fn=occ_eval_fn,
                    occ_thre=1e-2,
                    ema_decay=0.1,
                    warmup_steps=256,
                    n=1,
                )

        self.nerf.reset_geo() #  = NGPNeRF(aabb=self.aabb)
        geo_optimizer = torch.optim.Adam(self.nerf.geo_mlp.parameters(), lr=self.train_conf.geo_optimizer.init_lr)

        for iter_i in tqdm(range(geo_res_iters)):
            self.update_lr(geo_optimizer, self.train_conf.geo_optimizer, iter_i / geo_res_iters)
            if hasattr(self, 'prop_optimizer'):
                self.update_lr(self.prop_optimizer, self.train_conf.prop_optimizer, iter_i / geo_res_iters)

            self.train_one_step_geo(geo_optimizer, sup_pool, pixel_sup_rand_mode, progress=iter_i / app_res_iters, grad_scaler=grad_scaler)

        app_optimizer = torch.optim.Adam(self.nerf.app_mlp.parameters(), lr=self.train_conf.app_optimizer.init_lr)

        for iter_i in tqdm(range(app_res_iters)):
            self.update_lr(app_optimizer, self.train_conf.app_optimizer, iter_i / app_res_iters)
            self.train_one_step_app(app_optimizer, sup_pool, pixel_sup_rand_mode, progress=iter_i / app_res_iters, grad_scaler=grad_scaler)

    def train_one_step_geo(self, optimizer, sup_pool: SupInfoPool, pixel_sup_rand_mode, progress, grad_scaler=None):
        train_conf = self.train_conf
        loss = 0.
        eps = 1e-7
        optimizer.zero_grad()

        batch_size = train_conf.pixel_loss_batch_size
        rays, gt_colors, gt_depths, gt_normals = sup_pool.rand_ray_color_data(batch_size,
                                                                              rand_mode=pixel_sup_rand_mode)

        if isinstance(self.renderer, NeRFOCCRenderer):
            query_keys = ['rgb', 'distance', 'weights', 't_starts', 't_ends', 'trans', 'ray_indices']
        else:
            query_keys = ['rgb', 'distance', 'weights', 't_starts', 't_ends', 'trans']
        render_result = self.render_once(rays, query_keys=query_keys,
                                         sampling_requires_grad=self.need_to_update_occ(self.global_iter_step_geo),
                                         app_inference=True)

        if (render_result is None) or (not render_result['is_valid']):
            self.global_iter_step_geo += 1
            return

        depth_loss_weight = train_conf.depth_loss_weight
        if depth_loss_weight > eps:
            pred_depths = render_result['distance']
            depth_loss = F.smooth_l1_loss(pred_depths, gt_depths, beta=1e-2, reduction='mean')
            loss = loss + depth_loss * depth_loss_weight
            self.writer.add_scalar('nerf_loss/depth_loss', depth_loss, self.global_iter_step_geo)

        # distortion loss
        distortion_loss_weight = train_conf.distortion_loss_weight
        if distortion_loss_weight > eps:
            if isinstance(self.renderer, NeRFPropRenderer):
                weights = render_result['weights']
                mid_dis = (render_result['t_ends'] + render_result['t_starts']) * .5
                sec_lens = render_result['t_ends'] - render_result['t_starts']
                dist_loss = eff_distloss(weights, mid_dis, sec_lens)
                ratio = progress
                loss = loss + dist_loss * distortion_loss_weight * ratio
            else:
                weights = render_result['weights']
                mid_dis = (render_result['t_ends'] + render_result['t_starts']) * .5
                sec_lens = render_result['t_ends'] - render_result['t_starts']
                ray_indices = render_result['ray_indices']
                dist_loss = flatten_eff_distloss(weights, mid_dis, sec_lens, ray_indices)
                # if progress < 0.0:
                #     ratio = 0.
                # else:
                #     local_progress = (progress - 0.1) / 0.9
                ratio = np.min([progress * 2., 1])
                loss = loss + dist_loss * distortion_loss_weight * ratio

            self.writer.add_scalar('nerf_loss/dist_loss', dist_loss, self.global_iter_step_geo)

        density_loss_weight = train_conf.density_loss_weight
        if density_loss_weight > eps:
            rand_pts = (torch.rand(8192, 3) * 2. - 1.) * 0.99
            density = self.nerf.query_density(rand_pts)
            density_loss = density.mean()
            loss = loss + density_loss * density_loss_weight

            self.writer.add_scalar('nerf_loss/density_loss', density_loss, self.global_iter_step_geo)

        if grad_scaler is None:
            loss.backward()
        else:
            grad_scaler.scale(loss).backward()
        optimizer.step()

        self.writer.add_scalar('others/lr_geo', optimizer.param_groups[0]['lr'], self.global_iter_step_geo)

        self.global_iter_step_geo += 1

    def train_one_step_app(self, optimizer, sup_pool: SupInfoPool, pixel_sup_rand_mode, progress, grad_scaler=None):
        train_conf = self.train_conf
        loss = 0.
        eps = 1e-7
        optimizer.zero_grad()

        batch_size = train_conf.pixel_loss_batch_size
        rays, gt_colors, gt_depths, gt_normals = sup_pool.rand_ray_color_data(batch_size,
                                                                              rand_mode=pixel_sup_rand_mode)

        if isinstance(self.renderer, NeRFOCCRenderer):
            query_keys = ['rgb', 'distance', 'weights', 't_starts', 't_ends', 'trans', 'ray_indices']
        else:
            query_keys = ['rgb', 'distance', 'weights', 't_starts', 't_ends', 'trans']
        render_result = self.render_once(rays, query_keys=query_keys,
                                         sampling_requires_grad=self.need_to_update_occ(self.global_iter_step_app),
                                         geo_inference=True)

        if (render_result is None) or (not render_result['is_valid']):
            self.global_iter_step_app += 1
            return

        color_loss_weight = train_conf.color_loss_weight
        if color_loss_weight > eps:
            pred_colors = render_result['rgb']

            color_loss = F.smooth_l1_loss(pred_colors, gt_colors, beta=5e-2, reduction='mean')
            self.writer.add_scalar('nerf_loss/color_loss', color_loss, self.global_iter_step_app)
            loss = loss + color_loss * color_loss_weight

        if grad_scaler is None:
            loss.backward()
        else:
            grad_scaler.scale(loss).backward()
        optimizer.step()

        self.writer.add_scalar('others/lr_app', optimizer.param_groups[0]['lr'], self.global_iter_step_app)

        self.global_iter_step_app += 1


    def update_lr(self, optimizer, optim_conf, progress):
        # Update scene optimizer lr
        if progress < optim_conf.peak_at:
            local_progress = progress / optim_conf.peak_at
            lr = optim_conf.peak_lr * local_progress + optim_conf.init_lr * (1. - local_progress)
        else:
            local_progress = (progress - optim_conf.peak_at) / (1. - optim_conf.peak_at)
            lr_factor = (np.cos(local_progress * np.pi) + 1.) * .5 * (1. - optim_conf.lr_alpha) + optim_conf.lr_alpha
            lr = optim_conf.peak_lr * lr_factor

        for p in optimizer.param_groups:
            p['lr'] = lr

    def to_bounded_rays(self, rays):
        rays_o = rays.o
        rays_d = rays.d
        batch_size = len(rays_o)
        near = 1e-2 * torch.ones([batch_size, 1])
        far  = 1 * torch.ones([batch_size, 1])
        return BoundedRays(rays_o, rays_d, near, far)

    def get_pano_visibility_mask(self, sup_pool, rays):
        distance = self.render(rays, query_keys=['distance'])['distance'].squeeze()
        height, width = distance.shape
        pts = rays.o + rays.d * distance[..., None]

        mask = torch.zeros([height, width, 1])

        for pano_idx in range(len(sup_pool.sup_infos)):
            sup_info = sup_pool.sup_infos[pano_idx]
            sup_distance_map = sup_info.distance_map * sup_info.mask
            new_dirs = apply_rot(pts - sup_info.pose[:3, 3], sup_info.pose[:3, :3].T)
            new_distances = torch.linalg.norm(new_dirs, 2, -1, True)
            new_dirs /= new_distances
            proj_coords = direction_to_img_coord(new_dirs)
            sample_coords = img_coord_to_sample_coord(proj_coords)
            proj_distances = F.grid_sample(sup_distance_map[None].permute(0, 3, 1, 2), sample_coords[None],
                                        padding_mode='border')
            proj_distances = proj_distances[0].permute(1, 2, 0)
            # bias = (new_distances - proj_distances).clip(0., None) / (new_distances / 256.0).clip(2.5e-3, None)
            # bias = torch.exp(-bias * bias * .5)
            # bias = (new_distances - proj_distances).clip(0., None) < (1 / 256.)
            bias = new_distances < proj_distances + 1 / 256.
            mask.clamp_(min=bias, max=None)

        l_size = (9, 9)
        s_size = (5, 5)

        kernel_l = cv.getStructuringElement(cv.MORPH_ELLIPSE, l_size)
        kernel_s = cv.getStructuringElement(cv.MORPH_ELLIPSE, s_size)
        kernel_l = torch.from_numpy(kernel_l).to(torch.float32).to(mask.device)
        kernel_s = torch.from_numpy(kernel_s).to(torch.float32).to(mask.device)

        mask = (mask[None, :, :, :] > 0.5).float()
        mask = mask.permute(0, 3, 1, 2)
        mask = dilation(mask, kernel=kernel_s)
        mask = erosion(mask, kernel=kernel_l)

        return mask.permute(0, 2, 3, 1).contiguous().squeeze()

    def need_to_update_occ(self, iter_step):
        # return (iter_step % 4 == 0) and self.nerf.training
        return True

    def update_occ_grid(self, iter_step, trans):
        if self.need_to_update_occ(iter_step):
            self.estimator._update(trans=trans.detach())

    def load_state_dict(self, state_dict):
        # self.optimizer.load_state_dict(state_dict['optimizer'])
        self.renderer.load_state_dict(state_dict['render'])
        self.nerf.load_state_dict(state_dict['nerf'])
        self.estimator.load_state_dict(state_dict['estimator'])

    def state_dict(self):
        return {
            # 'optimizer': self.optimizer.state_dict(),
            'render': self.renderer.state_dict(),
            'nerf': self.nerf.state_dict(),
            'estimator': self.estimator.state_dict(),
        }

    def set_train(self):
        self.nerf.train()
        self.estimator.train()
        if isinstance(self.estimator, PropNetEstimator):
            for p in self.prop_networks:
                p.train()
        self.renderer.train()

    def set_eval(self):
        self.nerf.eval()
        self.estimator.eval()
        if isinstance(self.estimator, PropNetEstimator):
            for p in self.prop_networks:
                p.eval()
        self.renderer.eval()
