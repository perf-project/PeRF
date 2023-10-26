import os
import cv2 as cv
import numpy as np
from shutil import copyfile
from os.path import join as pjoin

import trimesh
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf, DictConfig
from glob import glob
from tqdm import tqdm
import hydra

from icecream import ic

from modules.inpainters import PanoPersFusionInpainter
from modules.geo_predictors import PanoJointPredictor
from modules.dataset.dataset import WildDataset
from modules.dataset.sup_info import SupInfoPool
from modules.pose_sampler import CirclePoseSampler
from modules.pose_sampler import DenseTravelPoseSampler

from modules.scene.nerf import NeRFScene

from utils.utils import write_video, write_image, colorize_single_channel_image
from utils.debug_utils import printarr
from utils.camera_utils import *

backup_file_patterns = [
    './*.py', './modules/*.py', './modules/*/*.py', './utils/*.py,'
]


class CoreRunner:
    def __init__(self, conf, device=torch.device('cuda')):
        self.conf = conf
        self.device = device

        self.dataset = WildDataset(conf.dataset)

        self.base_dir = os.getcwd()
        self.base_exp_dir = conf.device.base_exp_dir
        self.exp_dir = pjoin(self.base_exp_dir, '{}_{}'.format(conf['dataset_class_name'], self.dataset.case_name), conf.exp_name)

        os.makedirs(self.exp_dir, exist_ok=True)

        # backup codes
        file_backup_dir = os.path.join(self.exp_dir, 'record/')
        os.makedirs(file_backup_dir, exist_ok=True)

        for file_pattern in backup_file_patterns:
            file_list = glob(os.path.join(self.base_dir, file_pattern))
            for file_name in file_list:
                new_file_name = file_name.replace(self.base_dir, file_backup_dir)
                os.makedirs(os.path.dirname(new_file_name), exist_ok=True)
                copyfile(file_name, new_file_name)

        resolved_conf = OmegaConf.to_container(conf, resolve=True)
        OmegaConf.save(resolved_conf, os.path.join(file_backup_dir, 'config.yaml'))
        OmegaConf.save(resolved_conf, './config.yaml')

        self.scene = globals()[conf.scene_class_name](self.exp_dir, **conf.scene)

        # Visualization
        write_image(pjoin(self.exp_dir, 'distance_vis.png'),
                    colorize_single_channel_image(
                        (self.dataset.ref_distance.min() + 1e-6) / (self.dataset.ref_distance + 1e-6)))
        if self.dataset.ref_normal is not None:
            write_image(pjoin(self.exp_dir, 'normal_vis.png'),
                        (self.dataset.ref_normal * .5 + .5) * 255.)

        self.pose_sampler = CirclePoseSampler(self.dataset.ref_distance,
                                              **conf.pose_sampler)

        self.sup_pool = SupInfoPool()
        self.sup_pool.register_sup_info(pose=torch.eye(4),
                                        mask=torch.ones([self.dataset.height, self.dataset.width]),
                                        rgb=self.dataset.image,
                                        distance=self.dataset.ref_distance,
                                        normal=self.dataset.ref_normal)
        self.sup_pool.gen_occ_grid(256)

        self.geo_predictor = PanoJointPredictor()
        self.inpainter = PanoPersFusionInpainter(inpainter_type=conf.pers_inpainter_type)

        self.phase = -1

        # Load checkpoint
        if conf.is_continue:
            self.load_checkpoint('ckpt.pth')

    def set_train(self):
        self.scene.set_train()

    def set_eval(self):
        self.scene.set_eval()

    def execute(self, mode):
        if mode == 'train':
            self.train()
        elif mode == 'render_dense':
            self.render_dense()

    def train(self, raw_only=False):
        ic('Train: begin')
        if self.phase < 0:
            self.set_train()

            self.scene.fit(self.sup_pool)

            render_result = self.scene.render(gen_pano_rays(torch.eye(4), 512, 1024), query_keys=['rgb', 'distance'])
            pano_rgb = render_result['rgb']

            pano_distances = (render_result['distance'].min() / render_result['distance']).squeeze()[..., None]
            write_image(pjoin(self.exp_dir, '1.png'), pano_rgb * 255.)
            write_image(pjoin(self.exp_dir, '1_distance.png'), colorize_single_channel_image(pano_distances))

            self.phase += 1
            self.save_checkpoint()

            if raw_only:
                return

        n_anchors = self.pose_sampler.n_anchors

        geo_check = True

        for anchor_idx in range(n_anchors):
            if anchor_idx < self.phase:
                continue

            pose = self.pose_sampler.sample_pose(anchor_idx)
            rays = gen_pano_rays(pose, self.dataset.height, self.dataset.width)

            visi_mask = self.scene.get_pano_visibility_mask(self.sup_pool, rays)  # 1 visible, 0 invisible
            with torch.no_grad():
                render_result = self.scene.render(rays, query_keys=['rgb', 'distance'])
            colors = render_result['rgb']
            distances = render_result['distance']

            inpaint_mask = 1. - visi_mask

            n_repeats = 1
            for sub_i in range(n_repeats):
                if visi_mask.min().item() > .5:
                    break

                colors, distances, normals = self.inpaint_new_panorama(sub_i, anchor_idx, colors=colors, distances=distances, mask=inpaint_mask)
                if geo_check:
                    # Perform geometric checking
                    conflict_mask = 1. - self.sup_pool.geo_check(rays, distances)    # 1 conflict, 0 not conflict
                    inpaint_mask = inpaint_mask * conflict_mask
                else:
                    inpaint_mask *= 0

                sub_i += 1

            vis_dir = pjoin(self.exp_dir, 'inpaint_vis', '{:0>4d}'.format(anchor_idx))
            os.makedirs(vis_dir, exist_ok=True)

            # Do not inpaint contents that are too close
            # inpaint_mask = torch.minimum(inpaint_mask, (distances > 0.05).float())
            inpaint_mask = torch.maximum(inpaint_mask, (distances < 0.1).float())
            inpaint_mask = torch.minimum(inpaint_mask, 1. - visi_mask)

            write_image(pjoin(vis_dir, 'final_mask.jpg'), inpaint_mask[..., None] * 255.)
            write_image(pjoin(vis_dir, 'final_masked.jpg'), (colors * (1. - inpaint_mask)[..., None]) * 255.)

            sup_mask = 1. - visi_mask
            sup_mask -= torch.minimum(sup_mask, inpaint_mask)

            self.sup_pool.register_sup_info(pose=pose, mask=sup_mask, rgb=colors, distance=distances, normal=normals)
            self.scene.fit(self.sup_pool)

            self.phase += 1
            self.save_checkpoint()

    def inpaint_new_panorama(self, phase, anchor_idx, colors, distances, mask):
        distances = distances.squeeze()[..., None]
        mask = mask.squeeze()[..., None]

        vis_dir = pjoin(self.exp_dir, 'inpaint_vis', '{:0>4d}'.format(anchor_idx))
        os.makedirs(vis_dir, exist_ok=True)

        write_image(pjoin(vis_dir, 'uninpainted_{}.jpg'.format(phase)), colors * 255.)
        write_image(pjoin(vis_dir, 'uninpainted_disparity_{}.jpg'.format(phase)), colorize_single_channel_image(distances.min() / distances))
        write_image(pjoin(vis_dir, 'mask_{}.jpg'.format(phase)), mask * 255.)
        write_image(pjoin(vis_dir, 'masked_{}.jpg'.format(phase)), (colors * (1. - mask)) * 255.)
        inpainted_distances = None
        inpainted_normals = None
        if self.conf.rgbd_inpaint:
            inpainted_img, inpainted_distances = self.inpainter.inpaint_rgbd(colors, distances, mask)
            write_image(pjoin(vis_dir, 'inpainted_{}.jpg'.format(phase)), inpainted_img * 255.)
        else:
            inpainted_img = self.inpainter.inpaint(colors, mask)
            inpainted_img = inpainted_img.cuda()
            write_image(pjoin(vis_dir, 'inpainted_{}.jpg'.format(phase)), inpainted_img * 255.)
            inpainted_distances, inpainted_normals = self.geo_predictor(inpainted_img,
                                                                        distances,
                                                                        mask=mask,
                                                                        reg_loss_weight=0.,
                                                                        normal_loss_weight=5e-2,
                                                                        normal_tv_loss_weight=5e-2)

        inpainted_distances = inpainted_distances.squeeze()

        height, width, _ = inpainted_img.shape
        write_image(pjoin(vis_dir, 'aligned_disparity_{}.jpg'.format(phase)),
                    colorize_single_channel_image(inpainted_distances.min().item() / inpainted_distances[:, :, None]))
        if inpainted_normals is not None:
            write_image(pjoin(vis_dir, 'aligned_normals_{}.jpg'.format(phase)), (inpainted_normals * .5 + .5).clip(0., 1.) * 255.)

        return inpainted_img, inpainted_distances, inpainted_normals

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.scene.load_state_dict(checkpoint['scene'])
        self.phase = checkpoint['phase']

    def render_dense(self, n_poses=180, cam_type='pano'):
        dense_pose_sampler = DenseTravelPoseSampler(self.pose_sampler, n_dense_poses=n_poses)
        out_dir = pjoin(self.exp_dir, 'dense_images_new_' + cam_type)
        os.makedirs(out_dir, exist_ok=True)

        color_frames = []
        for i in tqdm(range(dense_pose_sampler.n_poses)):
            pose = dense_pose_sampler.sample_pose(i)
            if cam_type == 'pano':
                pose[:3, :3] = torch.eye(3)
                rays = gen_pano_rays(pose, 512, 1024)
            else:
                rays = gen_pers_rays(pose, fov=np.deg2rad(75.), res=512)

            with torch.no_grad():
                render_result = self.scene.render(rays, query_keys=['rgb', 'distance'])
            colors = render_result['rgb']
            distances = render_result['distance']

            color_frames.append((colors.clip(0., 1.) * 255.).cpu().numpy().astype(np.uint8))
            write_image(pjoin(out_dir, 'image_{}.png'.format(i)), colors * 255.)
            write_image(pjoin(out_dir, 'distance_{}.png'.format(i)), colorize_single_channel_image(1. / distances))
        
        write_video(pjoin(out_dir, 'video.mp4'), color_frames, fps=30)

    def save_checkpoint(self):
        checkpoint = {
            'scene': self.scene.state_dict(),
            'sup_pool': self.sup_pool.state_dict(),
            'phase': self.phase
        }

        os.makedirs(os.path.join(self.exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.exp_dir, 'checkpoints', 'ckpt.pth'))


@hydra.main(version_base=None, config_path='./configs', config_name='nerf')
def main(conf: DictConfig) -> None:
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    mode = str(conf['mode'])
    runner = CoreRunner(conf)
    runner.set_eval()

    runner.execute(mode)


if __name__ == '__main__':
    main()
