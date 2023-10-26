import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from tqdm import tqdm
from kornia.morphology import erosion, dilation


from .inpainter import Inpainter
from .lama_inpainter import LamaInpainter
from .diffusion_local_inpainter import DiffusionLocalInpainter

from utils.geo_utils import panorama_to_pers_directions
from utils.camera_utils import img_coord_to_sample_coord,\
    direction_to_img_coord, img_coord_to_pano_direction, direction_to_pers_img_coord


class PanoPersFusionInpainter(Inpainter):
    def __init__(self, inpainter_type, use_lama_assist=True):
        super().__init__()
        if inpainter_type == 'stable_diffusion':
            self.diff_inpainter = DiffusionLocalInpainter()
        else:
            assert False
        if use_lama_assist:
            self.lama_inpainter = LamaInpainter()
        else:
            self.lama_inpainter = None

    @torch.no_grad()
    def inpaint(self, img, mask):
        img = img.squeeze().permute(2, 0, 1)
        mask = mask.squeeze()[None]
        inpainted_img = img.clone()

        pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = panorama_to_pers_directions(gen_res=512, ratio=1.1)

        n_pers = len(pers_dirs)
        img_coords = direction_to_img_coord(pers_dirs)
        sample_coords = img_coord_to_sample_coord(img_coords)

        _, pano_height, pano_width = img.shape
        pano_img_coords = torch.meshgrid(torch.linspace(.5 / pano_height, 1. - .5 / pano_height, pano_height),
                                         torch.linspace(.5 / pano_width,  1. - .5 / pano_width, pano_width),
                                         indexing='ij')
        pano_img_coords = torch.stack(list(pano_img_coords), dim=-1)

        pano_dirs = img_coord_to_pano_direction(pano_img_coords)

        for i in tqdm(range(n_pers)):
            cur_sample_coords = sample_coords[i]
            pers_image = F.grid_sample(inpainted_img[None], cur_sample_coords[None], padding_mode='border')[0]
            pers_mask = F.grid_sample(mask[None, :, :], cur_sample_coords[None], padding_mode='border')[0]
            pers_mask = (pers_mask > 0.5).float()
            if self.lama_inpainter is not None:
                kernel = torch.from_numpy(cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))).float().to(pers_mask.device)
                smooth_mask = erosion(pers_mask[None], kernel=kernel)[0]
                smooth_mask = dilation(smooth_mask[None], kernel=kernel)[0]
                smooth_mask = torch.minimum(smooth_mask, pers_mask)
                lama_inpainted = self.lama_inpainter.inpaint(pers_image[None], pers_mask[None])[0]
                if smooth_mask.max().item() > .5:
                    cur_inpainted = self.diff_inpainter.inpaint(lama_inpainted[None], smooth_mask[None])[0]
                else:
                    cur_inpainted = lama_inpainted
            else:
                if pers_mask.max().item() > .5:
                    cur_inpainted = self.diff_inpainter.inpaint(pers_image[None], pers_mask[None])[0]
                else:
                    cur_inpainted = pers_image

            proj_coord, proj_mask = direction_to_pers_img_coord(pano_dirs, to_vecs[i], down_vecs[i], right_vecs[i])
            proj_coord = img_coord_to_sample_coord(proj_coord)

            cur_inpainted_pano_img = F.grid_sample(cur_inpainted[None], proj_coord[None], padding_mode='border')[0]
            proj_mask = proj_mask.permute(2, 0, 1).float()
            inpainted_img = inpainted_img * (1. - proj_mask) + cur_inpainted_pano_img * proj_mask
            mask = mask * (1. - proj_mask) + 0. * proj_mask

        return inpainted_img.permute(1, 2, 0)

