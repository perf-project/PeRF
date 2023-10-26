import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image

from .geo_predictor import GeoPredictor
from .omnidata.modules.unet import UNet
from .omnidata.modules.midas.dpt_depth import DPTDepthModel
from .omnidata.data.transforms import get_transform


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean.item())

    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img

class OmnidataPredictor(GeoPredictor):
    def __init__(self):
        super().__init__()
        self.img_size = 384
        ckpt_path = './pre_checkpoints/omnidata_dpt_depth_v2.ckpt'
        self.model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=1)
        self.model.to(torch.device('cpu'))
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.trans_totensor = transforms.Compose([transforms.Resize(self.img_size, interpolation=Image.BILINEAR),
                                                  transforms.CenterCrop(self.img_size),
                                                  transforms.Normalize(mean=0.5, std=0.5)])
        # self.trans_rgb  = transforms.Compose([transforms.Resize(512, interpolation=PIL.Image.BILINEAR),
        #                   transforms.CenterCrop(512)])

    def predict_disparity(self, img, **kwargs):
        self.model.to(torch.device('cuda'))
        img_tensor = self.trans_totensor(img)
        output = self.model(img_tensor).clip(0., 1.)
        self.model.to(torch.device('cpu'))
        # output = F.interpolate(output[:, None], size=(512, 512), mode='bicubic')
        output = output.clip(0., 1.)
        # output = 1. - output
        output = 1. / (output + 1e-6)
        return output[:, None]

    def predict_depth(self, img, **kwargs):
        self.model.to(torch.device('cuda'))
        img_tensor = self.trans_totensor(img)
        output = self.model(img_tensor).clip(0., 1.)
        self.model.to(torch.device('cpu'))
        # output = F.interpolate(output[:, None], size=(512, 512), mode='bicubic')
        output = output.clip(0., 1.)
        return output[:, None]
