import sys
import cv2
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from imwatermark import WatermarkEncoder
from pathlib import Path
from icecream import ic

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

from .inpainter import Inpainter


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    ic(image.shape)
    ic(mask.shape)
    ic(masked_image.shape)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded


class DiffusionLocalInpainter(Inpainter):
    def __init__(self):
        super().__init__()
        self.sampler = initialize_model("./ldm/configs/stable-diffusion/v2-inpainting-inference.yaml",
                                        "./pre_checkpoints/512-inpainting-ema.ckpt")
        self.sampler.model = self.sampler.model.to(torch.device('cpu'))
        self.ddim_steps = 20
        self.unconditional_guidance_scale = 1.

    @torch.no_grad()
    def inpaint(self, img, mask):
        '''
        :param img: B C H W?
        :param mask:
        :return:
        '''
        num_samples, _, h, w = img.shape
        batch = {
            "image": img * 2. - 1.,
            "txt": [""],
            "mask": (mask > 0.5).float(),
            "masked_image": (img * 2. - 1.) * (mask <= 0.5),
        }
        self.sampler.model = self.sampler.model.to(torch.device('cuda'))
        model = self.sampler.model

        wm = "SDV2"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

        prng = np.random.RandomState(seed=0)
        start_code = prng.randn(num_samples, 4, h // 8, w // 8)
        start_code = torch.from_numpy(start_code).to(
            device=torch.device('cuda'), dtype=torch.float32)

        with torch.autocast('cuda'):
            c = model.cond_stage_model.encode(batch["txt"])

            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    bchw = [num_samples, 4, h // 8, w // 8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = model.get_first_stage_encoding(
                        model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            shape = [model.channels, h // 8, w // 8]
            samples_cfg, intermediates = self.sampler.sample(
                self.ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=1.0,
                unconditional_guidance_scale=self.unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
                x_T=start_code,
            )
            x_samples_ddim = model.decode_first_stage(samples_cfg)

            result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                 min=0.0, max=1.0)
            result = result * (mask > 0.5).float() + img * (mask <= 0.5).float()

        self.sampler.model = self.sampler.model.to(torch.device('cpu'))
        return result.to(torch.float32)
