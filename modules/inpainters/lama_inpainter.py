import torch

import os
from .inpainter import Inpainter
from .lama.saicinpainting.training.data.datasets import make_default_val_dataset
from .lama.saicinpainting.training.trainers import load_checkpoint

from omegaconf import OmegaConf


class LamaInpainter(Inpainter):
    def __init__(self):
        super().__init__()
        # predict_config = OmegaConf.create(yaml.safe_load())
        predict_config = OmegaConf.load('./modules/inpainters/lama/predict_config.yaml')
        train_config = OmegaConf.load('./pre_checkpoints/big-lama-config.yaml')

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        # checkpoint_path = os.path.join(predict_config.model.path,X
        #                                'models',
        #                                predict_config.model.checkpoint)

        checkpoint_path = './pre_checkpoints/big-lama.ckpt'
        self.model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        self.model.freeze()

    @torch.no_grad()
    def inpaint(self, img, mask, out_key='inpainted'):
        self.model = self.model.to('cuda')
        batch = { 'image': img, 'mask': mask }
        batch['image'] = (batch['image'] * 255.).to(torch.uint8).float() / 255.
        batch['mask'] = (batch['mask'] > 0) * 1
        batch = self.model(batch)
        cur_res = batch[out_key]
        unpad_to_size = batch.get('unpad_to_size', None)

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:, :orig_height, :orig_width]

        self.model = self.model.to('cpu')
        return cur_res
