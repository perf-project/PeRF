import torch
import torch.nn as nn
from utils.camera_utils import BoundedRays, Rays


class Scene:
    def __init__(self):
        pass

    @torch.no_grad()
    def render(self, rays: Rays, query_keys=('rgb',)):
        raise NotImplementedError

    def fit(self, sup_infos):
        raise NotImplementedError

    def get_pano_visibility_mask(self, sup_pool, pose):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def set_train(self):
        raise NotImplementedError

    def set_eval(self):
        raise NotImplementedError

