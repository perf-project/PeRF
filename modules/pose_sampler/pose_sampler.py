import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.camera_utils import *
from scipy.ndimage import minimum_filter1d, gaussian_filter1d


class PoseSampler:
    def __init__(self):
        self.n_poses = 0

    @torch.no_grad()
    def sample_pose(self, idx):
        raise NotImplementedError
