import os.path
import numpy as np
import torch
import trimesh
import hashlib
import cv2 as cv

from os.path import join as pjoin
from utils.utils import read_dpt, read_image, write_image
from utils.camera_utils import *
from utils.geo_utils import align_scale, get_edge_mask
from modules.geo_predictors import PanoFusionInvPredictor, PanoFusionNormalPredictor, PanoGeoRefiner, PanoJointPredictor


class Dataset:
    def __init__(self):
        self.image_path = None
        self.ref_distance_path = None
        self.ref_normal_path = None
        self.ref_geometry_path = None
        self.image = None
        self.gt_distance = None
        self.ref_distance = None
        self.ref_normal = None
        self.height = 0
        self.width = 0
        self.data_dir = None
        self.case_name = 'wp'

    def get_ref_distance(self):
        assert self.image is not None
        assert self.ref_distance_path is not None
        assert self.height > 0 and self.width > 0

        ref_distance = None
        if os.path.exists(self.ref_distance_path):
            ref_distance = np.load(self.ref_distance_path)
            ref_distance = torch.from_numpy(ref_distance.astype(np.float32)).cuda()
        else:
            distance_predictor = PanoFusionInvPredictor()
            ref_distance, _ = distance_predictor(self.image,
                                                 torch.zeros([self.height, self.width]),
                                                 torch.ones([self.height, self.width]))


        return ref_distance

    def get_ref_normal(self):
        assert self.image is not None
        assert self.ref_normal_path is not None
        assert self.height > 0 and self.width > 0

        ref_normal = None
        if os.path.exists(self.ref_normal_path):
            ref_normal = np.load(self.ref_normal_path)
            ref_normal = torch.from_numpy(ref_normal.astype(np.float32)).cuda()
        else:
            normal_predictor = PanoFusionNormalPredictor()
            ref_normal = normal_predictor.inpaint_normal(self.image,
                                                         torch.ones([self.height, self.width, 3]) / np.sqrt(3.),
                                                         torch.ones([self.height, self.width]))


        return ref_normal

    def refine_geometry(self, distance_map, normal_map):
        refiner = PanoGeoRefiner()
        return refiner.refine(distance_map, normal_map)

    def get_joint_distance_normal(self, org_distance=None):
        assert self.image is not None
        assert self.ref_distance_path is not None
        assert self.ref_normal_path is not None
        assert self.height > 0 and self.width > 0

        if os.path.exists(self.ref_distance_path) and\
           os.path.exists(self.ref_normal_path):
            ref_distance = np.load(self.ref_distance_path)
            ref_distance = torch.from_numpy(ref_distance.astype(np.float32)).cuda()
            ref_normal = np.load(self.ref_normal_path)
            ref_normal = torch.from_numpy(ref_normal.astype(np.float32)).cuda()
        else:
            joint_predictor = PanoJointPredictor()
            if org_distance is None:
                ref_distance, ref_normal = joint_predictor(self.image,
                                                           torch.ones([self.height, self.width, 1]),
                                                           torch.ones([self.height, self.width]))
            else:
                ref_distance, ref_normal = joint_predictor(self.image,
                                                           ref_distance=org_distance,
                                                           mask=torch.zeros([self.height, self.width]),
                                                           reg_loss_weight=0.)


        return ref_distance, ref_normal

    def normalization(self):
        # Normalization
        # self.ref_distance /= torch.exp(torch.log(self.ref_distance.mean()))
        scale = self.ref_distance.max().item() * 1.05
        self.ref_distance /= scale

    def save_ref_geometry(self):
        # Save distance and normal data
        if self.ref_distance_path is not None:
            np.save(self.ref_distance_path, self.ref_distance.cpu().numpy())
        if self.ref_normal_path is not None:
            np.save(self.ref_normal_path, self.ref_normal.cpu().numpy())

        # Save point cloud
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(self.height, self.width))
        pts = pano_dirs * self.ref_distance.squeeze()[..., None]
        pts = pts.cpu().numpy().reshape(-1, 3)
        if self.image is not None:
            pcd = trimesh.PointCloud(pts, vertex_colors=self.image.reshape(-1, 3).cpu().numpy())
        else:
            pcd = trimesh.PointCloud(pts)

        assert self.ref_geometry_path is not None and self.ref_geometry_path[-4:] == '.ply'
        pcd.export(self.ref_geometry_path)

    @torch.no_grad()
    def ref_point_cloud(self):
        '''
        return reference point cloud [h, w, 3]
        '''
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(self.height, self.width))
        pts = pano_dirs * self.ref_distance.squeeze()[..., None]
        return pts


class WildDataset(Dataset):
    def __init__(self, conf):
        super().__init__()
        self.image_path = conf.image_path
        self.ref_distance_path = '.'.join(self.image_path.split('.')[:-1]) + '_ref_distance.npy'
        self.ref_normal_path = '.'.join(self.image_path.split('.')[:-1]) + '_ref_normal.npy'
        self.ref_geometry_path = '.'.join(self.image_path.split('.')[:-1]) + '_ref_geometry.ply'

        self.case_name = self.image_path.split('/')[-2]

        self.image = read_image(self.image_path, to_torch=True, squeeze=True).cuda()
        if 'image_resize' in conf:
            self.width, self.height = conf['image_resize']
            self.image = cv.resize(self.image.cpu().numpy(), (self.width, self.height), cv.INTER_AREA)
            self.image = torch.from_numpy(self.image).cuda()
        else:
            self.height, self.width, _ = self.image.shape

        self.ref_distance, self.ref_normal = self.get_joint_distance_normal()

        self.normalization()
        self.save_ref_geometry()

