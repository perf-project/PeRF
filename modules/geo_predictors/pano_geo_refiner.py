import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.camera_utils import *
from modules.fields.networks import VanillaMLP
import tinycudann as tcnn

from tqdm import tqdm


class SphereDistanceField(nn.Module):
    def __init__(self,
                 n_levels=16,
                 log2_hashmap_size=19,
                 base_res=16,
                 fine_res=4096):
        super().__init__()
        per_level_scale = np.exp(np.log(fine_res / base_res) / (n_levels - 1))
        self.hash_grid = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": per_level_scale,
                "interpolation": "Smoothstep",
            }
        )

        self.geo_mlp = VanillaMLP(dim_in=n_levels * 2 + 3,
                                  dim_out=1,
                                  n_neurons=64,
                                  n_hidden_layers=2,
                                  sphere_init=True,
                                  weight_norm=True)

    def forward(self, directions, requires_grad=False):
        if requires_grad:
            if not self.training:
                directions = directions.clone()  # get a copy to enable grad
            directions.requires_grad_(True)

        dir_scaled = directions * 0.49 + 0.49
        selector = ((dir_scaled > 0.0) & (dir_scaled < 1.0)).all(dim=-1).to(torch.float32)
        scene_feat = self.hash_grid(dir_scaled)

        distance = self.geo_mlp(torch.cat([directions, scene_feat], -1))[..., 0]

        if requires_grad:
            grad = torch.autograd.grad(
                distance, directions, grad_outputs=torch.ones_like(distance),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]

            return distance, grad
        else:
            return distance


class PanoGeoRefiner:
    def __init__(self):
        pass

    def grads_to_normal(self, grads):
        height, width, _ = grads.shape
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width))
        ortho_a = torch.randn([height, width, 3])
        ortho_b = torch.linalg.cross(pano_dirs, ortho_a)
        ortho_b = ortho_b / torch.linalg.norm(ortho_b, 2, -1, True)
        ortho_a = torch.linalg.cross(ortho_b, pano_dirs)
        ortho_a = ortho_a / torch.linalg.norm(ortho_a, 2, -1, True)

        val_a = (grads * ortho_a).sum(-1, True) * pano_dirs + ortho_a
        val_a = val_a / torch.linalg.norm(val_a, 2, -1, True)
        val_b = (grads * ortho_b).sum(-1, True) * pano_dirs + ortho_b
        val_b = val_b / torch.linalg.norm(val_b, 2, -1, True)

        normals = torch.cross(val_a, val_b)
        normals = normals / torch.linalg.norm(normals, 2, -1, True)
        is_inside = ((normals * pano_dirs).sum(-1, True) < 0.).float()
        normals = normals * is_inside + -normals * (1. - is_inside)
        return normals


    def refine(self, distances, normals):
        '''
        :param distances: [H, W, 1] or [H, W]
        :param normals: [H, W, 3]
        :return: distance [H, W, 1] and normal [H, W, 3]
        '''

        distances = distances.squeeze()[..., None]
        height, width, _ = distances.shape
        sp_dis_field = SphereDistanceField()
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width))

        batch_size = 32768
        n_iters = 5000
        init_lr = 1e-2
        lr_alpha = 1e-2

        optimizer = torch.optim.Adam(sp_dis_field.parameters(), lr=init_lr)

        distances = distances.permute(2, 0, 1).contiguous()
        normals = normals.permute(2, 0, 1).contiguous()
        for iter_i in tqdm(range(n_iters)):
            rand_dirs = torch.randn([batch_size, 3])
            rand_dirs = rand_dirs / torch.linalg.norm(rand_dirs, 2, -1, True)
            ortho_a = torch.randn([batch_size, 3])
            ortho_b = torch.linalg.cross(rand_dirs, ortho_a)
            ortho_b = ortho_b / torch.linalg.norm(ortho_b, 2, -1, True)
            ortho_a = torch.linalg.cross(ortho_b, rand_dirs)
            ortho_a = ortho_a / torch.linalg.norm(ortho_a, 2, -1, True)

            pano_image_coord = direction_to_img_coord(rand_dirs)
            sample_coord = img_coord_to_sample_coord(pano_image_coord)
            ref_distance = F.grid_sample(distances[None], sample_coord[None, None], padding_mode='border')[0, 0, 0]
            ref_normal = F.grid_sample(normals[None], sample_coord[None, None], padding_mode='border')[0, :, 0].permute(1, 0).contiguous()
            pred_distance, grads = sp_dis_field(rand_dirs, requires_grad=True)

            val_a = (grads * ortho_a).sum(-1, True) * rand_dirs + ortho_a
            val_a = val_a / torch.linalg.norm(val_a, 2, -1, True)
            val_b = (grads * ortho_b).sum(-1, True) * rand_dirs + ortho_b
            val_b = val_b / torch.linalg.norm(val_b, 2, -1, True)
            error_a = (val_a * ref_normal).sum(-1, True)
            error_b = (val_b * ref_normal).sum(-1, True)
            errors = torch.cat([error_a, error_b], -1)

            distance_loss = F.smooth_l1_loss(ref_distance, pred_distance, beta=1e-2, reduction='mean')
            normal_loss = F.smooth_l1_loss(errors, torch.zeros_like(errors), beta=5e-1, reduction='mean')

            loss = distance_loss + normal_loss * 5e-2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress = iter_i / n_iters
            lr_factor = (np.cos(progress * np.pi) * .5 + .5) * (1. - lr_alpha) + lr_alpha
            for p in optimizer.param_groups:
                p['lr'] = init_lr * lr_factor

            if iter_i % 10 == 0:
                print(loss.item())


        # Get new distance map and normal map
        new_distances, new_grads = sp_dis_field(pano_dirs.reshape(-1, 3), requires_grad=True)
        new_distances = new_distances.detach().reshape(height, width, 1)
        new_normals = self.grads_to_normal(new_grads.detach().reshape(height, width, 3))

        return new_distances, new_normals

        # import trimesh
        # pcd = trimesh.PointCloud((new_distances * pano_dirs).reshape(-1, 3).cpu().numpy())
        # pcd.export('/home/ppwang/tmp/tmp.ply')
        # new_grads = new_grads.detach().reshape(height, width, 3)
        # new_distances /= new_distances.max()
        # from utils.utils import write_image, colorize_single_channel_image
        # write_image('/home/ppwang/tmp/tmp.png', colorize_single_channel_image(1. / new_distances))
        # write_image('/home/ppwang/tmp/tmp_normals.png', (new_normals * .5 + .5) * 255.)
        # exit()