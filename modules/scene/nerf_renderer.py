import torch
import torch.nn as nn

# from nerfacc import OccupancyGrid, ray_marching, rendering, ContractionType
from nerfacc import accumulate_along_rays, render_weight_from_density, render_transmittance_from_alpha
from nerfacc.estimators.prop_net import PropNetEstimator
from nerfacc.estimators.occ_grid import OccGridEstimator
from modules.fields.ngp_nerf import NGPNeRF

class NeRFPropRenderer(nn.Module):
    def __init__(self, max_radius, bg_color):
        super().__init__()
        self.max_radius = max_radius
        self.bg_color = bg_color
        assert self.bg_color in ['rand_noise', 'black', 'white']

        self.n_samples = 64
        self.n_samples_per_prop = [128, 64]

    def occ_eval_fn(self, x):
        t = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
        step_size = torch.clamp(t / 256., min=1e-2)
        density = self.radiance_field.query_density(x)
        return density * step_size

    def render(self,
               nerf: NGPNeRF,
               prop_networks,
               estimator: PropNetEstimator,
               rays_o, rays_d, near, far,
               sampling_requires_grad):
        assert near.shape[-1] == 1 and len(near.shape) == 2
        near = near[..., 0]
        far = far[..., 0]

        n_rays = rays_o.shape[0]

        def prop_sigma_fn(t_starts, t_ends, proposal_network):
            t_origins = rays_o[..., None, :]
            t_dirs = rays_d[..., None, :]
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            sigmas = proposal_network(positions)
            sigmas[..., -1, :] = torch.inf
            return sigmas.squeeze(-1)

        def rgb_alpha_fn(t_starts, t_ends):
            n_rays, n_samples = t_starts.shape
            t_origins = rays_o[..., None, :]
            t_dirs = rays_d[..., None, :]
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            rgb, density = nerf(positions.reshape(-1, 3))
            rgb = rgb.reshape(n_rays, n_samples, 3)
            density = density.reshape(n_rays, n_samples, 1)

            dists = (t_ends - t_starts)[..., None]
            alpha = 1. - torch.exp(-density * dists)

            return rgb, alpha[..., 0]

        t_starts, t_ends = estimator.sampling(
            prop_sigma_fns=[lambda *args: prop_sigma_fn(*args, p) for p in prop_networks],
            prop_samples=self.n_samples_per_prop,
            num_samples=self.n_samples,
            n_rays=len(rays_o),
            near_plane=1e-2,
            far_plane=2.,
            sampling_type='uniform',
            stratified=nerf.training,
            requires_grad=sampling_requires_grad,
        )

        rgbs, alphas = rgb_alpha_fn(t_starts, t_ends)
        weights, trans = render_weight_from_alpha(alphas)

        colors = accumulate_along_rays(weights, rgbs)
        opacities = accumulate_along_rays(weights, values=None)

        sampled_distances = ((t_starts + t_ends) / 2.0)[..., None]
        sampled_pts = rays_o[..., None, :] + rays_d[..., None, :] * sampled_distances

        distances = accumulate_along_rays(weights, sampled_distances)

        if self.bg_color == 'rand_noise':
            bg_color = torch.rand(n_rays, 3)
        elif self.bg_color == 'white':
            bg_color = torch.ones(n_rays, 3)
        else:
            bg_color = torch.zeros(n_rays, 3)

        colors = colors + bg_color * (1.0 - opacities)
        distances = distances + torch.rand_like(distances) * (1. - opacities)

        return {
            'rgb': colors[:, :3],
            'distance': distances,
            'sampled_pts': sampled_pts,
            'weights': weights,
            'opacities': opacities,
            'trans': trans,
            't_starts': t_starts,
            't_ends': t_ends,
        }


class NeRFOCCRenderer(nn.Module):
    def __init__(self, max_radius, bg_color):
        super().__init__()
        self.max_radius = max_radius
        self.bg_color = bg_color
        assert self.bg_color in ['rand_noise', 'black', 'white']

    def render(self,
               nerf: NGPNeRF,
               estimator: OccGridEstimator,
               rays_o, rays_d, near, far,
               geo_inference=False,
               app_inference=False):
        assert near.shape[-1] == 1 and len(near.shape) == 2
        near = near[..., 0]
        far = far[..., 0]

        n_rays = rays_o.shape[0]

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = nerf.query_density(positions)
            return sigmas.squeeze(-1)

        def rgb_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            rgbs = nerf.query_rgb(positions)
            return rgbs

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            rgbs, sigmas = nerf(positions, t_dirs)
            return rgbs, sigmas.squeeze(-1)

        ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            near_plane=0.,
            far_plane=1.5,
            render_step_size=5e-4,
            stratified=nerf.training,
            cone_angle=0.,
            alpha_thre=0.,
        )
        if ray_indices.numel() <= 0:
            return {
                'is_valid': False,
                'rgb': torch.zeros(n_rays, 3),
                'distance': torch.zeros(n_rays, 1),
                'opacities': torch.zeros(n_rays, 1)
            }

        if geo_inference:
            with torch.no_grad():
                sigmas = sigma_fn(t_starts, t_ends, ray_indices)
        else:
            sigmas = sigma_fn(t_starts, t_ends, ray_indices)

        weights, trans, alphas =\
            render_weight_from_density(t_starts, t_ends, sigmas, ray_indices=ray_indices)

        opacities = accumulate_along_rays(weights, values=None, ray_indices=ray_indices, n_rays=n_rays)
        sampled_distances = ((t_starts + t_ends) / 2.0)[..., None]
        distances = accumulate_along_rays(weights, sampled_distances, ray_indices=ray_indices, n_rays=n_rays)

        if app_inference:
            with torch.no_grad():
                rgbs = rgb_fn(t_starts, t_ends, ray_indices)
        else:
            rgbs = rgb_fn(t_starts, t_ends, ray_indices)

        colors = accumulate_along_rays(weights.detach(), values=rgbs, ray_indices=ray_indices, n_rays=n_rays)

        if self.bg_color == 'rand_noise':
            bg_color = torch.rand(n_rays, 3)
        elif self.bg_color == 'white':
            bg_color = torch.ones(n_rays, 3)
        else:
            bg_color = torch.zeros(n_rays, 3)

        if nerf.training:
            distances = torch.relu(distances + (torch.rand_like(distances) * 2. - 1.) * (1. - opacities))
            colors = colors + bg_color * (1. - opacities).detach()
        else:
            distances = distances + torch.ones_like(distances) * 5. * (1. - opacities).detach()
            colors = colors + torch.ones(n_rays, 3) * .5 * (1. - opacities).detach()

        return {
            'is_valid': True,
            'rgb': colors,
            'distance': distances,
            'weights': weights,
            'opacities': opacities,
            'trans': trans,
            't_starts': t_starts,
            't_ends': t_ends,
            'ray_indices': ray_indices,
        }
