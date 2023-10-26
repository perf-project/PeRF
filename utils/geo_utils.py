import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from kornia.filters import laplacian
from kornia.morphology import erosion, dilation

from trimesh.creation import icosphere as IcoSphere


def _apply_rot(pts, rot_mat):
    assert rot_mat.shape == (3, 3)
    return torch.matmul(rot_mat, pts[..., None])[..., 0]

def _verts_to_dirs(pt_a, pt_b, pt_c, gen_res, ratio):
    # make pt_a the sole point
    def same_z(a, b):
        return np.abs(a[2] - b[2]) < 1e-4

    assert same_z(pt_a, pt_b) or same_z(pt_b, pt_c) or same_z(pt_a, pt_c)

    if same_z(pt_a, pt_b):
        pt_a, pt_c = pt_c, pt_a
    elif same_z(pt_a, pt_c):
        pt_a, pt_b = pt_b, pt_a

    assert same_z(pt_b, pt_c)

    if np.cross(pt_c, pt_b)[2] < 0.:
        pt_b, pt_c = pt_c, pt_b

    pt_a = torch.from_numpy(pt_a).cuda()
    pt_b = torch.from_numpy(pt_b).cuda()
    pt_c = torch.from_numpy(pt_c).cuda()

    pt_m = (pt_b + pt_c) * .5
    down_vec = pt_a - pt_m
    if down_vec[2] > 0.:
        down_vec = -down_vec

    pt_center = (pt_a + pt_b + pt_c) / 3.
    right_vec = pt_c - pt_b

    right_len = torch.linalg.norm(right_vec, 2, -1).item()
    down_len = torch.linalg.norm(down_vec, 2, -1).item()
    half_len = torch.linalg.norm(pt_center - pt_b, 2, -1).item() * ratio
    right_vec = right_vec / right_len * half_len
    down_vec = down_vec / down_len * half_len
    pt_base = pt_center - right_vec - down_vec
    right_vec *= 2
    down_vec *= 2

    ii, jj = torch.meshgrid(torch.linspace(.5 / gen_res, 1. - .5 / gen_res, gen_res),
                            torch.linspace(.5 / gen_res, 1. - .5 / gen_res, gen_res),
                            indexing='ij')
    to_vec = pt_base + right_vec * .5 + down_vec * .5

    dirs = pt_base[None, None, :] + \
           down_vec[None, None, :] * ii[:, :, None] + \
           right_vec[None, None, :] * jj[:, :, None]

    pers_ratios = torch.linalg.norm(dirs, 2, -1, True) / torch.linalg.norm(to_vec, 2, -1, True)[None, None]

    dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)
    return dirs, pers_ratios, to_vec, down_vec * .5, right_vec * .5


@torch.no_grad()
def panorama_to_pers_directions(gen_res=512, ratio=1.):
    '''
    :param img: [H, W, C]
    :param gen_res:
    :return:
    '''
    ico_sphere = IcoSphere(subdivisions=0)
    vertices, faces = ico_sphere.vertices, ico_sphere.faces
    ang = np.arctan(.525731112119133606 / .850650808352039932)
    rot_vec = np.array([ang, 0., 0.])
    rot = Rotation.from_rotvec(rot_vec)
    vertices = rot.apply(vertices)
    vertices = vertices.astype(np.float32)

    # Generate coords for each face
    all_dirs = []
    all_ratios = []
    to_vecs = []
    down_vecs = []
    right_vecs = []

    for i in range(len(faces)):
        face = faces[i]
        pt_a, pt_b, pt_c = vertices[face[0]].copy(), vertices[face[1]].copy(), vertices[face[2]].copy()

        dirs, ratios, to_vec, down_vec, right_vec = _verts_to_dirs(pt_a, pt_b, pt_c, gen_res=gen_res, ratio=ratio)
        all_dirs.append(dirs)
        all_ratios.append(ratios)
        to_vecs.append(to_vec)
        down_vecs.append(down_vec)
        right_vecs.append(right_vec)

    return torch.stack(all_dirs, dim=0),\
           torch.stack(all_ratios, dim=0),\
           torch.stack(to_vecs, dim=0),\
           torch.stack(down_vecs, dim=0),\
           torch.stack(right_vecs, dim=0)

@torch.no_grad()
def panorama_to_pers_directions(gen_res=512, ratio=1., ex_rot=None):
    '''
    Split too may perspective cameras that covers the whole sphere
    :param img: [H, W, C]
    :param gen_res:
    :return:
    '''
    ico_sphere = IcoSphere(subdivisions=0)
    vertices, faces = ico_sphere.vertices, ico_sphere.faces
    ang = np.arctan(.525731112119133606 / .850650808352039932)
    rot_vec = np.array([ang, 0., 0.])
    rot = Rotation.from_rotvec(rot_vec)
    vertices = rot.apply(vertices)
    vertices = vertices.astype(np.float32)

    # Generate coords for each face
    all_dirs = []
    all_ratios = []
    to_vecs = []
    down_vecs = []
    right_vecs = []

    for i in range(len(faces)):
        face = faces[i]
        pt_a, pt_b, pt_c = vertices[face[0]].copy(), vertices[face[1]].copy(), vertices[face[2]].copy()

        dirs, ratios, to_vec, down_vec, right_vec = _verts_to_dirs(pt_a, pt_b, pt_c, gen_res=gen_res, ratio=ratio)
        all_dirs.append(dirs)
        all_ratios.append(ratios)
        to_vecs.append(to_vec)
        down_vecs.append(down_vec)
        right_vecs.append(right_vec)

    all_dirs = torch.stack(all_dirs, dim=0)
    all_ratios = torch.stack(all_ratios, dim=0)
    to_vecs = torch.stack(to_vecs, dim=0)
    down_vecs = torch.stack(down_vecs, dim=0)
    right_vecs = torch.stack(right_vecs, dim=0)

    if ex_rot is None:
        return all_dirs, all_ratios, to_vecs, down_vecs, right_vecs

    if isinstance(ex_rot, str) and ex_rot == 'rand':
        ang = np.random.rand() * 2. * np.pi
        rot_vec = np.array([0., 0., ang])
        rot = Rotation.from_rotvec(rot_vec).as_matrix().astype(np.float32)
        rot = torch.from_numpy(rot).to(all_dirs.device)
        all_dirs = _apply_rot(all_dirs, rot)
        to_vecs = _apply_rot(to_vecs, rot)
        down_vecs = _apply_rot(down_vecs, rot)
        right_vecs = _apply_rot(right_vecs, rot)
        return all_dirs, all_ratios, to_vecs, down_vecs, right_vecs

    raise NotImplementedError



@torch.no_grad()
def panorama_to_pers_cameras(ratio=1.):
    '''
    Split too may perspective cameras that covers the whole sphere
    :param img: [H, W, C]
    :param gen_res:
    :return:
    '''

    _, _, to_vecs, down_vecs, right_vecs = panorama_to_pers_directions(ratio=ratio)
    down_vecs_len = torch.linalg.norm(down_vecs, 2, -1, True)
    right_vecs_len = torch.linalg.norm(right_vecs, 2, -1, True)
    fovy = torch.arctan(down_vecs_len) * 2.
    fovx = torch.arctan(right_vecs_len) * 2.

    down_vecs = down_vecs / down_vecs_len
    right_vecs = right_vecs / right_vecs_len

    w2c = torch.stack([right_vecs, down_vecs, to_vecs], dim=1)
    c2w = torch.linalg.inv(w2c)

    return c2w, fovy, fovx


@torch.no_grad()
def get_edge_mask(val, threshold=0.01):
    x_laplacian = laplacian(val.squeeze()[None, :, :, None].permute(0, 3, 1, 2), kernel_size=3)
    edge_mask = (x_laplacian.abs() < threshold).float()
    edge_mask = erosion(edge_mask, kernel=torch.ones(3, 3, device=val.device))
    edge_mask = dilation(edge_mask, kernel=torch.ones(3, 3, device=val.device))
    edge_mask = edge_mask > .5
    edge_mask = edge_mask[0].permute(1, 2, 0)
    return edge_mask

def _get_cliped_mask(x, q):
    q_min = torch.quantile(x, 1. - q).item()
    q_max = torch.quantile(x, q).item()
    mask = ((x >= q_min) & (x <= q_max))
    return mask

@torch.no_grad()
def align_scale(a, b, mask, q=0.95):
    # return global scale b / a
    if torch.is_tensor(a):
        is_tensor = True
    else:
        is_tensor = False
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)

    mask = mask & _get_cliped_mask(a, q=q) & _get_cliped_mask(b, q=q)
    a = a[mask]
    b = b[mask]

    return (b.mean() / a.mean()).item()


# Test
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    ret = panorama_to_pers_cameras()
