# Borrow from OSX and agora_evaluation
import torch
from torchvision import transforms
import numpy as np
import scipy
import colorsys
from torch.nn import functional as F

unNormalize = transforms.Normalize(
        mean=-np.array([0.485,0.456,0.406]) / np.array([0.229,0.224,0.225]),
        std=1 / np.array([0.229,0.224,0.225]))

def adjust_colors(colors, saturation_threshold = 0.3, brightness_threshold = 0.8):

    def adjust_func(rgb_color):
        r, g, b = rgb_color
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if v < brightness_threshold:
            v = brightness_threshold
        if s > saturation_threshold:
            s = saturation_threshold
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return r, g, b

    adjusted_colors = np.apply_along_axis(adjust_func, 1, colors)
    return adjusted_colors

def to_zorder(img, z_order_map, y_coords, x_coords):
    h, w = img.shape[:2]
    assert max(h,w) <= z_order_map.shape[0]
    clipped_z = z_order_map[:h,:w].flatten()
    sorted_idx = torch.argsort(clipped_z)

    img_z = img.flatten(0,1)[sorted_idx]
    z_order_idx = clipped_z[sorted_idx]
    y_idx = y_coords[:h,:w].flatten()[sorted_idx]
    x_idx = x_coords[:h,:w].flatten()[sorted_idx]

    return img_z, z_order_idx, y_idx, x_idx

def img2patch(x, patch_size):
    assert x.ndim == 3  # (c,h,w)
    c, h, w = x.shape
    feature_h = h//patch_size
    feature_w = w//patch_size
    x_patched = x.view(c, feature_h, patch_size, feature_w, patch_size).permute(1,3,0,2,4)
    return x_patched

def img2patch_flat(x, patch_size):
    assert x.ndim == 3  # (c,h,w)
    c, h, w = x.shape
    feature_h = h//patch_size
    feature_w = w//patch_size
    x_patched = x.view(c, feature_h, patch_size, feature_w, patch_size).permute(1,3,0,2,4)
    return x_patched.flatten(0,1)

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Based on Zhou et al., "On the Continuity of Rotation
    Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    if isinstance(x, torch.Tensor):
        x = x.reshape(-1, 3, 2)
    elif isinstance(x, np.ndarray):
        x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert 3x4 rotation matrix to Rodrigues vector
    Args:
        rotation_matrix (Tensor): rotation matrix.
    Returns:
        Tensor: Rodrigues vector transformation.
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`
    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1],
                           dtype=torch.float32,
                           device=rotation_matrix.device)
        hom = hom.reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert quaternion vector to angle axis of rotation.
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
    Args:
        quaternion (torch.Tensor): tensor with quaternions.
    Return:
        torch.Tensor: tensor with angle axis of rotation.
    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`
    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            'Input must be a tensor of shape Nx4 or 4. Got {}'.format(
                quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert 3x4 rotation matrix to 4d quaternion vector
    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201
    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.
    Return:
        Tensor: the rotation in quaternion
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`
    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            'Input size must be a three dimensional tensor. Got {}'.format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            'Input size must be a N x 3 x 4  tensor. Got {}'.format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([
        rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0,
        rmat_t[:, 0, 1] + rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2]
    ], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([
        rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
        t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]
    ], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([
        rmat_t[:, 0, 1] - rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2
    ], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([
        t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
        rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] - rmat_t[:, 1, 0]
    ], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def rot6d_to_axis_angle(x):
    bs,num_queries,_ = x.shape
    rot_mat = rot6d_to_rotmat(x)
    rot_mat = torch.cat([rot_mat, torch.zeros((rot_mat.shape[0], 3, 1)).cuda().float()], 2)  # 3x4 rotation matrix
    axis_angle = rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)  # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle.reshape(bs,num_queries,-1)


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)

    t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c * R, np.transpose(A))) + t
    return A2


def pelvis_align(joints, verts=None):

    left_id = 1
    right_id = 2

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.0
    if verts is not None:
        return verts - np.expand_dims(pelvis, axis=0)
    else:
        return joints - np.expand_dims(pelvis, axis=0)


def root_align(joints, verts=None):

    left_id = 1
    right_id = 2

    root = joints[0, :]
    if verts is not None:
        return verts - np.expand_dims(root, axis=0)
    else:
        return joints - np.expand_dims(root, axis=0)