"""Dependency-light SO(3) conversions shared by TCP data and control adapters."""

from __future__ import annotations

import torch


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to matrices with stable small-angle terms."""
    x, y, z = axis_angle.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero],
        dim=-1,
    ).reshape(*axis_angle.shape[:-1], 3, 3)
    theta_squared = axis_angle.square().sum(dim=-1, keepdim=True)
    theta = theta_squared.sqrt()
    small = theta_squared < 1.0e-8
    sine_scale = torch.where(
        small,
        1.0 - theta_squared / 6.0 + theta_squared.square() / 120.0,
        torch.sin(theta) / theta.clamp_min(1.0e-8),
    )
    cosine_scale = torch.where(
        small,
        0.5 - theta_squared / 24.0 + theta_squared.square() / 720.0,
        (1.0 - torch.cos(theta)) / theta_squared.clamp_min(1.0e-8),
    )
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    identity = identity.expand(*axis_angle.shape[:-1], 3, 3)
    return identity + sine_scale.unsqueeze(-1) * skew + cosine_scale.unsqueeze(-1) * (skew @ skew)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert matrices to axis-angle while preserving the axis near pi."""
    m00, m01, m02 = matrix[..., 0, :].unbind(dim=-1)
    m10, m11, m12 = matrix[..., 1, :].unbind(dim=-1)
    m20, m21, m22 = matrix[..., 2, :].unbind(dim=-1)
    quaternion_abs = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    ).clamp_min(0.0).sqrt()
    quaternion_candidates = torch.stack(
        [
            torch.stack([quaternion_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, quaternion_abs[..., 1].square(), m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, quaternion_abs[..., 2].square(), m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, quaternion_abs[..., 3].square()], dim=-1),
        ],
        dim=-2,
    )
    quaternion_candidates = quaternion_candidates / (2.0 * quaternion_abs.unsqueeze(-1).clamp_min(0.1))
    candidate_index = quaternion_abs.argmax(dim=-1)
    gather_index = candidate_index[..., None, None].expand(*candidate_index.shape, 1, 4)
    quaternion_wxyz = quaternion_candidates.gather(-2, gather_index).squeeze(-2)
    quaternion_wxyz = torch.where(quaternion_wxyz[..., :1] < 0.0, -quaternion_wxyz, quaternion_wxyz)
    vector = quaternion_wxyz[..., 1:]
    sine_half_angle = torch.linalg.vector_norm(vector, dim=-1)
    half_angle = torch.atan2(sine_half_angle, quaternion_wxyz[..., 0])
    scale = torch.where(
        sine_half_angle < 1.0e-8,
        torch.full_like(sine_half_angle, 2.0),
        2.0 * half_angle / sine_half_angle.clamp_min(1.0e-8),
    )
    return vector * scale.unsqueeze(-1)


def quaternion_xyzw_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized xyzw quaternions into rotation matrices."""
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-8)
    x, y, z, w = quaternion.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def matrix_to_quaternion_xyzw(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to normalized xyzw quaternions."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Rotation matrices must end with [3,3], got {matrix.shape}.")
    m00, m01, m02 = matrix[..., 0, :].unbind(dim=-1)
    m10, m11, m12 = matrix[..., 1, :].unbind(dim=-1)
    m20, m21, m22 = matrix[..., 2, :].unbind(dim=-1)
    magnitudes = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    ).clamp_min(0.0).sqrt()
    candidates = torch.stack(
        [
            torch.stack([magnitudes[..., 0].square(), m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, magnitudes[..., 1].square(), m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, magnitudes[..., 2].square(), m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, magnitudes[..., 3].square()], dim=-1),
        ],
        dim=-2,
    )
    candidates = candidates / (2.0 * magnitudes.unsqueeze(-1).clamp_min(0.1))
    candidate_index = magnitudes.argmax(dim=-1)
    gather_index = candidate_index[..., None, None].expand(*candidate_index.shape, 1, 4)
    quaternion_wxyz = candidates.gather(-2, gather_index).squeeze(-2)
    quaternion_xyzw = torch.cat([quaternion_wxyz[..., 1:], quaternion_wxyz[..., :1]], dim=-1)
    return quaternion_xyzw / torch.linalg.vector_norm(
        quaternion_xyzw, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)


def quaternion_slerp_xyzw(
    start: torch.Tensor,
    end: torch.Tensor,
    fraction: torch.Tensor,
) -> torch.Tensor:
    """Shortest-path spherical interpolation between xyzw quaternions."""
    start = start / torch.linalg.vector_norm(start, dim=-1, keepdim=True).clamp_min(1.0e-8)
    end = end / torch.linalg.vector_norm(end, dim=-1, keepdim=True).clamp_min(1.0e-8)
    dot = (start * end).sum(dim=-1, keepdim=True)
    end = torch.where(dot < 0.0, -end, end)
    dot = dot.abs().clamp(max=1.0)
    fraction = torch.as_tensor(fraction, dtype=start.dtype, device=start.device).unsqueeze(-1)

    angle = torch.acos(dot)
    sine = torch.sin(angle)
    interpolated = (
        torch.sin((1.0 - fraction) * angle) / sine.clamp_min(1.0e-8) * start
        + torch.sin(fraction * angle) / sine.clamp_min(1.0e-8) * end
    )
    linear = (1.0 - fraction) * start + fraction * end
    value = torch.where(dot > 0.9995, linear, interpolated)
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1.0e-8)


def matrix_to_rotation_6d_rows(matrix: torch.Tensor) -> torch.Tensor:
    """Flatten the first two matrix rows using the canonical row-Rot6D ABI."""
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_rows_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Project row-Rot6D values onto a right-handed rotation matrix."""
    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:]
    first = first / torch.linalg.vector_norm(first, dim=-1, keepdim=True).clamp_min(1.0e-8)
    second = second - (first * second).sum(dim=-1, keepdim=True) * first
    second = second / torch.linalg.vector_norm(second, dim=-1, keepdim=True).clamp_min(1.0e-8)
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-2)
