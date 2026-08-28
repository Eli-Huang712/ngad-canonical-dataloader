"""Canonical TCP geometry, normalization, and fixed 128D feature ABI."""

from __future__ import annotations

import torch

from ngad_canonical_dataloader import rotation as rotation_utils


WAM_FEATURE_DIM = 128
TCP_FEATURE_DIM = 10
DUAL_ARM_TCP_FEATURE_DIM = 2 * TCP_FEATURE_DIM


def minmax_normalize(value: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
    """Normalize featurewise values to ``[-1, 1]`` with stable constant dimensions."""
    value_range = maximum - minimum
    stable_range = torch.where(value_range < 1e-4, torch.full_like(value_range, 2.0), value_range)
    normalized = 2.0 * (value - minimum) / stable_range - 1.0
    normalized = torch.where(value_range < 1e-4, value - minimum, normalized)
    return normalized.clamp(-5.0, 5.0)


def minmax_denormalize(value: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
    """Invert :func:`minmax_normalize` with the same constant-dimension rule."""
    value_range = maximum - minimum
    stable_range = torch.where(value_range < 1e-4, torch.full_like(value_range, 2.0), value_range)
    denormalized = (value + 1.0) * stable_range / 2.0 + minimum
    return torch.where(value_range < 1e-4, value + minimum, denormalized)


def tcp_target_relative_to_anchor(anchor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Express target TCP poses in the fixed anchor TCP frame."""
    anchor_rotation = rotation_utils.rotation_6d_rows_to_matrix(anchor[..., 3:9])
    target_rotation = rotation_utils.rotation_6d_rows_to_matrix(target[..., 3:9])
    anchor_inverse_rotation = anchor_rotation.transpose(-1, -2)
    relative_position = (
        anchor_inverse_rotation @ (target[..., :3] - anchor[..., :3]).unsqueeze(-1)
    ).squeeze(-1)
    relative_rotation = anchor_inverse_rotation @ target_rotation
    return torch.cat(
        [
            relative_position,
            rotation_utils.matrix_to_rotation_6d_rows(relative_rotation),
            target[..., 9:10],
        ],
        dim=-1,
    )


def tcp_chunk_relative_to_first(tcp_sequence: torch.Tensor) -> torch.Tensor:
    """Use the chunk's first TCP pose as one anchor for every action slot."""
    return tcp_target_relative_to_anchor(tcp_sequence[..., :1, :], tcp_sequence)


def dual_arm_tcp_target_relative_to_anchor(anchor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Express each arm's target TCP10 in that arm's fixed anchor frame."""
    if anchor.shape[-1] != DUAL_ARM_TCP_FEATURE_DIM or target.shape[-1] != DUAL_ARM_TCP_FEATURE_DIM:
        raise ValueError("Dual-arm relative TCP conversion requires 20D anchor and target tensors.")
    return torch.cat(
        [
            tcp_target_relative_to_anchor(anchor[..., :TCP_FEATURE_DIM], target[..., :TCP_FEATURE_DIM]),
            tcp_target_relative_to_anchor(anchor[..., TCP_FEATURE_DIM:], target[..., TCP_FEATURE_DIM:]),
        ],
        dim=-1,
    )


def dual_arm_tcp_chunk_relative_to_first(tcp_sequence: torch.Tensor) -> torch.Tensor:
    """Use each arm's first TCP pose as the fixed anchor for a dual-arm chunk."""
    return dual_arm_tcp_target_relative_to_anchor(tcp_sequence[..., :1, :], tcp_sequence)


def tcp_relative_to_absolute(anchor: torch.Tensor, relative: torch.Tensor) -> torch.Tensor:
    """Reconstruct an absolute target with T_target = T_anchor @ T_relative."""
    anchor_rotation = rotation_utils.rotation_6d_rows_to_matrix(anchor[..., 3:9])
    relative_rotation = rotation_utils.rotation_6d_rows_to_matrix(relative[..., 3:9])
    target_position = anchor[..., :3] + (
        anchor_rotation @ relative[..., :3].unsqueeze(-1)
    ).squeeze(-1)
    target_rotation = anchor_rotation @ relative_rotation
    return torch.cat(
        [
            target_position,
            rotation_utils.matrix_to_rotation_6d_rows(target_rotation),
            relative[..., 9:10],
        ],
        dim=-1,
    )


def normalize_absolute_tcp(
    tcp: torch.Tensor,
    xyz_minimum: torch.Tensor,
    xyz_maximum: torch.Tensor,
) -> torch.Tensor:
    """Normalize absolute xyz while preserving Rot6D and absolute openness."""
    normalized = tcp.clone()
    normalized[..., :3] = minmax_normalize(tcp[..., :3], xyz_minimum, xyz_maximum)
    normalized[..., 9] = normalized[..., 9].clamp(0.0, 1.0)
    return normalized


def denormalize_absolute_tcp(
    tcp: torch.Tensor,
    xyz_minimum: torch.Tensor,
    xyz_maximum: torch.Tensor,
) -> torch.Tensor:
    """Restore physical xyz while preserving Rot6D and absolute openness."""
    denormalized = tcp.clone()
    denormalized[..., :3] = minmax_denormalize(tcp[..., :3], xyz_minimum, xyz_maximum)
    denormalized[..., 9] = denormalized[..., 9].clamp(0.0, 1.0)
    return denormalized


def normalize_relative_tcp(tcp: torch.Tensor, xyz_scale: torch.Tensor) -> torch.Tensor:
    """Symmetrically scale fixed-anchor relative xyz around zero."""
    normalized = tcp.clone()
    normalized[..., :3] = tcp[..., :3] / xyz_scale.clamp_min(1.0e-6)
    normalized[..., 9] = normalized[..., 9].clamp(0.0, 1.0)
    return normalized


def denormalize_relative_tcp(tcp: torch.Tensor, xyz_scale: torch.Tensor) -> torch.Tensor:
    """Restore relative xyz in meters while keeping Rot6D and openness unchanged."""
    denormalized = tcp.clone()
    denormalized[..., :3] = tcp[..., :3] * xyz_scale
    denormalized[..., 9] = denormalized[..., 9].clamp(0.0, 1.0)
    return denormalized


def normalize_dual_arm_absolute_tcp(
    tcp: torch.Tensor,
    xyz_minimum: torch.Tensor,
    xyz_maximum: torch.Tensor,
) -> torch.Tensor:
    """Normalize the xyz block of each arm while preserving Rot6D and openness."""
    if tcp.shape[-1] != DUAL_ARM_TCP_FEATURE_DIM or xyz_minimum.shape != (2, 3) or xyz_maximum.shape != (2, 3):
        raise ValueError("Dual-arm absolute normalization requires TCP20 and [2,3] xyz statistics.")
    return torch.cat(
        [
            normalize_absolute_tcp(tcp[..., :TCP_FEATURE_DIM], xyz_minimum[0], xyz_maximum[0]),
            normalize_absolute_tcp(tcp[..., TCP_FEATURE_DIM:], xyz_minimum[1], xyz_maximum[1]),
        ],
        dim=-1,
    )


def normalize_dual_arm_relative_tcp(tcp: torch.Tensor, xyz_scale: torch.Tensor) -> torch.Tensor:
    """Normalize each arm's fixed-anchor xyz with independent symmetric scales."""
    if tcp.shape[-1] != DUAL_ARM_TCP_FEATURE_DIM or xyz_scale.shape != (2, 3):
        raise ValueError("Dual-arm relative normalization requires TCP20 and [2,3] xyz scales.")
    return torch.cat(
        [
            normalize_relative_tcp(tcp[..., :TCP_FEATURE_DIM], xyz_scale[0]),
            normalize_relative_tcp(tcp[..., TCP_FEATURE_DIM:], xyz_scale[1]),
        ],
        dim=-1,
    )


def denormalize_dual_arm_relative_tcp(tcp: torch.Tensor, xyz_scale: torch.Tensor) -> torch.Tensor:
    """Restore both arms' relative xyz from independent symmetric scales."""
    if tcp.shape[-1] != DUAL_ARM_TCP_FEATURE_DIM or xyz_scale.shape != (2, 3):
        raise ValueError("Dual-arm relative denormalization requires TCP20 and [2,3] xyz scales.")
    return torch.cat(
        [
            denormalize_relative_tcp(tcp[..., :TCP_FEATURE_DIM], xyz_scale[0]),
            denormalize_relative_tcp(tcp[..., TCP_FEATURE_DIM:], xyz_scale[1]),
        ],
        dim=-1,
    )


def pack_single_arm_tcp(tcp: torch.Tensor) -> torch.Tensor:
    """Place single-arm TCP10 in slots 0:10 and zero reserved 10:128."""
    packed = torch.zeros(*tcp.shape[:-1], WAM_FEATURE_DIM, dtype=tcp.dtype, device=tcp.device)
    packed[..., :TCP_FEATURE_DIM] = tcp
    return packed


def pack_dual_arm_tcp(tcp: torch.Tensor) -> torch.Tensor:
    """Place left/right TCP10 blocks in slots 0:20 and zero reserved 20:128."""
    if tcp.shape[-1] != DUAL_ARM_TCP_FEATURE_DIM:
        raise ValueError(
            f"Dual-arm TCP must have {DUAL_ARM_TCP_FEATURE_DIM} features, got {tcp.shape[-1]}."
        )
    packed = torch.zeros(*tcp.shape[:-1], WAM_FEATURE_DIM, dtype=tcp.dtype, device=tcp.device)
    packed[..., :DUAL_ARM_TCP_FEATURE_DIM] = tcp
    return packed


def element_mask_to_feature_mask(element_mask: torch.Tensor) -> torch.Tensor:
    """Place the canonical TCP20 element mask in the active TCP128 slots."""
    if element_mask.shape[-1] != DUAL_ARM_TCP_FEATURE_DIM:
        raise ValueError(
            "Canonical element mask must end with 20 entries, "
            f"got {tuple(element_mask.shape)}."
        )
    element_mask = element_mask.to(dtype=torch.bool)
    feature_mask = torch.zeros(
        *element_mask.shape[:-1],
        WAM_FEATURE_DIM,
        dtype=torch.bool,
        device=element_mask.device,
    )
    feature_mask[..., :DUAL_ARM_TCP_FEATURE_DIM] = element_mask
    return feature_mask
