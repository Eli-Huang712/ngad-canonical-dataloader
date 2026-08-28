"""Hy-Embodied source fields to canonical dual-arm TCP20 conversion."""

from __future__ import annotations

import torch

from ngad_canonical_dataloader.rotation import matrix_to_rotation_6d_rows, quaternion_xyzw_to_matrix


HY_GRIPPER_CLOSED_MM = 90.0
HY_SOURCE_STATE_DIM = 16


def _validate_source_tensor(value: torch.Tensor, *, expected_dim: int, name: str) -> None:
    """Reject malformed source fields before any normalization can hide them."""
    if value.shape[-1] != expected_dim:
        raise ValueError(f"{name} must have {expected_dim} features, got {value.shape[-1]}.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values.")


def hy_gripper_mm_to_open_fraction(gripper_mm: torch.Tensor) -> torch.Tensor:
    """Map Hy absolute closure travel, 0 mm open and 90 mm closed, to openness."""
    if not torch.isfinite(gripper_mm).all():
        raise ValueError("Hy gripper values contain non-finite entries.")
    tolerance = 1.0e-3
    if torch.any(gripper_mm < -tolerance) or torch.any(gripper_mm > HY_GRIPPER_CLOSED_MM + tolerance):
        minimum = float(gripper_mm.amin().item())
        maximum = float(gripper_mm.amax().item())
        raise ValueError(f"Hy gripper values must be in [0, 90] mm, got [{minimum}, {maximum}].")
    return (1.0 - gripper_mm.clamp(0.0, HY_GRIPPER_CLOSED_MM) / HY_GRIPPER_CLOSED_MM).clamp(0.0, 1.0)


def _arm_state_to_tcp(
    state: torch.Tensor,
    *,
    position_start: int,
    quaternion_start: int,
    gripper_index: int,
) -> torch.Tensor:
    """Convert one xyz/xyzw/gripper block to xyz/row-Rot6D/openness."""
    quaternion = state[..., quaternion_start : quaternion_start + 4]
    quaternion_norm = torch.linalg.vector_norm(quaternion, dim=-1)
    if torch.any(quaternion_norm < 1.0e-8):
        raise ValueError("Hy observation.state contains a near-zero quaternion.")
    return torch.cat(
        [
            state[..., position_start : position_start + 3],
            matrix_to_rotation_6d_rows(quaternion_xyzw_to_matrix(quaternion)),
            hy_gripper_mm_to_open_fraction(state[..., gripper_index : gripper_index + 1]),
        ],
        dim=-1,
    )


def hy_state_to_tcp(state: torch.Tensor) -> torch.Tensor:
    """Map Hy state16 to canonical absolute state [...,2,10]."""
    _validate_source_tensor(state, expected_dim=HY_SOURCE_STATE_DIM, name="observation.state")
    left = _arm_state_to_tcp(state, position_start=0, quaternion_start=3, gripper_index=7)
    right = _arm_state_to_tcp(state, position_start=8, quaternion_start=11, gripper_index=15)
    return torch.stack([left, right], dim=-2)
