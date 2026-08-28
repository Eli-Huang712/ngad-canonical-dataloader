"""UMI source state to canonical dual-arm TCP20 conversion."""

from __future__ import annotations

import torch

from ngad_canonical_dataloader.rotation import matrix_to_rotation_6d_rows, quaternion_xyzw_to_matrix


UMI_SOURCE_STATE_DIM = 16
UMI_GRIPPER_OPEN_POSITION = 105.0


def umi_gripper_position_to_open_fraction(position: torch.Tensor) -> torch.Tensor:
    """Map the Cora encoder range, 0 closed and 105 open, to absolute openness."""
    return (position / UMI_GRIPPER_OPEN_POSITION).clamp(0.0, 1.0)


def _arm_state_to_tcp(
    state: torch.Tensor,
    *,
    position_start: int,
    quaternion_start: int,
    gripper_index: int,
) -> torch.Tensor:
    """Convert one UMI xyz/xyzw/gripper block to canonical TCP10."""
    quaternion = state[..., quaternion_start : quaternion_start + 4]
    if torch.any(torch.linalg.vector_norm(quaternion, dim=-1) < 1.0e-8):
        raise ValueError("UMI observation.state contains a near-zero quaternion.")
    return torch.cat(
        [
            state[..., position_start : position_start + 3],
            matrix_to_rotation_6d_rows(quaternion_xyzw_to_matrix(quaternion)),
            umi_gripper_position_to_open_fraction(
                state[..., gripper_index : gripper_index + 1]
            ),
        ],
        dim=-1,
    )


def umi_state_to_tcp(state: torch.Tensor) -> torch.Tensor:
    """Map UMI state16 to canonical absolute state [...,2,10]."""
    if state.shape[-1] != UMI_SOURCE_STATE_DIM:
        raise ValueError(
            f"UMI observation.state must have {UMI_SOURCE_STATE_DIM} features, "
            f"got {state.shape[-1]}."
        )
    if not torch.isfinite(state).all():
        raise ValueError("UMI observation.state contains non-finite values.")
    # UMI appends both grippers after the two seven-dimensional wrist poses.
    left = _arm_state_to_tcp(state, position_start=0, quaternion_start=3, gripper_index=14)
    right = _arm_state_to_tcp(state, position_start=7, quaternion_start=10, gripper_index=15)
    return torch.stack([left, right], dim=-2)
