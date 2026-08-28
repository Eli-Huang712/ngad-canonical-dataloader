"""LIBERO source-state conversion into the canonical TCP representation."""

from __future__ import annotations

import torch

from ngad_canonical_dataloader.rotation import axis_angle_to_matrix, matrix_to_rotation_6d_rows


LIBERO_TCP_SCHEMA = "libero_tcp128_v1"


def libero_gripper_qpos_to_open_fraction(gripper_qpos: torch.Tensor) -> torch.Tensor:
    """Collapse Panda's signed finger joints into one aperture in [0, 1]."""
    return ((gripper_qpos[..., 0] - gripper_qpos[..., 1]) / 0.08).clamp(0.0, 1.0)


def tcp_from_position_rotation_gripper(
    position: torch.Tensor,
    rotation: torch.Tensor,
    gripper_qpos: torch.Tensor,
) -> torch.Tensor:
    """Build TCP10 from LIBERO grip-site pose and Panda finger positions."""
    return torch.cat(
        [
            position,
            matrix_to_rotation_6d_rows(rotation),
            libero_gripper_qpos_to_open_fraction(gripper_qpos).unsqueeze(-1),
        ],
        dim=-1,
    )


def libero_state_to_tcp(state: torch.Tensor) -> torch.Tensor:
    """Map state8 xyz/axis-angle/qpos2 to TCP10 xyz/row-Rot6D/openness."""
    return tcp_from_position_rotation_gripper(
        state[..., :3],
        axis_angle_to_matrix(state[..., 3:6]),
        state[..., 6:8],
    )
