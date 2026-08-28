"""Episode split and anchor-window helpers extracted from NGADv1pp."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch


def split_episode_indices(
    episode_indices: Sequence[int], validation_split: float, seed: int
) -> tuple[set[int], set[int]]:
    """Split episode identifiers deterministically into train and validation sets."""
    episode_indices = sorted({int(index) for index in episode_indices})
    ratio = float(validation_split)
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"validation_split must be in [0, 1), got {ratio}.")
    if ratio == 0.0:
        return set(episode_indices), set()
    if len(episode_indices) < 2:
        raise ValueError("Episode-level validation requires at least two episodes per dataset root.")

    validation_count = max(
        1,
        min(len(episode_indices) - 1, int(len(episode_indices) * ratio + 0.5)),
    )
    ranked = sorted(
        episode_indices,
        key=lambda index: hashlib.sha256(f"{int(seed)}:{index}".encode()).digest(),
    )
    validation = set(ranked[:validation_count])
    return set(episode_indices).difference(validation), validation


def wam_window_indices(
    anchor_rgb_index: int,
    *,
    rgb_episode_length: int,
    action_episode_length: int,
    action_horizon: int,
    target_rgb_fps: float,
    target_action_fps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the canonical RGB and future-action window around one real RGB anchor."""
    if rgb_episode_length <= 0 or action_episode_length <= 0 or action_horizon <= 0:
        raise ValueError("Episode lengths and action_horizon must be positive.")
    if target_rgb_fps <= 0 or target_action_fps <= 0:
        raise ValueError("target_rgb_fps and target_action_fps must be positive.")
    if not 0 <= anchor_rgb_index < rgb_episode_length:
        raise ValueError(
            f"anchor_rgb_index must be in [0,{rgb_episode_length}), got {anchor_rgb_index}."
        )

    action_steps_per_rgb = target_action_fps / target_rgb_fps
    rounded_ratio = int(round(action_steps_per_rgb))
    if rounded_ratio <= 0 or abs(action_steps_per_rgb - rounded_ratio) > 1.0e-9:
        raise ValueError("target_action_fps / target_rgb_fps must be a positive integer.")
    if action_horizon % rounded_ratio:
        raise ValueError("action_horizon must be divisible by the action/RGB rate ratio.")

    anchor_action_index = anchor_rgb_index * rounded_ratio
    action_raw = anchor_action_index + torch.arange(1, action_horizon + 1, dtype=torch.long)
    observation_raw = anchor_rgb_index + torch.arange(
        action_horizon // rounded_ratio + 1, dtype=torch.long
    )
    action_is_pad = action_raw >= action_episode_length
    image_is_pad = observation_raw >= rgb_episode_length
    return (
        observation_raw.clamp(max=rgb_episode_length - 1),
        action_raw.clamp(max=action_episode_length - 1),
        image_is_pad,
        action_is_pad,
    )
