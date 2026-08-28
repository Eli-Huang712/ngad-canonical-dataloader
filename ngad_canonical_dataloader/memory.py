"""Episode-bounded temporal indices for SANA-WAM training memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WAMMemoryIndices:
    """Chronological past-only RGB and reached-state indices plus validity."""

    recent_rgb: torch.Tensor
    recent_valid: torch.Tensor
    long_rgb: torch.Tensor
    long_valid: torch.Tensor
    action_history: torch.Tensor
    action_history_valid: torch.Tensor


def wam_memory_indices(
    anchor_rgb_index: int,
    *,
    rgb_episode_length: int,
    action_episode_length: int,
    target_rgb_fps: float,
    target_action_fps: float,
    recent_memory_frames: int,
    long_memory_anchor_interval_frames: int,
    long_memory_window_frames: int,
    long_memory_slots: int,
    action_history_horizon: int,
) -> WAMMemoryIndices:
    """Select strict-past memory without crossing the current episode boundary."""
    sizes = (
        rgb_episode_length,
        action_episode_length,
        recent_memory_frames,
        long_memory_anchor_interval_frames,
        long_memory_window_frames,
        long_memory_slots,
        action_history_horizon,
    )
    if any(int(value) <= 0 for value in sizes):
        raise ValueError("Episode lengths and all memory sizes must be positive.")
    if target_rgb_fps <= 0 or target_action_fps <= 0:
        raise ValueError("Target RGB/action rates must be positive.")
    if not 0 <= anchor_rgb_index < rgb_episode_length:
        raise ValueError(
            f"anchor_rgb_index must be in [0,{rgb_episode_length}), got {anchor_rgb_index}."
        )
    rate_ratio = target_action_fps / target_rgb_fps
    rounded_ratio = int(round(rate_ratio))
    if rounded_ratio <= 0 or abs(rate_ratio - rounded_ratio) > 1.0e-9:
        raise ValueError("target_action_fps / target_rgb_fps must be a positive integer.")

    recent_raw = anchor_rgb_index - torch.arange(
        recent_memory_frames, 0, -1, dtype=torch.long
    )
    recent_valid = recent_raw >= 0
    recent_rgb = recent_raw.clamp(0, rgb_episode_length - 1)

    latest_long_anchor_limit = anchor_rgb_index - recent_memory_frames - 1
    if latest_long_anchor_limit >= 0:
        latest_long_anchor = (
            latest_long_anchor_limit // long_memory_anchor_interval_frames
        ) * long_memory_anchor_interval_frames
    else:
        latest_long_anchor = -long_memory_anchor_interval_frames
    long_anchors = latest_long_anchor - torch.arange(
        long_memory_slots - 1, -1, -1, dtype=torch.long
    ) * long_memory_anchor_interval_frames
    offsets = torch.arange(
        long_memory_window_frames - 1, -1, -1, dtype=torch.long
    )
    long_raw = long_anchors[:, None] - offsets[None, :]
    complete_slots = long_raw[:, 0] >= 0
    long_valid = complete_slots[:, None].expand_as(long_raw).clone()
    long_rgb = long_raw.clamp(0, rgb_episode_length - 1)

    anchor_action_index = anchor_rgb_index * rounded_ratio
    history_raw = anchor_action_index - torch.arange(
        action_history_horizon, 0, -1, dtype=torch.long
    )
    action_history_valid = history_raw >= 0
    action_history = history_raw.clamp(0, action_episode_length - 1)
    return WAMMemoryIndices(
        recent_rgb=recent_rgb,
        recent_valid=recent_valid,
        long_rgb=long_rgb,
        long_valid=long_valid,
        action_history=action_history,
        action_history_valid=action_history_valid,
    )
