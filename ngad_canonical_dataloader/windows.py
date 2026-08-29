"""Episode splits and the anchor-relative canonical timeline layout."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class TimelineLayout:
    """One fixed set of semantic RGB offsets and per-frame action substeps."""

    frame_offsets: torch.Tensor
    action_step_offsets: torch.Tensor
    action_steps_per_rgb_frame: int
    offset_to_position: dict[int, int]

    def position(self, offset: int) -> int:
        """Return the tensor position of one semantic frame offset."""
        return self.offset_to_position[int(offset)]


@dataclass(frozen=True)
class TimelineSampleIndices:
    """Episode-local target-grid indices and validity for one sampled anchor."""

    frame_indices: torch.Tensor
    frame_valid: torch.Tensor
    action_indices: torch.Tensor
    action_valid: torch.Tensor


def build_timeline_layout(
    frame_ranges: Sequence[Sequence[int]],
    action_steps_per_rgb_frame: int,
) -> TimelineLayout:
    """Expand inclusive ranges into one chronological anchor-relative layout."""
    steps_per_frame = int(action_steps_per_rgb_frame)
    if steps_per_frame <= 0:
        raise ValueError("action_steps_per_rgb_frame must be positive.")
    if not frame_ranges:
        raise ValueError("frame_ranges must contain at least one inclusive range.")

    ranges: list[tuple[int, int]] = []
    previous_end: int | None = None
    for value in frame_ranges:
        if len(value) != 2:
            raise ValueError("Each frame range must contain exactly [start, end].")
        start, end = int(value[0]), int(value[1])
        if start > end:
            raise ValueError(f"Frame range start {start} exceeds end {end}.")
        if previous_end is not None and start <= previous_end:
            raise ValueError("frame_ranges must be chronological and non-overlapping.")
        ranges.append((start, end))
        previous_end = end

    frame_offsets = torch.cat(
        [torch.arange(start, end + 1, dtype=torch.long) for start, end in ranges]
    )
    if not torch.any(frame_offsets == 0):
        raise ValueError("frame_ranges must include anchor offset 0 exactly once.")
    intra_frame_steps = torch.arange(
        1 - steps_per_frame,
        1,
        dtype=torch.long,
    )
    action_step_offsets = (
        frame_offsets[:, None] * steps_per_frame + intra_frame_steps[None, :]
    )
    return TimelineLayout(
        frame_offsets=frame_offsets,
        action_step_offsets=action_step_offsets,
        action_steps_per_rgb_frame=steps_per_frame,
        offset_to_position={
            int(offset): position
            for position, offset in enumerate(frame_offsets.tolist())
        },
    )


def timeline_sample_indices(
    anchor_rgb_index: int,
    *,
    rgb_episode_length: int,
    action_episode_length: int,
    layout: TimelineLayout,
) -> TimelineSampleIndices:
    """Instantiate one relative layout without crossing its current episode."""
    if rgb_episode_length <= 0 or action_episode_length <= 0:
        raise ValueError("Episode lengths must be positive.")
    if not 0 <= anchor_rgb_index < rgb_episode_length:
        raise ValueError(
            f"anchor_rgb_index must be in [0,{rgb_episode_length}), got {anchor_rgb_index}."
        )

    frame_raw = int(anchor_rgb_index) + layout.frame_offsets
    frame_valid = (frame_raw >= 0) & (frame_raw < rgb_episode_length)
    anchor_action_index = (
        int(anchor_rgb_index) * layout.action_steps_per_rgb_frame
    )
    action_raw = anchor_action_index + layout.action_step_offsets
    action_valid = (action_raw >= 0) & (action_raw < action_episode_length)
    return TimelineSampleIndices(
        frame_indices=frame_raw.clamp(0, rgb_episode_length - 1),
        frame_valid=frame_valid,
        action_indices=action_raw.clamp(0, action_episode_length - 1),
        action_valid=action_valid,
    )
