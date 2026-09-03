#!/usr/bin/env python3
"""Benchmark adjacent full UMI samples whose five timeline segments are valid."""

from __future__ import annotations

import gc
import time

import torch

from ngad_canonical_dataloader.config import build_dataset_from_yaml
from ngad_canonical_dataloader.loader import DEFAULT_UMI_CONFIG_PATH


def main() -> None:
    torch.set_num_threads(4)
    dataset = build_dataset_from_yaml(DEFAULT_UMI_CONFIG_PATH)
    minimum_anchor = 232
    future_frames = 16
    episode_position = next(
        position
        for position, episode in enumerate(dataset._episodes)
        if episode["rgb_target_length"] > minimum_anchor + future_frames + 3
    )
    episode = dataset._episodes[episode_position]
    previous_end = 0 if episode_position == 0 else dataset._episode_window_ends[episode_position - 1]
    sample_index = previous_end + minimum_anchor
    print(
        f"episode_position={episode_position} episode_index={episode['episode_index']} "
        f"source_length={episode['length']} rgb_length={episode['rgb_target_length']} "
        f"sample_index={sample_index}",
        flush=True,
    )
    for adjacent_offset in range(3):
        started = time.perf_counter()
        sample = dataset[sample_index + adjacent_offset]
        elapsed = time.perf_counter() - started
        print(
            f"offset={adjacent_offset} elapsed={elapsed:.6f} "
            f"video_shape={tuple(sample['video'].shape)} "
            f"video_sum={sample['video'].sum().item():.6f}",
            flush=True,
        )
        del sample
        gc.collect()


if __name__ == "__main__":
    main()
