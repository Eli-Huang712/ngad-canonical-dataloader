#!/usr/bin/env python3
"""Smoke-test the registered full canonical UMI DataLoader."""

from __future__ import annotations

import torch

from ngad_canonical_dataloader import build_dataloader


def main() -> None:
    loader = build_dataloader(
        dataset_names="umi_selfcollect",
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )
    batch = next(iter(loader))
    expected_shapes = {
        "video": (1, 81, 6, 3, 256, 256),
        "state": (1, 81, 2, 128),
        "action": (1, 81, 2, 128),
        "observation.tactile.values": (1, 4, 3, 25, 6),
        "observation.tactile.dt": (1, 4, 3),
    }
    actual_shapes = {key: tuple(batch[key].shape) for key in expected_shapes}
    if actual_shapes != expected_shapes:
        raise AssertionError(f"Unexpected registered UMI batch shapes: {actual_shapes}")
    if not torch.all(batch["state_feature_mask"][:, :20]):
        raise AssertionError("UMI state TCP20 mask must be enabled.")
    if not torch.all(batch["action_feature_mask"][:, :20]):
        raise AssertionError("UMI action TCP20 mask must be enabled.")
    print(
        f"registered_umi_loader_smoke=ok samples={len(loader.dataset)} "
        f"shapes={actual_shapes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
