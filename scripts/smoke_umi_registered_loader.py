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
        "observation.tactile.values": (1, 81, 4, 8, 25, 6),
        "observation.tactile.dt": (1, 81, 4, 8),
        "tactile_valid": (1, 81, 4, 8),
    }
    actual_shapes = {key: tuple(batch[key].shape) for key in expected_shapes}
    if actual_shapes != expected_shapes:
        raise AssertionError(f"Unexpected registered UMI batch shapes: {actual_shapes}")
    if not torch.all(batch["state_feature_mask"][:, :20]):
        raise AssertionError("UMI state TCP20 mask must be enabled.")
    if not torch.all(batch["action_feature_mask"][:, :20]):
        raise AssertionError("UMI action TCP20 mask must be enabled.")
    if torch.isnan(batch["observation.tactile.dt"]).any():
        raise AssertionError("Aligned UMI tactile dt must not contain NaN.")
    if torch.any(
        batch["observation.tactile.values"][~batch["tactile_valid"]] != 0
    ):
        raise AssertionError("Invalid UMI tactile slots must contain zero values.")
    tactile_valid = batch["tactile_valid"]
    if not torch.any(tactile_valid):
        raise AssertionError("UMI smoke sample must contain aligned tactile observations.")
    valid_dt = batch["observation.tactile.dt"][tactile_valid]
    if torch.any(valid_dt > 1.0e-6) or torch.any(valid_dt <= -0.1 - 1.0e-6):
        raise AssertionError("Aligned UMI tactile dt must remain in (-0.1, 0].")
    print(
        f"registered_umi_loader_smoke=ok samples={len(loader.dataset)} "
        f"shapes={actual_shapes} tactile_valid={int(tactile_valid.sum())}/"
        f"{tactile_valid.numel()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
