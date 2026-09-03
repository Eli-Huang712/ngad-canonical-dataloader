#!/usr/bin/env python3
"""Smoke-test the name-only training DataLoader against deployed HY configs."""

from __future__ import annotations

from torch.utils.data import ConcatDataset

from ngad_canonical_dataloader import HY_TABLE_DATASET_NAMES, build_dataloader


def main() -> None:
    single_loader = build_dataloader(
        dataset_names="hy_table_000",
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )
    batch = next(iter(single_loader))
    expected_shapes = {
        "video": (1, 81, 6, 3, 256, 256),
        "state": (1, 81, 2, 128),
        "action": (1, 81, 2, 128),
    }
    actual_shapes = {key: tuple(batch[key].shape) for key in expected_shapes}
    if actual_shapes != expected_shapes:
        raise AssertionError(
            f"Unexpected registered-loader batch shapes: {actual_shapes}"
        )

    all_loader = build_dataloader(
        dataset_names="hy_all",
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )
    if not isinstance(all_loader.dataset, ConcatDataset):
        raise AssertionError("hy_all must create one ConcatDataset.")
    table_count = len(all_loader.dataset.datasets)
    if table_count != len(HY_TABLE_DATASET_NAMES):
        raise AssertionError(
            f"hy_all resolved {table_count} tables, expected "
            f"{len(HY_TABLE_DATASET_NAMES)}."
        )

    print(
        "registered_loader_smoke=ok "
        f"single_shapes={actual_shapes} "
        f"hy_tables={table_count} total_samples={len(all_loader.dataset)}"
    )


if __name__ == "__main__":
    main()
