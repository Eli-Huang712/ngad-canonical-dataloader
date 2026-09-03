"""Smoke-test all HY tables against staged v2 z-score statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch
import yaml
from torch.utils.data import DataLoader

from ngad_canonical_dataloader import build_dataset_from_yaml


def _temporary_config(
    config_path: Path,
    stats_path: Path,
    directory: Path,
    *,
    max_samples: int | None = None,
) -> Path:
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = value["dataset"]["dataset_dirs"][0]
    source_mask_path = Path(root["mask_and_mapping_path"])
    mask = json.loads(source_mask_path.read_text(encoding="utf-8"))
    mask["field_mask"]["action"] = True
    mask["element_mask"]["action"] = [True] * 20
    pixel_path = Path(mask["image_pixel_mask"]["path"])
    if not pixel_path.is_absolute():
        pixel_path = (source_mask_path.parent / pixel_path).resolve()
    mask["image_pixel_mask"]["path"] = str(pixel_path)

    mask_path = directory / f"{config_path.stem}-mask.json"
    mask_path.write_text(json.dumps(mask, indent=2) + "\n", encoding="utf-8")
    root["mask_and_mapping_path"] = str(mask_path)
    value["dataset"]["normalization_stats_path"] = str(stats_path)
    if max_samples is not None:
        value["dataset"]["max_samples"] = int(max_samples)
    temporary_config = directory / config_path.name
    temporary_config.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return temporary_config


def _validate_sample(dataset, sample: dict) -> dict[str, float]:
    if sample["video"].shape != (81, 6, 3, 256, 256):
        raise AssertionError(f"Unexpected video shape {tuple(sample['video'].shape)}")
    for key in ("state", "action"):
        if sample[key].shape != (81, 2, 128):
            raise AssertionError(f"Unexpected {key} shape {tuple(sample[key].shape)}")
        if not torch.isfinite(sample[key]).all():
            raise AssertionError(f"Non-finite {key} values")
    if not sample["state_feature_mask"][:20].all():
        raise AssertionError("State TCP20 mask is not fully enabled")
    if not sample["action_feature_mask"][:20].all():
        raise AssertionError("Action TCP20 mask is not fully enabled")
    if sample["state_feature_mask"][20:].any() or sample["action_feature_mask"][20:].any():
        raise AssertionError("Reserved TCP128 features must remain masked")

    valid = sample["action_valid"]
    for tensor_name in ("state", "action"):
        openness = sample[tensor_name][..., [9, 19]][valid]
        if openness.numel() and (openness.min() < 0 or openness.max() > 1):
            raise AssertionError(f"{tensor_name} openness left [0,1]")
    if not torch.any(sample["action"][valid][..., :20].abs() > 1.0e-7):
        raise AssertionError("All valid Action values are zero")

    denormalized = dataset.denormalize_action(sample["action"])
    zero_position = torch.nonzero(
        sample["action_step_offsets"] == 0, as_tuple=False
    )
    if zero_position.shape != (1, 2):
        raise AssertionError("Timeline must contain exactly one zero action offset")
    frame_position, substep_position = zero_position[0].tolist()
    current = denormalized[frame_position, substep_position]
    expected_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    if current[[0, 1, 2, 10, 11, 12]].abs().max() > 1.0e-5:
        raise AssertionError("Anchor relative translation is not zero")
    torch.testing.assert_close(current[3:9], expected_rotation, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(current[13:19], expected_rotation, atol=1.0e-5, rtol=1.0e-5)
    return {
        "state_abs_max": float(sample["state"][valid].abs().max()),
        "action_abs_max": float(sample["action"][valid].abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-root", required=True, type=Path)
    parser.add_argument("--stats-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()

    results = []
    with tempfile.TemporaryDirectory(prefix="hy-zscore-validation-") as temporary:
        temporary_root = Path(temporary)
        configs = sorted(args.config_root.glob("hy_table_*.yaml"))
        for config_path in configs:
            stats_path = args.stats_root / f"{config_path.stem}.json"
            config = (
                config_path
                if args.direct
                else _temporary_config(config_path, stats_path, temporary_root)
            )
            dataset = build_dataset_from_yaml(config)
            sample_index = len(dataset) // 2
            sample = dataset[sample_index]
            metrics = _validate_sample(dataset, sample)
            results.append({
                "table": config_path.stem,
                "dataset_samples": len(dataset),
                "sample_index": sample_index,
                **metrics,
            })
            print(f"PASS {config_path.stem} {metrics}", flush=True)

        first_value = yaml.safe_load(configs[0].read_text(encoding="utf-8"))
        first_stats = (
            Path(first_value["dataset"]["normalization_stats_path"])
            if args.direct
            else args.stats_root / f"{configs[0].stem}.json"
        )
        first_config = _temporary_config(
            configs[0],
            first_stats,
            temporary_root,
            max_samples=1,
        )
        first_dataset = build_dataset_from_yaml(first_config)
        loader = DataLoader(
            first_dataset,
            batch_size=1,
            num_workers=2,
            multiprocessing_context="spawn",
        )
        iterator = iter(loader)
        batch = next(iterator)
        if batch["state"].shape != (1, 81, 2, 128):
            raise AssertionError(
                f"Unexpected worker batch shape {tuple(batch['state'].shape)}"
            )
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise AssertionError("One-sample multiworker Dataset yielded extra batches")

    report = {
        "tables": results,
        "table_count": len(results),
        "multiworker_spawn": "passed",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote={args.report}", flush=True)


if __name__ == "__main__":
    main()
