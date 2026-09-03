"""Publish staged HY v2 statistics as the only shared DataLoader config set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from ngad_canonical_dataloader.config import load_dataset_config
from ngad_canonical_dataloader.datasets.canonical import NGADCanonicalDataset


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def _write_json_in_place(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _table_ids(config_root: Path) -> list[str]:
    return [path.stem.rsplit("_", 1)[-1] for path in sorted(config_root.glob("hy_table_*.yaml"))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--staging-root", required=True, type=Path)
    parser.add_argument("--backup-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        raise SystemExit("Refusing to publish without --apply")

    config_root = args.dataset_root / "dataset_configs"
    source_configs = config_root / "configs_canonical_state"
    source_masks = config_root / "masks_canonical_state"
    target_configs = config_root / "configs"
    target_masks = config_root / "masks"
    normalization = config_root / "normalization"
    expected_ids = _table_ids(source_configs)
    if len(expected_ids) != 19 or len(set(expected_ids)) != 19:
        raise RuntimeError(f"Expected 19 unique canonical HY configs, got {expected_ids}")
    expected_config_names = {f"hy_table_{table_id}.yaml" for table_id in expected_ids}
    expected_mask_names = {
        f"mask_and_mapping_hy_table_{table_id}.json" for table_id in expected_ids
    }
    expected_stat_names = {f"hy_table_{table_id}.json" for table_id in expected_ids}
    if {p.name for p in source_configs.glob("*.yaml")} != expected_config_names:
        raise RuntimeError("Canonical config set does not match expected HY tables")
    if {p.name for p in source_masks.glob("mask_and_mapping_*.json")} != expected_mask_names:
        raise RuntimeError("Canonical mask set does not match expected HY tables")
    if {p.name for p in args.staging_root.glob("hy_table_*.json")} != expected_stat_names:
        raise RuntimeError("Staged statistics set does not match expected HY tables")
    if args.backup_root.exists():
        raise FileExistsError(f"Backup destination already exists: {args.backup_root}")

    args.backup_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config_root, args.backup_root)

    for table_id in expected_ids:
        config_name = f"hy_table_{table_id}.yaml"
        source_config = yaml.safe_load(
            (source_configs / config_name).read_text(encoding="utf-8")
        )
        dataset_entry = source_config["dataset"]["dataset_dirs"][0]
        dataset_entry["mask_and_mapping_path"] = str(
            target_masks / f"mask_and_mapping_hy_table_{table_id}.json"
        )
        source_config["dataset"]["normalization_stats_path"] = str(
            normalization / f"table_{table_id}.json"
        )
        with (target_configs / config_name).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(source_config, handle, sort_keys=False)

        mask_path = target_masks / f"mask_and_mapping_hy_table_{table_id}.json"
        mask = _read_json(mask_path)
        if "action" in mask["field_mapping"]:
            raise RuntimeError(f"Derived Action must not be mapped in {mask_path}")
        mask["field_mask"]["observation.state"] = True
        mask["field_mask"]["action"] = True
        mask["element_mask"]["observation.state"] = [True] * 20
        mask["element_mask"]["action"] = [True] * 20
        _write_json_in_place(mask_path, mask)

        stats = _read_json(args.staging_root / f"hy_table_{table_id}.json")
        NGADCanonicalDataset._normalization_transform(stats)
        _write_json_in_place(normalization / f"table_{table_id}.json", stats)

    for config_path in sorted(target_configs.glob("hy_table_*.yaml")):
        load_dataset_config(config_path)
    if len(list(target_configs.glob("hy_table_*.yaml"))) != 19:
        raise RuntimeError("Published config count is not 19")

    shutil.rmtree(source_configs)
    shutil.rmtree(source_masks)

    manifest_entries = []
    for directory in (target_configs, target_masks, normalization):
        for path in sorted(p for p in directory.iterdir() if p.is_file()):
            payload = path.read_bytes()
            manifest_entries.append({
                "path": str(path.relative_to(config_root)),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            })
    _write_json_in_place(config_root / "CANONICAL_STATE_MANIFEST.json", manifest_entries)
    print(
        json.dumps(
            {
                "published_tables": len(expected_ids),
                "manifest_entries": len(manifest_entries),
                "backup": str(args.backup_root),
                "removed_shared_directories": [
                    str(source_configs),
                    str(source_masks),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
