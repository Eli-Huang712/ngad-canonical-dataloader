"""Strict YAML configuration for the single canonical Dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ngad_canonical_dataloader.datasets.canonical import (
    CANONICAL_CAMERA_KEYS,
    CANONICAL_IMAGE_SIZE,
    NGADCanonicalDataset,
)


CONFIG_SCHEMA_VERSION = "ngad_canonical_dataloader_v1"


@dataclass(frozen=True)
class DatasetRootConfig:
    """One named physical dataset root and its external normalization statistics."""

    name: str
    path: str
    normalization_stats_path: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetRootConfig":
        expected = {"name", "path", "normalization_stats_path"}
        if set(value) != expected:
            raise ValueError(f"Each dataset root must contain exactly {sorted(expected)}.")
        return cls(
            name=str(value["name"]),
            path=str(value["path"]),
            normalization_stats_path=str(value["normalization_stats_path"]),
        )

    def to_dataset_entry(self) -> dict[str, str]:
        return {
            "name": self.name,
            "path": self.path,
            "normalization_stats_path": self.normalization_stats_path,
        }


@dataclass(frozen=True)
class DatasetConfig:
    """All arguments required to construct :class:`NGADCanonicalDataset`."""

    dataset_dirs: tuple[DatasetRootConfig, ...]
    target_rgb_fps: float
    target_action_fps: float
    camera_keys: tuple[str, ...] = CANONICAL_CAMERA_KEYS
    num_frames: int = 17
    action_horizon: int = 32
    recent_memory_frames: int = 24
    long_memory_anchor_interval_frames: int = 50
    long_memory_window_frames: int = 8
    long_memory_slots: int = 5
    action_history_horizon: int = 10
    max_samples: int | None = None
    validation_split: float = 0.0
    validation_seed: int = 3407
    split: str = "train"
    resolution: int = CANONICAL_IMAGE_SIZE

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetConfig":
        allowed = {
            "dataset_dirs",
            "target_rgb_fps",
            "target_action_fps",
            "camera_keys",
            "num_frames",
            "action_horizon",
            "recent_memory_frames",
            "long_memory_anchor_interval_frames",
            "long_memory_window_frames",
            "long_memory_slots",
            "action_history_horizon",
            "max_samples",
            "validation_split",
            "validation_seed",
            "split",
            "resolution",
        }
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown dataset configuration fields: {sorted(unknown)}.")
        required = {"dataset_dirs", "target_rgb_fps", "target_action_fps"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"Missing dataset configuration fields: {sorted(missing)}.")
        roots_value = value["dataset_dirs"]
        if not isinstance(roots_value, list) or not roots_value:
            raise ValueError("dataset_dirs must be a non-empty list.")
        roots = tuple(DatasetRootConfig.from_mapping(root) for root in roots_value)
        camera_keys = tuple(value.get("camera_keys", CANONICAL_CAMERA_KEYS))
        return cls(
            dataset_dirs=roots,
            target_rgb_fps=float(value["target_rgb_fps"]),
            target_action_fps=float(value["target_action_fps"]),
            camera_keys=camera_keys,
            num_frames=int(value.get("num_frames", 17)),
            action_horizon=int(value.get("action_horizon", 32)),
            recent_memory_frames=int(value.get("recent_memory_frames", 24)),
            long_memory_anchor_interval_frames=int(
                value.get("long_memory_anchor_interval_frames", 50)
            ),
            long_memory_window_frames=int(value.get("long_memory_window_frames", 8)),
            long_memory_slots=int(value.get("long_memory_slots", 5)),
            action_history_horizon=int(value.get("action_history_horizon", 10)),
            max_samples=(
                None if value.get("max_samples") is None else int(value["max_samples"])
            ),
            validation_split=float(value.get("validation_split", 0.0)),
            validation_seed=int(value.get("validation_seed", 3407)),
            split=str(value.get("split", "train")),
            resolution=int(value.get("resolution", CANONICAL_IMAGE_SIZE)),
        )

    def to_dataset_kwargs(self) -> dict[str, Any]:
        return {
            "dataset_dirs": [root.to_dataset_entry() for root in self.dataset_dirs],
            "target_rgb_fps": self.target_rgb_fps,
            "target_action_fps": self.target_action_fps,
            "camera_keys": list(self.camera_keys),
            "num_frames": self.num_frames,
            "action_horizon": self.action_horizon,
            "recent_memory_frames": self.recent_memory_frames,
            "long_memory_anchor_interval_frames": self.long_memory_anchor_interval_frames,
            "long_memory_window_frames": self.long_memory_window_frames,
            "long_memory_slots": self.long_memory_slots,
            "action_history_horizon": self.action_history_horizon,
            "max_samples": self.max_samples,
            "validation_split": self.validation_split,
            "validation_seed": self.validation_seed,
            "split": self.split,
            "resolution": self.resolution,
        }


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Read and strictly validate one versioned YAML Dataset configuration."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or set(value) != {"schema_version", "dataset"}:
        raise ValueError("YAML root must contain exactly schema_version and dataset.")
    if value["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Expected schema_version={CONFIG_SCHEMA_VERSION!r}, "
            f"got {value['schema_version']!r}."
        )
    if not isinstance(value["dataset"], dict):
        raise TypeError("YAML dataset field must be a mapping.")
    return DatasetConfig.from_mapping(value["dataset"])


def build_dataset_from_yaml(path: str | Path) -> NGADCanonicalDataset:
    """Construct the canonical Dataset from one strict YAML file."""
    config = load_dataset_config(path)
    return NGADCanonicalDataset(**config.to_dataset_kwargs())

