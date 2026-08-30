"""Strict YAML configuration for the single canonical Dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ngad_canonical_dataloader.datasets.canonical import NGADCanonicalDataset


CONFIG_SCHEMA_VERSION = "ngad_canonical_dataloader_v2"


@dataclass(frozen=True)
class TimelineConfig:
    """One anchor-relative RGB grid with an integer action substep axis."""

    rgb_rate_hz: float
    action_steps_per_rgb_frame: int
    anchor_offset: int
    frame_ranges: tuple[tuple[int, int], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TimelineConfig":
        expected = {
            "rgb_rate_hz",
            "action_steps_per_rgb_frame",
            "anchor_offset",
            "frame_ranges",
        }
        if set(value) != expected:
            raise ValueError(f"timeline must contain exactly {sorted(expected)}.")
        ranges_value = value["frame_ranges"]
        if not isinstance(ranges_value, list):
            raise TypeError("timeline.frame_ranges must be a list of [start, end] pairs.")
        frame_ranges = tuple(
            tuple(int(bound) for bound in frame_range)
            for frame_range in ranges_value
        )
        return cls(
            rgb_rate_hz=float(value["rgb_rate_hz"]),
            action_steps_per_rgb_frame=int(value["action_steps_per_rgb_frame"]),
            anchor_offset=int(value["anchor_offset"]),
            frame_ranges=frame_ranges,
        )


@dataclass(frozen=True)
class DatasetRootConfig:
    """One named root with its field contract and normalization statistics."""

    name: str
    path: str
    mask_and_mapping_path: str
    normalization_stats_path: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetRootConfig":
        expected = {
            "name",
            "path",
            "mask_and_mapping_path",
            "normalization_stats_path",
        }
        if set(value) != expected:
            raise ValueError(f"Each dataset root must contain exactly {sorted(expected)}.")
        return cls(
            name=str(value["name"]),
            path=str(value["path"]),
            mask_and_mapping_path=str(value["mask_and_mapping_path"]),
            normalization_stats_path=str(value["normalization_stats_path"]),
        )

    def to_dataset_entry(self) -> dict[str, str]:
        return {
            "name": self.name,
            "path": self.path,
            "mask_and_mapping_path": self.mask_and_mapping_path,
            "normalization_stats_path": self.normalization_stats_path,
        }


@dataclass(frozen=True)
class DatasetConfig:
    """All arguments required to construct :class:`NGADCanonicalDataset`."""

    dataset_dirs: tuple[DatasetRootConfig, ...]
    timeline: TimelineConfig
    max_samples: int | None = None
    validation_split: float = 0.0
    validation_seed: int = 3407
    split: str = "train"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetConfig":
        allowed = {
            "dataset_dirs",
            "timeline",
            "max_samples",
            "validation_split",
            "validation_seed",
            "split",
        }
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown dataset configuration fields: {sorted(unknown)}.")
        required = {"dataset_dirs", "timeline"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"Missing dataset configuration fields: {sorted(missing)}.")
        roots_value = value["dataset_dirs"]
        if not isinstance(roots_value, list) or not roots_value:
            raise ValueError("dataset_dirs must be a non-empty list.")
        roots = tuple(DatasetRootConfig.from_mapping(root) for root in roots_value)
        return cls(
            dataset_dirs=roots,
            timeline=TimelineConfig.from_mapping(value["timeline"]),
            max_samples=(
                None if value.get("max_samples") is None else int(value["max_samples"])
            ),
            validation_split=float(value.get("validation_split", 0.0)),
            validation_seed=int(value.get("validation_seed", 3407)),
            split=str(value.get("split", "train")),
        )

    def to_dataset_kwargs(self) -> dict[str, Any]:
        return {
            "dataset_dirs": [root.to_dataset_entry() for root in self.dataset_dirs],
            "rgb_rate_hz": self.timeline.rgb_rate_hz,
            "action_steps_per_rgb_frame": self.timeline.action_steps_per_rgb_frame,
            "anchor_offset": self.timeline.anchor_offset,
            "frame_ranges": self.timeline.frame_ranges,
            "max_samples": self.max_samples,
            "validation_split": self.validation_split,
            "validation_seed": self.validation_seed,
            "split": self.split,
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
