"""Standalone public API for NGAD canonical Dataset loading."""

from ngad_canonical_dataloader.datasets import NGADCanonicalDataset
from ngad_canonical_dataloader.config import (
    DatasetConfig,
    DatasetRootConfig,
    TimelineConfig,
    build_dataset_from_yaml,
    load_dataset_config,
)
from ngad_canonical_dataloader.loader import (
    HY_TABLE_DATASET_NAMES,
    SUPPORTED_DATASET_NAMES,
    build_dataloader,
    build_dataloader_from_yamls,
    registered_dataset_config_paths,
    resolve_dataset_names,
    resolve_registered_config_paths,
)

__all__ = [
    "DatasetConfig",
    "DatasetRootConfig",
    "NGADCanonicalDataset",
    "TimelineConfig",
    "HY_TABLE_DATASET_NAMES",
    "SUPPORTED_DATASET_NAMES",
    "build_dataloader",
    "build_dataloader_from_yamls",
    "build_dataset_from_yaml",
    "load_dataset_config",
    "registered_dataset_config_paths",
    "resolve_dataset_names",
    "resolve_registered_config_paths",
]
