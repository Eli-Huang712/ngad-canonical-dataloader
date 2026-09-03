"""Training-facing DataLoader factory backed by registered Dataset YAMLs."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset, DataLoader

from ngad_canonical_dataloader.config import build_dataset_from_yaml


HY_TABLE_DATASET_NAMES = (
    "hy_table_000",
    "hy_table_001",
    "hy_table_002",
    "hy_table_003",
    "hy_table_004",
    "hy_table_005",
    "hy_table_007",
    "hy_table_008",
    "hy_table_009",
    "hy_table_010",
    "hy_table_011",
    "hy_table_012",
    "hy_table_013",
    "hy_table_015",
    "hy_table_016",
    "hy_table_017",
    "hy_table_018",
    "hy_table_020",
    "hy_table_021",
)
SUPPORTED_DATASET_NAMES = (
    *HY_TABLE_DATASET_NAMES,
    "hy_all",
    "umi_selfcollect",
    "libero",
)

# Training code selects only stable dataset names. Deployment owns these roots;
# environment overrides relocate a complete config set without changing model code.
DEFAULT_HY_CONFIG_ROOT = Path(
    "/gpfs/jiuquyun/datasets/PRETRAIN_DATA/Hy-Embodied-0.5-VLA-Data/"
    "dataset_configs/configs"
)
DEFAULT_UMI_CONFIG_PATH = Path(
    "/gpfs/jiuquyun/datasets/PRETRAIN_DATA/UMI-Collectsite-KS3-canonical-v3/"
    "dataset_configs_v2/configs/umi_table_000.yaml"
)
DEFAULT_LIBERO_CONFIG_PATH = Path(
    "/gpfs/jiuquyun/datasets/PRETRAIN_DATA/LIBERO/"
    "dataset_configs/configs/libero.yaml"
)


def registered_dataset_config_paths() -> dict[str, Path]:
    """Return the current deployment's stable dataset-name registry."""
    hy_root = Path(
        os.environ.get("NGAD_HY_CONFIG_ROOT", str(DEFAULT_HY_CONFIG_ROOT))
    ).expanduser().resolve()
    umi_path = Path(
        os.environ.get("NGAD_UMI_CONFIG_PATH", str(DEFAULT_UMI_CONFIG_PATH))
    ).expanduser().resolve()
    libero_path = Path(
        os.environ.get("NGAD_LIBERO_CONFIG_PATH", str(DEFAULT_LIBERO_CONFIG_PATH))
    ).expanduser().resolve()
    registry = {name: hy_root / f"{name}.yaml" for name in HY_TABLE_DATASET_NAMES}
    registry["umi_selfcollect"] = umi_path
    registry["libero"] = libero_path
    return registry


def resolve_dataset_names(dataset_names: str | Sequence[str]) -> tuple[str, ...]:
    """Validate explicit names and expand the ``hy_all`` aggregate."""
    if isinstance(dataset_names, str):
        names = (dataset_names,)
    else:
        names = tuple(str(name) for name in dataset_names)
    if not names:
        raise ValueError("dataset_names must contain at least one registered name.")
    unknown = sorted(set(names).difference(SUPPORTED_DATASET_NAMES))
    if unknown:
        raise ValueError(
            f"Unsupported dataset names {unknown}; expected entries from "
            f"{SUPPORTED_DATASET_NAMES}."
        )
    if "hy_all" in names:
        if names != ("hy_all",):
            raise ValueError("hy_all must be selected alone; it already expands every HY table.")
        return HY_TABLE_DATASET_NAMES
    if len(set(names)) != len(names):
        raise ValueError("dataset_names must not contain duplicates.")
    return names


def resolve_registered_config_paths(
    dataset_names: str | Sequence[str],
) -> tuple[Path, ...]:
    """Resolve stable names to existing deployment-owned Dataset YAMLs."""
    names = resolve_dataset_names(dataset_names)
    registry = registered_dataset_config_paths()
    paths = tuple(registry[name] for name in names)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Registered Dataset YAMLs do not exist; update the deployment registry "
            "or NGAD_* path overrides: " + ", ".join(missing)
        )
    return paths


def build_dataloader_from_yamls(
    *,
    config_paths: Sequence[str | Path],
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader from already-resolved per-dataset YAMLs."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")
    paths = tuple(Path(path).expanduser().resolve() for path in config_paths)
    if not paths:
        raise ValueError("config_paths must contain at least one Dataset YAML.")
    datasets = [build_dataset_from_yaml(path) for path in paths]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    worker_options: dict[str, Any] = {}
    if num_workers > 0:
        worker_options = {
            "multiprocessing_context": "spawn",
            "persistent_workers": True,
            "prefetch_factor": 2,
        }
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=bool(shuffle),
        **worker_options,
    )


def build_dataloader(
    *,
    dataset_names: str | Sequence[str],
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    """Build a training DataLoader using registered dataset names only."""
    return build_dataloader_from_yamls(
        config_paths=resolve_registered_config_paths(dataset_names),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )
