from pathlib import Path

import pytest
import torch
from torch.utils.data import ConcatDataset, TensorDataset

from ngad_canonical_dataloader import (
    HY_TABLE_DATASET_NAMES,
    build_dataloader,
    resolve_dataset_names,
    resolve_registered_config_paths,
)
from ngad_canonical_dataloader import loader as loader_module


def test_hy_all_expands_the_fixed_19_table_set() -> None:
    assert resolve_dataset_names("hy_all") == HY_TABLE_DATASET_NAMES


def test_explicit_registered_names_preserve_order() -> None:
    assert resolve_dataset_names(["hy_table_002", "hy_table_000"]) == (
        "hy_table_002",
        "hy_table_000",
    )


def test_dataset_names_reject_unknown_duplicates_and_mixed_hy_all() -> None:
    with pytest.raises(ValueError, match="Unsupported dataset"):
        resolve_dataset_names("hy_table_006")
    with pytest.raises(ValueError, match="duplicates"):
        resolve_dataset_names(["hy_table_000", "hy_table_000"])
    with pytest.raises(ValueError, match="selected alone"):
        resolve_dataset_names(["hy_all", "hy_table_000"])


def test_registered_names_resolve_without_call_site_paths(tmp_path, monkeypatch) -> None:
    hy_root = tmp_path / "hy"
    hy_root.mkdir()
    for name in HY_TABLE_DATASET_NAMES:
        (hy_root / f"{name}.yaml").touch()
    umi_path = tmp_path / "umi.yaml"
    umi_path.touch()
    libero_path = tmp_path / "libero.yaml"
    libero_path.touch()
    monkeypatch.setenv("NGAD_HY_CONFIG_ROOT", str(hy_root))
    monkeypatch.setenv("NGAD_UMI_CONFIG_PATH", str(umi_path))
    monkeypatch.setenv("NGAD_LIBERO_CONFIG_PATH", str(libero_path))

    assert resolve_registered_config_paths("hy_table_000") == (
        (hy_root / "hy_table_000.yaml").resolve(),
    )
    assert len(resolve_registered_config_paths("hy_all")) == 19
    assert resolve_registered_config_paths("umi_selfcollect") == (umi_path.resolve(),)
    assert resolve_registered_config_paths("libero") == (libero_path.resolve(),)


def test_build_dataloader_exposes_batch_workers_and_shuffle(tmp_path, monkeypatch) -> None:
    hy_root = tmp_path / "hy"
    hy_root.mkdir()
    for name in HY_TABLE_DATASET_NAMES:
        (hy_root / f"{name}.yaml").touch()
    monkeypatch.setenv("NGAD_HY_CONFIG_ROOT", str(hy_root))
    monkeypatch.setattr(
        loader_module,
        "build_dataset_from_yaml",
        lambda _: TensorDataset(torch.arange(6)),
    )
    dataloader = build_dataloader(
        dataset_names="hy_table_000",
        batch_size=2,
        num_workers=0,
        shuffle=False,
    )
    assert dataloader.batch_size == 2
    assert dataloader.num_workers == 0
    assert next(iter(dataloader))[0].tolist() == [0, 1]


def test_hy_all_keeps_one_normalized_dataset_per_table(tmp_path, monkeypatch) -> None:
    hy_root = tmp_path / "hy"
    hy_root.mkdir()
    for name in HY_TABLE_DATASET_NAMES:
        (hy_root / f"{name}.yaml").touch()
    monkeypatch.setenv("NGAD_HY_CONFIG_ROOT", str(hy_root))
    monkeypatch.setattr(
        loader_module,
        "build_dataset_from_yaml",
        lambda _: TensorDataset(torch.arange(1)),
    )
    dataloader = build_dataloader(
        dataset_names="hy_all",
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )
    assert isinstance(dataloader.dataset, ConcatDataset)
    assert len(dataloader.dataset.datasets) == 19
