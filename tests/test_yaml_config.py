from pathlib import Path

import pytest

from ngad_canonical_dataloader.config import (
    CONFIG_SCHEMA_VERSION,
    DatasetRootConfig,
    load_dataset_config,
)


def test_example_yaml_matches_the_strict_config_contract() -> None:
    path = Path(__file__).parents[1] / "configs" / "canonical.yaml"
    config = load_dataset_config(path)
    assert CONFIG_SCHEMA_VERSION == "ngad_canonical_dataloader_v2"
    assert config.timeline.rgb_rate_hz == 10
    assert config.timeline.action_steps_per_rgb_frame == 2
    assert config.timeline.anchor_offset == 0
    assert config.timeline.frame_ranges[-1] == (-32, 16)
    assert config.dataset_dirs[0].name == "libero"
    assert (
        config.dataset_dirs[0].mask_and_mapping_path
        == "/path/to/canonical/libero/mask_and_mapping.json"
    )


def test_legacy_mask_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="mask_and_mapping_path"):
        DatasetRootConfig.from_mapping(
            {
                "name": "legacy",
                "path": "/data/legacy",
                "mask_path": "/data/legacy/mask.json",
                "normalization_stats_path": "/data/legacy/stats.json",
            }
        )
