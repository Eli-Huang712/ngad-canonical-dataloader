from pathlib import Path

import pytest

from ngad_canonical_dataloader.config import (
    DatasetConfig,
    DatasetRootConfig,
    TimelineConfig,
    load_dataset_config,
)


def test_example_yaml_matches_the_strict_config_contract() -> None:
    path = Path(__file__).parents[1] / "configs" / "canonical.yaml"
    config = load_dataset_config(path)
    assert config.timeline.rgb_rate_hz == 10
    assert config.timeline.action_steps_per_rgb_frame == 2
    assert config.timeline.tactile_steps_per_rgb_frame == 8
    assert config.timeline.anchor_offset == 0
    assert config.timeline.frame_ranges[-1] == (-32, 16)
    assert (
        config.normalization_stats_path
        == "/path/to/stats/canonical_global_normalization.json"
    )
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
            }
        )


def test_dataset_root_rejects_per_root_normalization_and_table_name() -> None:
    for extra in (
        {"normalization_stats_path": "/data/stats.json"},
        {"table_name": "table_000"},
    ):
        with pytest.raises(ValueError, match="exactly"):
            DatasetRootConfig.from_mapping(
                {
                    "name": "hy_table_000",
                    "path": "/data/table_000",
                    "mask_and_mapping_path": "/data/mask.json",
                    **extra,
                }
            )


def test_global_normalization_is_required_and_forwarded() -> None:
    timeline = {
        "rgb_rate_hz": 10,
        "action_steps_per_rgb_frame": 2,
        "anchor_offset": 0,
        "frame_ranges": [[0, 0]],
    }
    root = {
        "name": "hy_table_000",
        "path": "/data/table_000",
        "mask_and_mapping_path": "/data/mask.json",
    }
    with pytest.raises(ValueError, match="normalization_stats_path"):
        DatasetConfig.from_mapping({"dataset_dirs": [root], "timeline": timeline})
    with pytest.raises(ValueError, match="non-empty string"):
        DatasetConfig.from_mapping(
            {
                "normalization_stats_path": "",
                "dataset_dirs": [root],
                "timeline": timeline,
            }
        )

    config = DatasetConfig(
        normalization_stats_path="/data/global-stats.json",
        dataset_dirs=(DatasetRootConfig.from_mapping(root),),
        timeline=TimelineConfig.from_mapping(timeline),
    )

    assert config.to_dataset_kwargs()["normalization_stats_path"] == "/data/global-stats.json"
    assert set(config.to_dataset_kwargs()["dataset_dirs"][0]) == {
        "name",
        "path",
        "mask_and_mapping_path",
    }


def test_null_global_normalization_round_trips_as_video_only(tmp_path) -> None:
    path = tmp_path / "video-only.yaml"
    path.write_text(
        """dataset:
  normalization_stats_path: null
  dataset_dirs:
    - name: hy_table_000
      path: /data/table_000
      mask_and_mapping_path: /data/mask.json
  timeline:
    rgb_rate_hz: 10
    action_steps_per_rgb_frame: 2
    anchor_offset: 0
    frame_ranges:
      - [0, 0]
""",
        encoding="utf-8",
    )

    config = load_dataset_config(path)

    assert config.normalization_stats_path is None
    assert config.to_dataset_kwargs()["normalization_stats_path"] is None


def test_schema_version_is_rejected_without_compatibility(tmp_path) -> None:
    path = tmp_path / "versioned.yaml"
    path.write_text(
        """schema_version: ngad_canonical_dataloader_v2
dataset: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly dataset"):
        load_dataset_config(path)
