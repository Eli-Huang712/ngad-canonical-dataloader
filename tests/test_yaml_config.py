from pathlib import Path

from ngad_canonical_dataloader.config import (
    CONFIG_SCHEMA_VERSION,
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
    assert config.dataset_dirs[0].mask_path == "/path/to/canonical/libero/mask.json"
