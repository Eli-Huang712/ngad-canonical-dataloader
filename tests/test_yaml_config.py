from pathlib import Path

from ngad_canonical_dataloader.config import (
    CONFIG_SCHEMA_VERSION,
    load_dataset_config,
)


def test_example_yaml_matches_the_strict_config_contract() -> None:
    path = Path(__file__).parents[1] / "configs" / "canonical.yaml"
    config = load_dataset_config(path)
    assert CONFIG_SCHEMA_VERSION == "ngad_canonical_dataloader_v1"
    assert config.target_rgb_fps == 10
    assert config.target_action_fps == 20
    assert config.dataset_dirs[0].name == "libero"
    assert config.dataset_dirs[0].mask_path == "/path/to/canonical/libero/mask.json"
    assert config.action_horizon == 32
