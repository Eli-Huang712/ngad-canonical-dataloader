import json

import numpy as np
import torch

from ngad_canonical_dataloader import NGADCanonicalDataset
from ngad_canonical_dataloader.datasets.canonical import (
    CANONICAL_CAMERA_KEYS,
    CANONICAL_TACTILE_DT_KEY,
    CANONICAL_TACTILE_VALUES_KEY,
)
from ngad_canonical_dataloader.action import element_mask_to_feature_mask, pack_dual_arm_tcp
from ngad_canonical_dataloader.windows import wam_window_indices


def test_dataset_classes_are_importable() -> None:
    assert NGADCanonicalDataset.__name__ == "NGADCanonicalDataset"


def test_canonical_feature_names_match_the_published_contract() -> None:
    assert CANONICAL_CAMERA_KEYS == (
        "observation.images.cam_head_left",
        "observation.images.cam_head_right",
        "observation.images.cam_left_wrist_left",
        "observation.images.cam_left_wrist_right",
        "observation.images.cam_right_wrist_left",
        "observation.images.cam_right_wrist_right",
    )
    assert CANONICAL_TACTILE_VALUES_KEY == "observation.tactile.values"
    assert CANONICAL_TACTILE_DT_KEY == "observation.tactile.dt"


def test_tcp128_and_window_helpers_remain_available() -> None:
    packed = pack_dual_arm_tcp(torch.zeros(20))
    observations, actions, image_is_pad, action_is_pad = wam_window_indices(
        0,
        rgb_episode_length=17,
        action_episode_length=33,
        action_horizon=32,
        target_rgb_fps=10,
        target_action_fps=20,
    )
    assert packed.shape == (128,)
    assert observations.shape == (17,)
    assert actions.shape == (32,)
    assert not image_is_pad.any()
    assert not action_is_pad.any()


def test_tcp20_element_mask_is_preserved_in_tcp128() -> None:
    element_mask = torch.tensor([True, False] * 10)
    feature_mask = element_mask_to_feature_mask(element_mask)
    assert feature_mask.shape == (128,)
    assert torch.equal(feature_mask[:20], element_mask)
    assert not feature_mask[20:].any()


def test_canonical_sidecar_produces_tensor_masks(tmp_path) -> None:
    np.savez(
        tmp_path / "image_pixel_mask_libero.npz",
        mask=np.ones((256, 256), dtype=np.bool_),
    )
    available_cameras = {CANONICAL_CAMERA_KEYS[0], CANONICAL_CAMERA_KEYS[2]}
    field_mask = {
        camera: camera in available_cameras for camera in CANONICAL_CAMERA_KEYS
    }
    field_mask.update(
        {
            "observation.state": True,
            "action": True,
            CANONICAL_TACTILE_VALUES_KEY: False,
            CANONICAL_TACTILE_DT_KEY: False,
            "timestamp": True,
            "frame_index": True,
            "episode_index": True,
            "index": True,
            "task_index": True,
        }
    )
    manifest = {
        "dataset": "libero",
        "image_pixel_mask": {
            "path": "image_pixel_mask_libero.npz",
            "key": "mask",
            "shape": [256, 256],
            "applies_to_all_available_images": True,
        },
        "field_mask": field_mask,
        "element_mask": {
            "observation.state": [True] * 10 + [False] * 10,
            "action": [True] * 10 + [False] * 10,
        },
    }
    manifest_path = tmp_path / "mask.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = NGADCanonicalDataset.__new__(NGADCanonicalDataset)
    dataset.camera_keys = CANONICAL_CAMERA_KEYS
    dataset.resolution = 256
    contract = dataset._load_mask_contract(manifest_path, "libero")
    pixel_mask = dataset._load_pixel_mask(contract)

    assert torch.equal(
        contract["camera_mask"],
        torch.tensor([True, False, True, False, False, False]),
    )
    assert torch.equal(contract["tactile_mask"], torch.tensor([False, False]))
    assert torch.equal(
        contract["state_element_mask"],
        torch.tensor([True] * 10 + [False] * 10),
    )
    assert pixel_mask.shape == (256, 256)
    assert pixel_mask.all()
