import torch

from ngad_canonical_dataloader import NGADCanonicalDataset
from ngad_canonical_dataloader.tcp import pack_dual_arm_tcp
from ngad_canonical_dataloader.windows import wam_window_indices


def test_dataset_classes_are_importable() -> None:
    assert NGADCanonicalDataset.__name__ == "NGADCanonicalDataset"


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
