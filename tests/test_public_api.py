import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ngad_canonical_dataloader import NGADCanonicalDataset
from ngad_canonical_dataloader.backends import create_storage_backends
from ngad_canonical_dataloader.backends.image import H264ImageBackend, JpegImageBackend
from ngad_canonical_dataloader.backends.table import LanceTableBackend, ParquetTableBackend
from ngad_canonical_dataloader.datasets.canonical import (
    CANONICAL_CAMERA_KEYS,
    CANONICAL_TACTILE_DT_KEY,
    CANONICAL_TACTILE_VALUES_KEY,
    _discover_published_tables,
)
from ngad_canonical_dataloader.datasets import canonical as canonical_module
from ngad_canonical_dataloader.action import element_mask_to_feature_mask, pack_dual_arm_tcp
from ngad_canonical_dataloader.windows import (
    build_timeline_layout,
    timeline_sample_indices,
)


def test_dataset_classes_are_importable() -> None:
    assert NGADCanonicalDataset.__name__ == "NGADCanonicalDataset"


def test_fixed_canonical_abi_is_not_configurable() -> None:
    with pytest.raises(TypeError, match="target_rgb_fps"):
        NGADCanonicalDataset(
            dataset_dirs=[],
            normalization_stats_path="/data/global-stats.json",
            rgb_rate_hz=10,
            action_steps_per_rgb_frame=2,
            anchor_offset=0,
            frame_ranges=((0, 0),),
            target_rgb_fps=10,
        )


def test_storage_backend_factory_selects_only_supported_physical_pairs(tmp_path) -> None:
    lance_table_root = tmp_path / "lance" / "table_000"
    lance_root = lance_table_root / "table_000.lance"
    (lance_root / "_versions").mkdir(parents=True)
    (lance_root / "data").mkdir()
    (lance_root / "data" / "fragment.lance").touch()
    table_backend, image_backend = create_storage_backends(
        lance_table_root,
        "table_000",
        {
            "canonical_schema": "ngad_hy_canonical_lance_v2",
        },
    )
    assert isinstance(table_backend, LanceTableBackend)
    assert isinstance(image_backend, JpegImageBackend)
    assert table_backend.table_root == lance_table_root
    assert table_backend.lance_root == lance_root
    assert image_backend.feature_dtype == "image"

    parquet_root = tmp_path / "lerobot" / "table_001"
    (parquet_root / "data").mkdir(parents=True)
    (parquet_root / "videos").mkdir()
    table_backend, image_backend = create_storage_backends(
        parquet_root,
        "table_001",
        {
            "data_path": "data/{file_index}.parquet",
            "video_path": "videos/{video_key}.mp4",
        },
    )
    assert isinstance(table_backend, ParquetTableBackend)
    assert isinstance(image_backend, H264ImageBackend)
    assert image_backend.feature_dtype == "video"

    with pytest.raises(ValueError, match="exactly one"):
        create_storage_backends(tmp_path / "unknown", "table_002", {})


def test_dataset_root_discovers_direct_tables_in_numeric_order(tmp_path) -> None:
    table_root = tmp_path / "table_000"
    (table_root / "meta").mkdir(parents=True)
    second_table_root = tmp_path / "table_001"
    (second_table_root / "meta").mkdir(parents=True)
    (table_root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 2, "total_frames": 80}),
        encoding="utf-8",
    )
    (second_table_root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 1, "total_frames": 20}),
        encoding="utf-8",
    )
    records = _discover_published_tables(tmp_path)
    assert records == [
        {
            "table_index": 0,
            "table_name": "table_000",
            "table_root": table_root,
            "num_episodes": 2,
            "num_frames": 80,
        },
        {
            "table_index": 1,
            "table_name": "table_001",
            "table_root": second_table_root,
            "num_episodes": 1,
            "num_frames": 20,
        }
    ]


def test_single_table_root_is_returned_without_nested_discovery(tmp_path) -> None:
    table_root = tmp_path / "table_000"
    (table_root / "meta").mkdir(parents=True)
    (table_root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 2, "total_frames": 80}),
        encoding="utf-8",
    )
    nested = table_root / "table_001" / "meta"
    nested.mkdir(parents=True)
    (nested / "info.json").write_text(
        json.dumps({"total_episodes": 1, "total_frames": 20}),
        encoding="utf-8",
    )

    records = _discover_published_tables(table_root)

    assert len(records) == 1
    assert records[0]["table_name"] == "table_000"
    assert records[0]["table_root"] == table_root


def test_invalid_dataset_path_without_direct_tables_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="neither a table_NNN root"):
        _discover_published_tables(tmp_path)


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


def test_tcp128_and_unified_timeline_helpers_remain_available() -> None:
    packed = pack_dual_arm_tcp(torch.zeros(20))
    layout = build_timeline_layout(
        (
            (-232, -225),
            (-182, -175),
            (-132, -125),
            (-82, -75),
            (-32, 16),
        ),
        action_steps_per_rgb_frame=2,
    )
    sample = timeline_sample_indices(
        233,
        rgb_episode_length=300,
        action_episode_length=600,
        layout=layout,
    )
    assert packed.shape == (128,)
    assert layout.frame_offsets.shape == (81,)
    assert layout.action_step_offsets.shape == (81, 2)
    assert layout.position(-1) == 63
    assert layout.position(0) == 64
    assert layout.action_step_offsets[layout.position(-1)].tolist() == [-3, -2]
    assert layout.action_step_offsets[layout.position(0)].tolist() == [-1, 0]
    assert layout.action_step_offsets[layout.position(16)].tolist() == [31, 32]
    assert sample.frame_indices.shape == (81,)
    assert sample.action_indices.shape == (81, 2)
    assert sample.frame_valid.all()
    assert sample.action_valid.all()


def test_unified_timeline_masks_episode_boundaries_per_substep() -> None:
    layout = build_timeline_layout(((-1, 1),), action_steps_per_rgb_frame=2)
    sample = timeline_sample_indices(
        0,
        rgb_episode_length=2,
        action_episode_length=4,
        layout=layout,
    )
    assert sample.frame_valid.tolist() == [False, True, True]
    assert sample.action_valid.tolist() == [
        [False, False],
        [False, True],
        [True, True],
    ]


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
        "field_mapping": {
            CANONICAL_CAMERA_KEYS[0]: "observation.images.cam_head",
            CANONICAL_CAMERA_KEYS[2]: "observation.images.cam_left_wrist",
        },
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
    manifest_path = tmp_path / "mask_and_mapping.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = NGADCanonicalDataset.__new__(NGADCanonicalDataset)
    dataset.camera_keys = CANONICAL_CAMERA_KEYS
    contract = dataset._load_mask_and_mapping_contract(manifest_path, "libero")
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
    assert contract["field_mapping"] == manifest["field_mapping"]
    assert pixel_mask.shape == (256, 256)
    assert pixel_mask.all()

    features = {
        "observation.images.cam_head": {
            "dtype": "video",
            "shape": [256, 256, 3],
        },
        "observation.images.cam_left_wrist": {
            "dtype": "video",
            "shape": [256, 256, 3],
        },
        "observation.state": {"shape": [20]},
        "action": {"shape": [20]},
        "timestamp": {"shape": [1]},
        "frame_index": {"shape": [1]},
        "episode_index": {"shape": [1]},
        "index": {"shape": [1]},
        "task_index": {"shape": [1]},
    }
    dataset._validate_features(tmp_path, features, contract, "video")
    features.pop("observation.images.cam_head")
    with pytest.raises(ValueError, match="missing physical field"):
        dataset._validate_features(tmp_path, features, contract, "video")


def test_field_mapping_rejects_disabled_canonical_fields(tmp_path) -> None:
    np.savez(tmp_path / "pixel_mask.npz", mask=np.ones((256, 256), dtype=np.bool_))
    field_mask = {camera: camera == CANONICAL_CAMERA_KEYS[0] for camera in CANONICAL_CAMERA_KEYS}
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
    path = tmp_path / "mask_and_mapping.json"
    path.write_text(
        json.dumps(
            {
                "dataset": "invalid",
                "field_mapping": {
                    CANONICAL_CAMERA_KEYS[1]: "observation.images.cam_head"
                },
                "field_mask": field_mask,
                "element_mask": {
                    "observation.state": [True] * 20,
                    "action": [True] * 20,
                },
                "image_pixel_mask": {
                    "path": "pixel_mask.npz",
                    "key": "mask",
                    "shape": [256, 256],
                    "applies_to_all_available_images": True,
                },
            }
        ),
        encoding="utf-8",
    )
    dataset = NGADCanonicalDataset.__new__(NGADCanonicalDataset)
    dataset.camera_keys = CANONICAL_CAMERA_KEYS
    with pytest.raises(ValueError, match="contains disabled fields"):
        dataset._load_mask_and_mapping_contract(path, "invalid")


def test_canonical_video_preparation_only_normalizes_fixed_uint8_frames() -> None:
    dataset = NGADCanonicalDataset.__new__(NGADCanonicalDataset)
    video = torch.zeros((2, 3, 256, 256), dtype=torch.uint8)
    prepared = dataset._prepare_video(video)
    assert prepared.shape == video.shape
    assert prepared.dtype == torch.float32
    assert torch.all(prepared == -1.0)

    with pytest.raises(ValueError, match=r"uint8 \[T,3,256,256\]"):
        dataset._prepare_video(torch.zeros((2, 3, 224, 224), dtype=torch.uint8))


def test_dataset_returns_one_frame_aligned_timeline(tmp_path, monkeypatch) -> None:
    root = tmp_path / "canonical"
    table_root = root / "table_000"
    (table_root / "meta").mkdir(parents=True)
    second_table_root = root / "table_001"
    (second_table_root / "meta").mkdir(parents=True)
    features = {
        "observation.state": {"shape": [20]},
        "action": {"shape": [20]},
        "timestamp": {"shape": [1]},
        "frame_index": {"shape": [1]},
        "episode_index": {"shape": [1]},
        "index": {"shape": [1]},
        "task_index": {"shape": [1]},
    }
    features.update(
        {
            camera: {"dtype": "video", "shape": [256, 256, 3]}
            for camera in CANONICAL_CAMERA_KEYS
        }
    )
    (table_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "total_episodes": 1,
                "total_frames": 40,
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    (second_table_root / "meta" / "info.json").write_text(
        (table_root / "meta" / "info.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    np.savez(tmp_path / "pixel_mask.npz", mask=np.ones((256, 256), dtype=np.bool_))
    field_mask = {camera: True for camera in CANONICAL_CAMERA_KEYS}
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
    mask_and_mapping_path = tmp_path / "mask_and_mapping.json"
    mask_and_mapping_path.write_text(
        json.dumps(
            {
                "dataset": "synthetic",
                "field_mapping": {},
                "image_pixel_mask": {
                    "path": "pixel_mask.npz",
                    "key": "mask",
                    "shape": [256, 256],
                    "applies_to_all_available_images": True,
                },
                "field_mask": field_mask,
                "element_mask": {
                    "observation.state": [True] * 20,
                    "action": [True] * 20,
                },
            }
        ),
        encoding="utf-8",
    )
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "schema_version": "ngad_canonical_tcp_v1",
                "state_xyz_min": [[-1, -1, -1], [-1, -1, -1]],
                "state_xyz_max": [[1, 1, 1], [1, 1, 1]],
                "action_xyz_scale": [[1, 1, 1], [1, 1, 1]],
            }
        ),
        encoding="utf-8",
    )

    class FakeTableBackend:
        def read_catalog(self, camera_keys, camera_mask, field_mapping):
            del camera_keys, camera_mask, field_mapping
            return {0: "synthetic task"}, [
                {
                    "episode_index": 0,
                    "length": 40,
                    "dataset_from_index": 0,
                    "dataset_to_index": 40,
                }
            ]

        def read_rows(
            self,
            episode,
            relative_indices,
            field_mask,
            field_mapping,
            camera_keys,
            camera_mask,
        ):
            del episode, field_mask, field_mapping, camera_keys, camera_mask
            rows = {}
            for frame in sorted({0, *relative_indices.tolist()}):
                tcp10 = [
                    frame / 10,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.5,
                ]
                rows[frame] = {
                    "timestamp": frame / 10,
                    "frame_index": frame,
                    "episode_index": 0,
                    "index": frame,
                    "task_index": 0,
                    "observation.state": tcp10 + tcp10,
                }
            return rows

    class FakeImageBackend:
        feature_dtype = "video"

        def read_camera(self, rows, episode, camera, source_indices, source_fps):
            del rows, episode, camera, source_fps
            return torch.zeros(
                (source_indices.numel(), 3, 256, 256),
                dtype=torch.uint8,
            )

    monkeypatch.setattr(
        canonical_module,
        "create_storage_backends",
        lambda root, table_name, info: (
            FakeTableBackend(),
            FakeImageBackend(),
        ),
    )
    original_read_json = canonical_module._read_json_object
    stats_reads = 0

    def counted_read_json(path):
        nonlocal stats_reads
        if path == stats_path.resolve():
            stats_reads += 1
        return original_read_json(path)

    monkeypatch.setattr(canonical_module, "_read_json_object", counted_read_json)
    dataset = NGADCanonicalDataset(
        dataset_dirs=[
            {
                "name": "synthetic",
                "path": str(root),
                "mask_and_mapping_path": str(mask_and_mapping_path),
            }
        ],
        normalization_stats_path=str(stats_path),
        rgb_rate_hz=10,
        action_steps_per_rgb_frame=2,
        anchor_offset=0,
        frame_ranges=((-1, 1),),
    )
    sample = dataset[2]

    assert sample["video"].shape == (3, 6, 3, 256, 256)
    assert sample["state"].shape == (3, 2, 128)
    assert sample["action"].shape == (3, 2, 128)
    assert sample["frame_offsets"].tolist() == [-1, 0, 1]
    assert sample["source_frame_indices"].tolist() == [1, 2, 3]
    assert sample["action_step_offsets"].tolist() == [[-3, -2], [-1, 0], [1, 2]]
    assert sample["frame_valid"].all()
    assert sample["action_valid"].all()
    assert sample["camera_mask"].shape == (3, 6)
    assert sample["image_pixel_mask"].shape == (3, 6, 256, 256)
    assert sample["state_feature_mask"].shape == (128,)
    assert "anchor_state" not in sample
    assert "anchor_state_feature_mask" not in sample
    torch.testing.assert_close(
        sample["state"][..., 0],
        torch.tensor([[0.05, 0.10], [0.15, 0.20], [0.25, 0.30]]),
    )
    torch.testing.assert_close(
        sample["action"][..., 0],
        torch.tensor([[-0.15, -0.10], [-0.05, 0.0], [0.05, 0.10]]),
    )
    assert sample["data_info"]["action_rate_hz"] == 20
    assert sample["data_info"]["sample_mode"] == "canonical"
    assert "normalization_id" not in sample["data_info"]
    assert stats_reads == 1
    assert dataset.normalization_stats() == json.loads(stats_path.read_text())
    with pytest.raises(TypeError):
        dataset.denormalize_action(torch.zeros(128), "synthetic")

    def fail_normalization_transform(_stats):
        raise AssertionError("video-only mode must not construct a TCP transform")

    monkeypatch.setattr(
        NGADCanonicalDataset,
        "_normalization_transform",
        staticmethod(fail_normalization_transform),
    )
    video_dataset = NGADCanonicalDataset(
        dataset_dirs=[
            {
                "name": "synthetic",
                "path": str(root),
                "mask_and_mapping_path": str(mask_and_mapping_path),
            }
        ],
        normalization_stats_path=None,
        rgb_rate_hz=10,
        action_steps_per_rgb_frame=2,
        anchor_offset=0,
        frame_ranges=((-1, 1),),
    )
    video_sample = video_dataset[2]

    assert set(video_sample) == {
        "video",
        "frame_offsets",
        "source_frame_indices",
        "frame_timestamps",
        "frame_valid",
        "camera_mask",
        "image_pixel_mask",
        "prompt",
        "data_info",
    }
    assert video_sample["video"].shape == (3, 6, 3, 256, 256)
    assert video_sample["camera_mask"].shape == (3, 6)
    assert video_sample["image_pixel_mask"].shape == (3, 6, 256, 256)
    assert video_sample["data_info"]["sample_mode"] == "video_only"
    assert video_dataset.normalization_stats() is None
    assert stats_reads == 1
    with pytest.raises(RuntimeError, match="video-only mode"):
        video_dataset.denormalize_action(torch.zeros(128))
