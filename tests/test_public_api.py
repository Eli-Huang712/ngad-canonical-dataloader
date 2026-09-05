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
from ngad_canonical_dataloader.action import (
    element_mask_to_feature_mask,
    normalize_dual_arm_absolute_tcp,
    pack_dual_arm_tcp,
)
from ngad_canonical_dataloader.windows import (
    build_timeline_layout,
    timeline_sample_indices,
)
from ngad_canonical_dataloader.statistics import RunningMoments


def test_dataset_classes_are_importable() -> None:
    assert NGADCanonicalDataset.__name__ == "NGADCanonicalDataset"


def test_running_moments_matches_population_zscore_statistics() -> None:
    moments = RunningMoments((2, 9))
    first = torch.arange(36, dtype=torch.float64).reshape(2, 2, 9)
    second = torch.arange(36, 72, dtype=torch.float64).reshape(2, 2, 9)
    moments.update(first, torch.ones_like(first, dtype=torch.bool))
    moments.update(second, torch.ones_like(second, dtype=torch.bool))
    result = moments.result(1.0e-5)
    expected = torch.cat([first, second], dim=0)
    torch.testing.assert_close(result["mean"], expected.mean(dim=0))
    torch.testing.assert_close(result["raw_std"], expected.std(dim=0, correction=0))


def test_running_moments_emit_neutral_stats_for_masked_arm() -> None:
    moments = RunningMoments((2, 9))
    values = torch.arange(18, dtype=torch.float64).reshape(1, 2, 9)
    valid = torch.zeros_like(values, dtype=torch.bool)
    valid[:, 0] = True
    moments.update(values, valid)
    required = torch.zeros((2, 9), dtype=torch.bool)
    required[0] = True

    result = moments.result(1.0e-5, required)

    torch.testing.assert_close(result["mean"][1], torch.zeros(9, dtype=torch.float64))
    torch.testing.assert_close(result["std"][1], torch.ones(9, dtype=torch.float64))
    assert result["count"][1].sum() == 0
    assert not result["std_floor_applied"][1].any()


def test_hy_gripper_values_map_to_canonical_openness() -> None:
    tcp = torch.zeros(3, 20)
    tcp[:, 9] = torch.tensor([0.0, 45.0, 90.0])
    tcp[:, 19] = torch.tensor([90.0, 45.0, 0.0])
    normalized = normalize_dual_arm_absolute_tcp(
        tcp,
        torch.zeros(2, 9),
        torch.ones(2, 9),
        torch.tensor([0.0, 0.0]),
        torch.tensor([90.0, 90.0]),
    )

    torch.testing.assert_close(normalized[:, 9], torch.tensor([1.0, 0.5, 0.0]))
    torch.testing.assert_close(normalized[:, 19], torch.tensor([0.0, 0.5, 1.0]))


def test_normalization_stats_require_gripper_endpoints() -> None:
    with pytest.raises(ValueError, match="gripper_closed_value.*gripper_open_value"):
        NGADCanonicalDataset._normalization_transform(
            {
                "schema_version": "ngad_canonical_tcp_v2",
                "state_tcp_mean": [[0] * 9, [0] * 9],
                "state_tcp_std": [[1] * 9, [1] * 9],
                "action_tcp_mean": [[0] * 9, [0] * 9],
                "action_tcp_std": [[1] * 9, [1] * 9],
            }
        )


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


def test_h264_backend_splits_sparse_timeline_decode_groups() -> None:
    requested = [132, 0, 3, 6, 129, 132, 261, 264]

    groups = H264ImageBackend._decode_groups(requested)

    assert groups == ((0, 3, 6), (129, 132), (261, 264))


def test_lance_backend_accepts_sparse_source_indices_after_row_filtering(
    tmp_path,
    monkeypatch,
) -> None:
    """Compacted episode offsets, not retained source indices, address Lance rows."""

    class FakeLanceDataset:
        def take(self, offsets, columns):
            requested = offsets.to_pylist()
            assert requested == [100, 102]
            values = {
                "index": [offset + 2 for offset in requested],
                "episode_index": [7 for _ in requested],
                "frame_index": [offset - 100 for offset in requested],
                "task_index": [3 for _ in requested],
                "timestamp": [(offset - 100) / 30 for offset in requested],
            }
            return pa.table({column: values[column] for column in columns})

    backend = LanceTableBackend(tmp_path, tmp_path / "table_000.lance")
    monkeypatch.setattr(backend, "_dataset", lambda: FakeLanceDataset())
    rows = backend.read_rows(
        episode={
            "episode_index": 7,
            "length": 3,
            "dataset_from_index": 100,
            "dataset_to_index": 103,
        },
        relative_indices=torch.tensor([0, 2]),
        field_mask={
            "observation.state": False,
            CANONICAL_TACTILE_VALUES_KEY: False,
            CANONICAL_TACTILE_DT_KEY: False,
        },
        field_mapping={},
        camera_keys=(),
        camera_mask=torch.tensor([], dtype=torch.bool),
    )

    assert sorted(rows) == [0, 2]
    assert rows[0]["index"] == 102
    assert rows[2]["index"] == 104
    assert rows[2]["episode_index"] == 7
    assert rows[2]["frame_index"] == 2


def test_parquet_backend_can_address_non_contiguous_global_episode_blocks(
    tmp_path,
) -> None:
    """UMI shards are physically dense even when their global indices have gaps."""
    data_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "index": [0, 1, 4, 5],
                "episode_index": [0, 0, 2, 2],
                "frame_index": [0, 1, 0, 1],
                "task_index": [0, 0, 0, 0],
                "timestamp": [0.0, 1.0, 0.0, 1.0],
            }
        ),
        data_path,
    )
    backend = ParquetTableBackend(
        tmp_path,
        {"data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"},
        row_addressing="episode_indexed",
    )
    backend._data_file_starts[(0, 0)] = 0

    rows = backend.read_rows(
        episode={
            "episode_index": 2,
            "length": 2,
            "dataset_from_index": 4,
            "dataset_to_index": 6,
            "data_chunk_index": 0,
            "data_file_index": 0,
        },
        relative_indices=torch.tensor([0, 1]),
        field_mask={
            "observation.state": False,
            CANONICAL_TACTILE_VALUES_KEY: False,
            CANONICAL_TACTILE_DT_KEY: False,
        },
        field_mapping={},
        camera_keys=(),
        camera_mask=torch.tensor([], dtype=torch.bool),
    )

    assert sorted(rows) == [0, 1]
    assert rows[0]["index"] == 4
    assert rows[0]["episode_index"] == 2
    assert rows[1]["index"] == 5
    assert rows[1]["frame_index"] == 1


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


def test_flat_lerobot_root_is_returned_as_one_physical_table(tmp_path) -> None:
    root = tmp_path / "umi-canonical-v3"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "videos").mkdir()
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 3, "total_frames": 120}),
        encoding="utf-8",
    )

    records = _discover_published_tables(root)

    assert records == [
        {
            "table_index": 0,
            "table_name": "umi-canonical-v3",
            "table_root": root,
            "num_episodes": 3,
            "num_frames": 120,
        }
    ]


def test_mixed_flat_and_table_children_are_rejected_as_ambiguous(tmp_path) -> None:
    (tmp_path / "meta").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "videos").mkdir()
    (tmp_path / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 3, "total_frames": 120}),
        encoding="utf-8",
    )
    table_meta = tmp_path / "table_000" / "meta"
    table_meta.mkdir(parents=True)
    (table_meta / "info.json").write_text(
        json.dumps({"total_episodes": 1, "total_frames": 40}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ambiguous"):
        _discover_published_tables(tmp_path)


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
            "action": False,
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
    dataset.video_only = False
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
    dataset.video_only = False
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


def test_tactile_events_align_to_fixed_rgb_slots_without_compaction() -> None:
    dataset = NGADCanonicalDataset.__new__(NGADCanonicalDataset)
    dataset.rgb_rate_hz = 10.0
    dataset.tactile_steps_per_rgb_frame = 8
    dataset.tactile_rate_hz = 80.0

    candidate_rows = dataset._tactile_candidate_source_indices(
        torch.tensor([0, 1, 2]),
        source_fps=30.0,
        source_length=20,
    )
    assert candidate_rows.tolist() == [[0, 0, 0], [1, 2, 3], [4, 5, 6]]

    event_times = [0.00625 + 0.0125 * index for index in range(8)]
    packed_events = {
        1: event_times[:3],
        2: [event_times[2], event_times[3], event_times[4]],
        3: event_times[5:],
    }
    rows = {}
    for row_index, events in packed_events.items():
        row_timestamp = row_index / 30.0
        values = torch.stack(
            [
                torch.stack(
                    [torch.full((25, 6), event_times.index(event) + 1.0) for event in events]
                )
                for _ in range(4)
            ]
        )
        dt = torch.tensor(
            [[event - row_timestamp for event in events] for _ in range(4)],
            dtype=torch.float64,
        )
        rows[row_index] = {
            "timestamp": row_timestamp,
            CANONICAL_TACTILE_VALUES_KEY: values,
            CANONICAL_TACTILE_DT_KEY: dt,
        }
    rows[2][CANONICAL_TACTILE_DT_KEY][1, 1] = torch.nan
    rows[2][CANONICAL_TACTILE_VALUES_KEY][1, 1] = 0

    values, dt, valid = dataset._align_tactile_to_rgb_frames(
        rows,
        frame_indices=torch.tensor([1]),
        candidate_source_indices=torch.tensor([[1, 2, 3]]),
        frame_valid=torch.tensor([True]),
        timestamp_start=torch.tensor(0.0, dtype=torch.float64),
        source_fps=30.0,
        use_stored_source_timestamps=True,
        tactile_available=True,
    )

    assert values.shape == (1, 4, 8, 25, 6)
    assert dt.shape == (1, 4, 8)
    assert valid.shape == (1, 4, 8)
    assert valid[0, 0].all()
    torch.testing.assert_close(
        dt[0, 0],
        torch.tensor([event - 0.1 for event in event_times], dtype=torch.float32),
    )
    assert not valid[0, 1, 3]
    assert torch.count_nonzero(values[0, 1, 3]) == 0
    assert dt[0, 1, 3] == 0
    assert valid[0, 1].sum() == 7

    irregular_rows = {
        row_index: {
            **row,
            "timestamp": float(row["timestamp"]) + (1.0 / 30.0 if row_index >= 2 else 0.0),
        }
        for row_index, row in rows.items()
    }
    frame_values, frame_dt, frame_valid = dataset._align_tactile_to_rgb_frames(
        irregular_rows,
        frame_indices=torch.tensor([1]),
        candidate_source_indices=torch.tensor([[1, 2, 3]]),
        frame_valid=torch.tensor([True]),
        timestamp_start=torch.tensor(0.0, dtype=torch.float64),
        source_fps=30.0,
        use_stored_source_timestamps=False,
        tactile_available=True,
    )
    torch.testing.assert_close(frame_values, values)
    torch.testing.assert_close(frame_dt, dt)
    torch.testing.assert_close(frame_valid, valid)


def test_dataset_returns_one_frame_aligned_timeline(tmp_path, monkeypatch) -> None:
    root = tmp_path / "canonical"
    table_root = root / "table_000"
    (table_root / "meta").mkdir(parents=True)
    second_table_root = root / "table_001"
    (second_table_root / "meta").mkdir(parents=True)
    features = {
        "observation.state": {"shape": [20]},
        "action": {"shape": [20]},
        CANONICAL_TACTILE_VALUES_KEY: {"shape": [4, 3, 25, 6]},
        CANONICAL_TACTILE_DT_KEY: {"shape": [4, 3]},
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
                "schema_version": "ngad_canonical_tcp_v2",
                "state_tcp_mean": [[0] * 9, [0] * 9],
                "state_tcp_std": [[1] * 9, [1] * 9],
                "action_tcp_mean": [[0] * 9, [0] * 9],
                "action_tcp_std": [[1] * 9, [1] * 9],
                "gripper_open_value": [1, 1],
                "gripper_closed_value": [0, 0],
            }
        ),
        encoding="utf-8",
    )

    read_field_masks = []

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
            del episode, field_mapping, camera_keys, camera_mask
            read_field_masks.append(dict(field_mask))
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
                row = {
                    "timestamp": frame / 10,
                    "frame_index": frame,
                    "episode_index": 0,
                    "index": frame,
                    "task_index": 0,
                }
                if field_mask["observation.state"]:
                    row["observation.state"] = tcp10 + tcp10
                rows[frame] = row
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
    assert read_field_masks[-1]["action"] is False
    with pytest.raises(TypeError):
        dataset.denormalize_action(torch.zeros(128), "synthetic")

    def fail_normalization_transform(_stats):
        raise AssertionError("video-only mode must not construct a TCP transform")

    monkeypatch.setattr(
        NGADCanonicalDataset,
        "_normalization_transform",
        staticmethod(fail_normalization_transform),
    )
    video_field_mask = dict(field_mask)
    video_field_mask["observation.state"] = False
    video_field_mask[CANONICAL_TACTILE_VALUES_KEY] = True
    video_field_mask[CANONICAL_TACTILE_DT_KEY] = True
    video_mask_path = tmp_path / "video_mask_and_mapping.json"
    video_mask_path.write_text(
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
                "field_mask": video_field_mask,
                "element_mask": {
                    "observation.state": [False] * 20,
                    "action": [False] * 20,
                },
            }
        ),
        encoding="utf-8",
    )
    video_dataset = NGADCanonicalDataset(
        dataset_dirs=[
            {
                "name": "synthetic",
                "path": str(root),
                "mask_and_mapping_path": str(video_mask_path),
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
    assert read_field_masks[-1]["observation.state"] is False
    assert read_field_masks[-1][CANONICAL_TACTILE_VALUES_KEY] is False
    assert read_field_masks[-1][CANONICAL_TACTILE_DT_KEY] is False
    with pytest.raises(RuntimeError, match="video-only mode"):
        video_dataset.denormalize_action(torch.zeros(128))
