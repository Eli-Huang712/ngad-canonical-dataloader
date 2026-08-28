"""Fail-closed LeRobot v3 Lance/JPEG reader for Hy-Embodied SANA-WAM training."""

from __future__ import annotations

from bisect import bisect_right
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ngad_canonical_dataloader.datasets.canonical import (
    CANONICAL_CAMERA_KEYS,
    CANONICAL_IMAGE_SIZE,
    CanonicalTCPTransform,
    NGAD_CANONICAL_SCHEMA,
)
from ngad_canonical_dataloader.datasets.hy_embodied_tcp import hy_state_to_tcp
from ngad_canonical_dataloader.tcp import WAM_FEATURE_DIM
from ngad_canonical_dataloader.windows import split_episode_indices, single_rate_wam_window_indices


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
HY_SOURCE_CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
HY_CAMERA_VIEW_MASK = torch.tensor([True, False, True, False, True, False])


def _lance_column(source_name: str) -> str:
    """Map public LeRobot dotted feature names to the physical Lance schema."""
    return source_name.replace(".", "_")


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


class SanaWAMHyEmbodiedDataset(Dataset):
    """Expose explicit Hy Lance tables as model-ready dual-arm TCP128 windows."""

    normalization_stats_filename = "hy_embodied_normalization_stats.json"

    def __init__(
        self,
        dataset_dirs: list[str],
        normalization_stats_path: str,
        camera_keys: list[str] | tuple[str, ...] = CANONICAL_CAMERA_KEYS,
        action_horizon: int = 32,
        history_chunks: int = 0,
        action_sample_stride: int = 3,
        video_sample_stride: int = 2,
        action_dim: int = WAM_FEATURE_DIM,
        proprio_dim: int = WAM_FEATURE_DIM,
        max_samples: int | None = None,
        validation_split: float = 0.0,
        validation_seed: int = 3407,
        split: str = "train",
        resolution: int = CANONICAL_IMAGE_SIZE,
        transform=None,
        **extra: Any,
    ) -> None:
        del transform
        if "concat_multi_camera" in extra:
            raise TypeError("SanaWAMHyEmbodiedDataset no longer accepts concat_multi_camera.")
        if not dataset_dirs:
            raise ValueError("SanaWAMHyEmbodiedDataset requires explicit Lance table directories.")
        if not normalization_stats_path:
            raise ValueError("SanaWAMHyEmbodiedDataset requires normalization_stats_path.")
        if int(history_chunks) != 0:
            raise ValueError("The frozen first-stage SANA-WAM contract requires history_chunks=0.")
        if int(action_dim) != WAM_FEATURE_DIM or int(proprio_dim) != WAM_FEATURE_DIM:
            raise ValueError("SANA-WAM Dataset outputs require action_dim=128 and proprio_dim=128.")
        if int(action_horizon) <= 0 or int(action_sample_stride) <= 0 or int(video_sample_stride) <= 0:
            raise ValueError("action_horizon, action_sample_stride, and video_sample_stride must be positive.")
        if int(action_horizon) % int(video_sample_stride):
            raise ValueError("action_horizon must be divisible by video_sample_stride.")
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'.")

        self.table_roots = [Path(os.path.expanduser(path)).resolve() for path in dataset_dirs]
        self.camera_keys = tuple(str(key) for key in camera_keys)
        if self.camera_keys != CANONICAL_CAMERA_KEYS:
            raise ValueError(f"Hy model-facing cameras are fixed to {CANONICAL_CAMERA_KEYS}.")
        if int(resolution) != CANONICAL_IMAGE_SIZE:
            raise ValueError("SanaWAMHyEmbodiedDataset requires the fixed 256x256 image ABI.")
        self.source_camera_keys = HY_SOURCE_CAMERA_KEYS
        self.action_horizon = int(action_horizon)
        self.action_sample_stride = int(action_sample_stride)
        self.video_sample_stride = int(video_sample_stride)
        self.resolution = int(resolution)
        self.load_vae_feat = False
        self.load_text_feat = False
        self.aspect_ratio = {"1.00": [self.resolution, self.resolution]}

        self._tables: list[dict[str, Any]] = []
        self._episodes: list[dict[str, int]] = []
        self._episode_window_ends: list[int] = []
        total_windows = 0
        for table_index, root in enumerate(self.table_roots):
            table_name = root.name
            info = _read_json_object(root / "meta" / "info.json")
            self._validate_features(root, info.get("features", {}))
            lance_path = root / f"{table_name}.lance"
            if not lance_path.is_dir():
                raise FileNotFoundError(f"Missing Lance dataset {lance_path}.")
            tasks = self._read_tasks(root)
            episodes = self._read_episodes(root)
            train_episodes, validation_episodes = split_episode_indices(
                [episode["episode_index"] for episode in episodes],
                validation_split=float(validation_split),
                seed=int(validation_seed) + table_index,
            )
            selected = train_episodes if split == "train" else validation_episodes
            self._tables.append(
                {
                    "name": table_name,
                    "root": root,
                    "lance_path": lance_path,
                    "tasks": tasks,
                    "info": info,
                }
            )
            for episode in episodes:
                if episode["episode_index"] not in selected:
                    continue
                length = episode["dataset_to_index"] - episode["dataset_from_index"]
                if length != episode["length"] or length <= 0:
                    raise ValueError(f"Invalid episode offsets in {root}: {episode}.")
                self._episodes.append({"table_index": table_index, **episode})
                total_windows += length
                self._episode_window_ends.append(total_windows)

        if not self._episodes:
            raise ValueError(f"Hy-Embodied split '{split}' has no episodes.")
        self.ori_imgs_nums = total_windows
        if max_samples is not None and int(max_samples) <= 0:
            raise ValueError("max_samples must be positive when provided.")
        self._length = min(total_windows, int(max_samples)) if max_samples is not None else total_windows
        self.ratio_nums = {next(iter(self.aspect_ratio)): self._length}
        self._normalization_stats = _read_json_object(
            Path(os.path.expanduser(normalization_stats_path)).resolve()
        )
        self._load_normalization_stats()
        self.tcp_transform = CanonicalTCPTransform(
            self.state_xyz_min,
            self.state_xyz_max,
            self.action_xyz_scale,
        )
        self._lance_handles: dict[int, Any] = {}
        self._lance_pid = os.getpid()

    def _validate_features(self, root: Path, features: dict[str, Any]) -> None:
        expected = {"observation.state": 16, "action": 2}
        for name, dimension in expected.items():
            shape = features.get(name, {}).get("shape")
            if shape != [dimension]:
                raise ValueError(f"{root} feature {name} must have shape [{dimension}], got {shape}.")
        for camera in self.source_camera_keys:
            feature = features.get(camera, {})
            if feature.get("dtype") != "image" or feature.get("shape") != [240, 424, 3]:
                raise ValueError(f"{root} camera {camera} must be JPEG image [240,424,3].")

    @staticmethod
    def _read_tasks(root: Path) -> dict[int, str]:
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("SanaWAMHyEmbodiedDataset requires pyarrow.") from error
        rows = pq.read_table(root / "meta" / "tasks.parquet").to_pylist()
        tasks = {int(row["task_index"]): str(row["task"]) for row in rows}
        if not tasks:
            raise ValueError(f"No task metadata found in {root}.")
        return tasks

    @staticmethod
    def _read_episodes(root: Path) -> list[dict[str, int]]:
        try:
            import pyarrow.dataset as ds
        except ImportError as error:
            raise ImportError("SanaWAMHyEmbodiedDataset requires pyarrow.") from error
        table = ds.dataset(root / "meta" / "episodes", format="parquet").to_table(
            columns=["episode_index", "length", "dataset_from_index", "dataset_to_index"]
        )
        episodes = [
            {name: int(row[name]) for name in ("episode_index", "length", "dataset_from_index", "dataset_to_index")}
            for row in table.to_pylist()
        ]
        return sorted(episodes, key=lambda row: row["dataset_from_index"])

    def _load_normalization_stats(self) -> None:
        stats = self._normalization_stats
        if stats.get("schema_version") != NGAD_CANONICAL_SCHEMA:
            raise ValueError(
                f"Expected normalization schema {NGAD_CANONICAL_SCHEMA}, got {stats.get('schema_version')}."
            )
        self.state_xyz_min = torch.tensor(stats["state_xyz_min"], dtype=torch.float32)
        self.state_xyz_max = torch.tensor(stats["state_xyz_max"], dtype=torch.float32)
        self.action_xyz_scale = torch.tensor(stats["action_xyz_scale"], dtype=torch.float32)
        if (
            self.state_xyz_min.shape != (2, 3)
            or self.state_xyz_max.shape != (2, 3)
            or self.action_xyz_scale.shape != (2, 3)
        ):
            raise ValueError("Hy normalization xyz statistics must each have shape [2,3].")
        if not torch.isfinite(self.state_xyz_min).all() or not torch.isfinite(self.state_xyz_max).all():
            raise ValueError("Hy state xyz normalization statistics contain non-finite values.")
        if torch.any(self.state_xyz_max <= self.state_xyz_min):
            raise ValueError("Hy state xyz maximum must be greater than minimum for every arm and axis.")
        if not torch.isfinite(self.action_xyz_scale).all() or torch.any(self.action_xyz_scale <= 0.0):
            raise ValueError("Hy action xyz scales must be finite and positive.")

    def __len__(self) -> int:
        return self._length

    def normalization_stats(self) -> dict[str, Any]:
        """Return the exact source schema, physical convention, and scales used by this Dataset."""
        stats = dict(self._normalization_stats)
        stats.update(
            {
                "source_state_layout": "left_xyz_quat_xyzw_gripper_mm_right_xyz_quat_xyzw_gripper_mm",
                "source_action_semantics": "unused_by_sana_wam",
                "action_supervision": "canonical_absolute_state_relative_to_chunk_first",
                "rotation_6d": "rotation_matrix_first_two_rows",
                "arm_mask": [True, True],
                "camera_view_mask": HY_CAMERA_VIEW_MASK.tolist(),
            }
        )
        return stats

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Restore model output through the shared canonical TCP transform."""
        return self.tcp_transform.denormalize_action(action)

    def _locate_window(self, index: int) -> tuple[dict[str, int], int]:
        if not 0 <= index < self._length:
            raise IndexError(index)
        episode_position = bisect_right(self._episode_window_ends, int(index))
        previous_end = 0 if episode_position == 0 else self._episode_window_ends[episode_position - 1]
        return self._episodes[episode_position], int(index) - previous_end

    def _lance_dataset(self, table_index: int):
        pid = os.getpid()
        if pid != self._lance_pid:
            self._lance_handles = {}
            self._lance_pid = pid
        if table_index not in self._lance_handles:
            try:
                import lance
            except ImportError as error:
                raise ImportError("SanaWAMHyEmbodiedDataset requires the pylance package.") from error
            self._lance_handles[table_index] = lance.dataset(str(self._tables[table_index]["lance_path"]))
        return self._lance_handles[table_index]

    def _take_rows(
        self,
        episode: dict[str, int],
        relative_indices: torch.Tensor,
    ) -> dict[int, dict[str, Any]]:
        try:
            import pyarrow as pa
        except ImportError as error:
            raise ImportError("SanaWAMHyEmbodiedDataset requires pyarrow.") from error
        offsets = sorted(
            {
                episode["dataset_from_index"] + int(relative_index)
                for relative_index in relative_indices.tolist()
            }
        )
        columns = [
            "index",
            "episode_index",
            "frame_index",
            "task_index",
            _lance_column("observation.state"),
            *[_lance_column(camera) for camera in self.source_camera_keys],
        ]
        table = self._lance_dataset(episode["table_index"]).take(
            pa.array(offsets, type=pa.int64()),
            columns=columns,
        )
        rows = table.to_pylist()
        if len(rows) != len(offsets):
            raise RuntimeError(f"Lance take returned {len(rows)} rows for {len(offsets)} requested offsets.")
        by_relative_index: dict[int, dict[str, Any]] = {}
        for offset, row in zip(offsets, rows):
            relative_index = offset - episode["dataset_from_index"]
            if (
                int(row["index"]) != offset
                or int(row["episode_index"]) != episode["episode_index"]
                or int(row["frame_index"]) != relative_index
            ):
                raise RuntimeError(
                    "Lance row identity mismatch: "
                    f"requested_offset={offset}, returned="
                    f"({row['index']}, {row['episode_index']}, {row['frame_index']})."
                )
            by_relative_index[relative_index] = row
        return by_relative_index

    def _decode_camera_rows(
        self,
        rows: dict[int, dict[str, Any]],
        camera: str,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        frames = []
        column = _lance_column(camera)
        for index in indices.tolist():
            payload = rows[int(index)].get(column)
            if not isinstance(payload, bytes) or len(payload) < 4:
                raise ValueError(f"Invalid JPEG payload for {camera} at episode frame {index}.")
            with Image.open(BytesIO(payload)) as image:
                image.load()
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if rgb.shape != (240, 424, 3):
                raise ValueError(f"Decoded {camera} frame {index} has shape {rgb.shape}, expected (240,424,3).")
            frames.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1))
        video = torch.stack(frames).float() / 127.5 - 1.0
        scale = max(self.resolution / video.shape[-2], self.resolution / video.shape[-1])
        resized_height = max(self.resolution, int(round(video.shape[-2] * scale)))
        resized_width = max(self.resolution, int(round(video.shape[-1] * scale)))
        video = F.interpolate(
            video,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        top = (resized_height - self.resolution) // 2
        left = (resized_width - self.resolution) // 2
        return video[..., top : top + self.resolution, left : left + self.resolution]

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, start = self._locate_window(int(index))
        observation_indices, action_indices, image_is_pad, action_is_pad = single_rate_wam_window_indices(
            start,
            episode_length=episode["length"],
            action_horizon=self.action_horizon,
            action_sample_stride=self.action_sample_stride,
            video_sample_stride=self.video_sample_stride,
        )
        all_indices = torch.unique(torch.cat([observation_indices, action_indices]), sorted=True)
        rows = self._take_rows(episode, all_indices)
        source_state = torch.tensor(
            [rows[int(frame_index)][_lance_column("observation.state")] for frame_index in action_indices.tolist()],
            dtype=torch.float32,
        )
        arm_mask = torch.tensor([True, True])
        absolute_state_chunk = hy_state_to_tcp(source_state)
        action, action_feature_mask = self.tcp_transform.encode_action_targets(
            absolute_state_chunk[0], absolute_state_chunk, arm_mask
        )
        proprio, proprio_feature_mask = self.tcp_transform.encode_proprio(
            absolute_state_chunk[0], arm_mask
        )

        cameras = [
            self._decode_camera_rows(rows, camera, observation_indices)
            for camera in self.source_camera_keys
        ]
        black = torch.full_like(cameras[0], -1.0)
        # Each monocular Hy stream occupies the left-eye slot of its stereo pair.
        canonical_cameras = [cameras[0], black, cameras[1], black, cameras[2], black]
        video = torch.stack(canonical_cameras, dim=0).permute(0, 2, 1, 3, 4).contiguous()
        camera_view_mask = HY_CAMERA_VIEW_MASK.clone()
        image_pixel_mask = camera_view_mask[:, None, None, None].expand(
            -1, video.shape[2], video.shape[3], video.shape[4]
        )
        task_index = int(rows[start]["task_index"])
        tasks = self._tables[episode["table_index"]]["tasks"]
        if task_index not in tasks:
            raise KeyError(f"Missing task_index={task_index} in table task metadata.")
        prompt = DEFAULT_PROMPT.format(task=tasks[task_index])
        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "action_feature_mask": action_feature_mask,
            "proprio_feature_mask": proprio_feature_mask,
            "camera_view_mask": camera_view_mask,
            "image_pixel_mask": image_pixel_mask,
            "prompt": prompt,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "data_info": {
                "img_hw": torch.tensor([video.shape[-2], video.shape[-1]], dtype=torch.float32),
                "aspect_ratio": torch.tensor(video.shape[-1] / video.shape[-2], dtype=torch.float32),
                "table_index": episode["table_index"],
                "episode_index": episode["episode_index"],
                "start": start,
                "action_sample_stride": self.action_sample_stride,
            },
        }
