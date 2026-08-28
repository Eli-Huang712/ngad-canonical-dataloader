"""为 SANA-WAM 提供 FastWAM 兼容的 LIBERO 窗口及 episode 级训练/验证切分。"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ngad_canonical_dataloader.datasets.canonical import (
    CANONICAL_CAMERA_KEYS,
    CANONICAL_IMAGE_SIZE,
    CanonicalTCPTransform,
    NGAD_CANONICAL_SCHEMA,
)
from ngad_canonical_dataloader.datasets.libero_tcp import libero_state_to_tcp
from ngad_canonical_dataloader.tcp import WAM_FEATURE_DIM, tcp_target_relative_to_anchor
from ngad_canonical_dataloader.windows import (
    split_episode_indices,
    single_rate_wam_window_indices,
)

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
LIBERO_SOURCE_CAMERA_KEYS = ("image", "wrist_image")
LIBERO_CAMERA_VIEW_MASK = torch.tensor([True, False, True, False, False, False])


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"Expected JSON objects in {path}.")
                records.append(value)
    return records


class SanaWAMLiberoDataset(Dataset):
    """Expose LIBERO episodes as model-ready TCP128 windows.

    LIBERO state8 is converted to physical TCP10 at the Dataset boundary.
    Action labels are future TCP states expressed relative to each window's
    first pose, so the DataLoader and model never depend on source fields.
    """

    normalization_stats_filename = "libero_normalization_stats.json"

    def __init__(
        self,
        dataset_dirs: list[str],
        camera_keys: list[str] | tuple[str, ...] = CANONICAL_CAMERA_KEYS,
        action_horizon: int = 32,
        history_chunks: int = 0,
        action_sample_stride: int = 1,
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
            raise TypeError("SanaWAMLiberoDataset no longer accepts concat_multi_camera.")
        if not dataset_dirs:
            raise ValueError("SanaWAMLiberoDataset requires at least one dataset_dirs entry.")
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

        self.roots = [Path(os.path.expanduser(path)).resolve() for path in dataset_dirs]
        self.camera_keys = tuple(str(key) for key in camera_keys)
        if self.camera_keys != CANONICAL_CAMERA_KEYS:
            raise ValueError(f"LIBERO model-facing cameras are fixed to {CANONICAL_CAMERA_KEYS}.")
        if int(resolution) != CANONICAL_IMAGE_SIZE:
            raise ValueError("SanaWAMLiberoDataset requires the fixed 256x256 image ABI.")
        self.source_camera_keys = LIBERO_SOURCE_CAMERA_KEYS
        self.action_horizon = int(action_horizon)
        self.action_sample_stride = int(action_sample_stride)
        self.video_sample_stride = int(video_sample_stride)
        self.resolution = int(resolution)
        self.load_vae_feat = False
        self.load_text_feat = False
        self.aspect_ratio = {"1.00": [self.resolution, self.resolution]}

        self._root_meta: list[dict[str, Any]] = []
        self._windows: list[tuple[int, int, int]] = []
        for root_index, root in enumerate(self.roots):
            info = _read_json(root / "meta" / "info.json")
            features = info.get("features", {})
            self._validate_features(root, features)
            tasks = {
                int(record["task_index"]): str(record["task"])
                for record in _read_jsonl(root / "meta" / "tasks.jsonl")
            }
            episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
            episode_tasks = {
                int(record["episode_index"]): str(record.get("tasks", [""])[0])
                for record in episodes
            }
            episode_lengths = {
                int(record["episode_index"]): int(record["length"])
                for record in episodes
            }
            train_episodes, validation_episodes = split_episode_indices(
                list(episode_lengths),
                validation_split=validation_split,
                seed=int(validation_seed) + root_index,
            )
            selected_episodes = train_episodes if split == "train" else validation_episodes
            self._root_meta.append(
                {
                    "info": info,
                    "tasks": tasks,
                    "episode_tasks": episode_tasks,
                    "episode_lengths": episode_lengths,
                    "normalization_episodes": train_episodes,
                }
            )
            for episode_index, length in sorted(episode_lengths.items()):
                if episode_index not in selected_episodes:
                    continue
                self._windows.extend((root_index, episode_index, start) for start in range(length))
        self.ori_imgs_nums = len(self._windows)
        if max_samples is not None:
            if int(max_samples) <= 0:
                raise ValueError("max_samples must be positive when provided.")
            self._windows = self._windows[: int(max_samples)]
        self.ratio_nums = {next(iter(self.aspect_ratio)): len(self._windows)}
        (
            self.state_xyz_min,
            self.state_xyz_max,
            self.action_xyz_min,
            self.action_xyz_max,
        ) = self._compute_canonical_xyz_stats()
        self.action_xyz_scale = torch.maximum(self.action_xyz_min.abs(), self.action_xyz_max.abs()).clamp_min(1.0e-6)
        self.tcp_transform = CanonicalTCPTransform(
            self.state_xyz_min,
            self.state_xyz_max,
            self.action_xyz_scale,
        )

    def _validate_features(self, root: Path, features: dict[str, Any]) -> None:
        expected = {"action": 7, "observation.state": 8}
        for name, dimension in expected.items():
            shape = features.get(name, {}).get("shape")
            if shape != [dimension]:
                raise ValueError(f"{root} feature {name} must have shape [{dimension}], got {shape}.")
        for camera in self.source_camera_keys:
            name = f"observation.images.{camera}"
            if features.get(name, {}).get("dtype") != "video":
                raise ValueError(f"{root} is missing video feature {name}.")

    def __len__(self) -> int:
        return len(self._windows)

    def normalization_stats(self) -> dict[str, Any]:
        """Serialize the TCP schema and exact scales consumed during rollout."""
        return {
            "schema_version": NGAD_CANONICAL_SCHEMA,
            "source_state_layout": "eef_xyz_axis_angle_gripper_qpos2",
            "source_action_semantics": "unused_osc_pose_delta_gripper_command",
            "action_supervision": "canonical_absolute_state_relative_to_chunk_first",
            "tcp_frame": "libero_grip_site",
            "eef_to_tcp": "identity",
            "rotation_6d": "rotation_matrix_first_two_rows",
            "state_xyz_min": self.state_xyz_min.tolist(),
            "state_xyz_max": self.state_xyz_max.tolist(),
            "action_xyz_min": self.action_xyz_min.tolist(),
            "action_xyz_max": self.action_xyz_max.tolist(),
            "action_xyz_scale": self.action_xyz_scale.tolist(),
            "gripper_range": [0.0, 1.0],
            "arm_mask": [True, False],
            "camera_view_mask": LIBERO_CAMERA_VIEW_MASK.tolist(),
        }

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Restore model output through the shared canonical TCP transform."""
        return self.tcp_transform.denormalize_action(action)

    def _compute_canonical_xyz_stats(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scan training TCP states and every valid fixed-anchor horizon pair."""
        state_min = torch.full((3,), torch.inf, dtype=torch.float32)
        state_max = torch.full((3,), -torch.inf, dtype=torch.float32)
        action_min = torch.full((3,), torch.inf, dtype=torch.float32)
        action_max = torch.full((3,), -torch.inf, dtype=torch.float32)

        with torch.no_grad():
            for root_index, meta in enumerate(self._root_meta):
                for episode_index in sorted(meta["normalization_episodes"]):
                    episode_length = meta["episode_lengths"][episode_index]
                    tcp = self._episode_table(root_index, episode_index)["state"][:episode_length, 0]
                    state_min = torch.minimum(state_min, tcp[:, :3].amin(dim=0))
                    state_max = torch.maximum(state_max, tcp[:, :3].amax(dim=0))
                    offsets = range(
                        0,
                        min(self.action_horizon * self.action_sample_stride, episode_length),
                        self.action_sample_stride,
                    )
                    for offset in offsets:
                        count = episode_length - offset
                        relative = tcp_target_relative_to_anchor(tcp[:count], tcp[offset : offset + count])
                        action_min = torch.minimum(action_min, relative[:, :3].amin(dim=0))
                        action_max = torch.maximum(action_max, relative[:, :3].amax(dim=0))

        # Invalid right-arm statistics use an identity scale and remain masked.
        return (
            torch.stack([state_min, torch.zeros_like(state_min)]),
            torch.stack([state_max, torch.ones_like(state_max)]),
            torch.stack([action_min, -torch.ones_like(action_min)]),
            torch.stack([action_max, torch.ones_like(action_max)]),
        )

    @lru_cache(maxsize=32)
    def _episode_table(self, root_index: int, episode_index: int) -> dict[str, torch.Tensor]:
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("SanaWAMLiberoDataset requires pyarrow for LeRobot parquet data.") from error

        meta = self._root_meta[root_index]
        info = meta["info"]
        chunks_size = int(info["chunks_size"])
        parquet_path = self.roots[root_index] / str(info["data_path"]).format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
        table = pq.read_table(parquet_path, columns=["observation.state", "task_index"])
        state = torch.tensor(table["observation.state"].combine_chunks().to_pylist(), dtype=torch.float32)
        task_index = int(table["task_index"][0].as_py())
        tcp = libero_state_to_tcp(state)
        canonical_state = torch.zeros((tcp.shape[0], 2, 10), dtype=tcp.dtype)
        canonical_state[:, 0] = tcp
        canonical_state[:, 1, 3:9] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=tcp.dtype
        )
        return {"state": canonical_state, "task_index": task_index}

    def _decode_camera_frames(
        self,
        root_index: int,
        episode_index: int,
        camera: str,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        try:
            import av
        except ImportError as error:
            raise ImportError("SanaWAMLiberoDataset requires PyAV for LeRobot videos.") from error

        meta = self._root_meta[root_index]
        info = meta["info"]
        chunks_size = int(info["chunks_size"])
        video_path = self.roots[root_index] / str(info["video_path"]).format(
            episode_chunk=episode_index // chunks_size,
            video_key=f"observation.images.{camera}",
            episode_index=episode_index,
        )
        requested_indices = [int(index) for index in indices.tolist()]
        first_index = min(requested_indices)
        last_index = max(requested_indices)
        requested_unique = set(requested_indices)
        decoded: dict[int, torch.Tensor] = {}

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            frame_rate = float(stream.average_rate or info["fps"])
            time_base = float(stream.time_base)
            start_pts = int(stream.start_time or 0)
            seek_pts = start_pts + int((first_index / frame_rate) / time_base)
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                frame_index = int(round((int(frame.pts) - start_pts) * time_base * frame_rate))
                if frame_index < first_index:
                    continue
                if frame_index > last_index:
                    break
                if frame_index in requested_unique and frame_index not in decoded:
                    decoded[frame_index] = torch.from_numpy(frame.to_ndarray(format="rgb24"))
                    if len(decoded) == len(requested_unique):
                        break

        missing = sorted(requested_unique.difference(decoded))
        if missing:
            raise RuntimeError(f"Failed to decode frame indices {missing} from {video_path}.")
        frames = torch.stack([decoded[index] for index in requested_indices]).permute(0, 3, 1, 2).float()
        frames = frames / 127.5 - 1.0
        return F.interpolate(frames, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)

    def __getitem__(self, index: int) -> dict[str, Any]:
        root_index, episode_index, start = self._windows[index]
        episode_length = self._root_meta[root_index]["episode_lengths"][episode_index]
        observation_indices, action_indices, image_is_pad, action_is_pad = single_rate_wam_window_indices(
            start,
            episode_length=episode_length,
            action_horizon=self.action_horizon,
            action_sample_stride=self.action_sample_stride,
            video_sample_stride=self.video_sample_stride,
        )
        table = self._episode_table(root_index, episode_index)
        arm_mask = torch.tensor([True, False])
        state_chunk = table["state"].index_select(0, action_indices)
        action, action_feature_mask = self.tcp_transform.encode_action_targets(
            state_chunk[0], state_chunk, arm_mask
        )
        proprio, proprio_feature_mask = self.tcp_transform.encode_proprio(table["state"][start], arm_mask)

        cameras = [
            self._decode_camera_frames(root_index, episode_index, camera, observation_indices)
            for camera in self.source_camera_keys
        ]
        black = torch.full_like(cameras[0], -1.0)
        # Map the two monocular LIBERO streams into the left-eye canonical slots.
        canonical_cameras = [cameras[0], black, cameras[1], black, black, black]
        video = torch.stack(canonical_cameras, dim=0).permute(0, 2, 1, 3, 4).contiguous()
        camera_view_mask = LIBERO_CAMERA_VIEW_MASK.clone()
        image_pixel_mask = camera_view_mask[:, None, None, None].expand(
            -1, video.shape[2], video.shape[3], video.shape[4]
        )
        task = self._root_meta[root_index]["episode_tasks"].get(episode_index)
        if not task:
            task = self._root_meta[root_index]["tasks"][table["task_index"]]
        prompt = DEFAULT_PROMPT.format(task=task)
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
                "root_index": root_index,
                "episode_index": episode_index,
                "start": start,
            },
        }
