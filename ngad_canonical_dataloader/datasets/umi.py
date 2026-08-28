"""UMI LeRobot v3 stereo-video adapter for the SANA-WAM input pipeline."""

from __future__ import annotations

from bisect import bisect_right
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ngad_canonical_dataloader.datasets.canonical import (
    CANONICAL_CAMERA_KEYS,
    CANONICAL_IMAGE_SIZE,
    NGADCanonicalDataset,
)
from ngad_canonical_dataloader.datasets.umi_tcp import umi_state_to_tcp


UMI_CAMERA_KEYS = CANONICAL_CAMERA_KEYS
UMI_STATE_NAMES = (
    "left_wrist.px",
    "left_wrist.py",
    "left_wrist.pz",
    "left_wrist.qx",
    "left_wrist.qy",
    "left_wrist.qz",
    "left_wrist.qw",
    "right_wrist.px",
    "right_wrist.py",
    "right_wrist.pz",
    "right_wrist.qx",
    "right_wrist.qy",
    "right_wrist.qz",
    "right_wrist.qw",
    "left_gripper.pos",
    "right_gripper.pos",
)
UMI_POSE_VALID_KEYS = (
    "observation.pose_valid.left_wrist",
    "observation.pose_valid.right_wrist",
)
UMI_SOURCE_IMAGE_HEIGHT = 480
UMI_SOURCE_IMAGE_WIDTH = 640
UMI_OUTPUT_RESOLUTION = CANONICAL_IMAGE_SIZE
UMI_RESIZED_HEIGHT = 192
UMI_VERTICAL_PADDING = 32


class SanaWAMUMIDataset(NGADCanonicalDataset):
    """Adapt UMI shard collections to six-view canonical TCP128 samples."""

    expected_camera_keys = UMI_CAMERA_KEYS

    def __init__(
        self,
        dataset_dirs: list[dict[str, Any]],
        target_rgb_fps: float,
        target_action_fps: float,
        camera_keys: list[str] | tuple[str, ...] = UMI_CAMERA_KEYS,
        resolution: int = UMI_OUTPUT_RESOLUTION,
        max_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        if int(resolution) != UMI_OUTPUT_RESOLUTION:
            raise ValueError("SanaWAMUMIDataset requires the fixed 256x256 image ABI.")
        self._umi_pixel_masks = self._build_pixel_masks()
        super().__init__(
            dataset_dirs=dataset_dirs,
            target_rgb_fps=target_rgb_fps,
            target_action_fps=target_action_fps,
            camera_keys=camera_keys,
            resolution=resolution,
            max_samples=max_samples,
            **kwargs,
        )
        self._filter_pose_invalid_windows(max_samples=max_samples)

    @staticmethod
    def _expand_shard_roots(dataset_dirs: list[str]) -> list[str]:
        """Resolve each configured UMI collection into ordered LeRobot shard roots."""
        shard_roots: list[str] = []
        for configured in dataset_dirs:
            root = Path(os.path.expanduser(configured)).resolve()
            if (root / "meta" / "info.json").is_file():
                shard_roots.append(str(root))
                continue
            infos = sorted(root.glob("shard_*/meta/info.json"))
            if not infos:
                raise ValueError(f"{root} is neither a UMI shard nor a UMI shard collection.")
            shard_roots.extend(str(path.parent.parent) for path in infos)
        return shard_roots

    @staticmethod
    def _backend_masks(backend: str) -> tuple[torch.Tensor, torch.Tensor]:
        """UMI provides two arms and all six ordered stereo camera streams."""
        if backend != "lerobot_v3":
            raise ValueError(f"UMI only supports the LeRobot v3 backend, got {backend}.")
        return torch.ones(2, dtype=torch.bool), torch.ones(6, dtype=torch.bool)

    @staticmethod
    def _build_pixel_masks() -> torch.Tensor:
        """Mark the 192-row resized image content inside the 256-square letterbox."""
        masks = torch.zeros(
            (len(UMI_CAMERA_KEYS), UMI_OUTPUT_RESOLUTION, UMI_OUTPUT_RESOLUTION),
            dtype=torch.bool,
        )
        masks[:, UMI_VERTICAL_PADDING : UMI_VERTICAL_PADDING + UMI_RESIZED_HEIGHT] = True
        return masks

    def _load_pixel_masks(self, dataset_root: Path) -> torch.Tensor:
        """Return the shared analytic letterbox mask; UMI has no mask sidecar."""
        del dataset_root
        return self._umi_pixel_masks

    def _validate_features(self, root: Path, features: dict[str, Any], backend: str) -> None:
        """Validate the exact UMI state, pose-valid, and six-video source schema."""
        if backend != "lerobot_v3":
            raise ValueError(f"UMI root {root} must use the LeRobot v3 backend.")
        state = features.get("observation.state", {})
        if state.get("shape") != [16] or tuple(state.get("names") or ()) != UMI_STATE_NAMES:
            raise ValueError(f"{root} must expose the named UMI observation.state[16] layout.")
        if features.get("timestamp", {}).get("shape") != [1]:
            raise ValueError(f"{root} timestamp must have shape [1].")
        for key in UMI_POSE_VALID_KEYS:
            if features.get(key, {}).get("shape") != [1]:
                raise ValueError(f"{root} feature {key} must have shape [1].")
        for camera in self.camera_keys:
            feature = features.get(camera, {})
            if feature.get("dtype") != "video" or feature.get("shape") != [480, 640, 3]:
                raise ValueError(f"{root} camera {camera} must be video [480,640,3].")

    def _read_metadata(
        self,
        root: Path,
        backend: str,
    ) -> tuple[dict[int, str], dict[int, str], list[dict[str, Any]]]:
        """Read UMI episode offsets and each camera's independent MP4 time range."""
        if backend != "lerobot_v3":
            raise ValueError(f"UMI root {root} must use the LeRobot v3 backend.")
        try:
            import pyarrow.dataset as ds
        except ImportError as error:
            raise ImportError("UMI metadata loading requires pyarrow.") from error

        video_columns = [
            f"videos/{camera}/{field}"
            for camera in self.camera_keys
            for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
        ]
        rows = (
            ds.dataset(root / "meta" / "episodes", format="parquet")
            .to_table(
                columns=[
                    "episode_index",
                    "tasks",
                    "length",
                    "data/chunk_index",
                    "data/file_index",
                    "dataset_from_index",
                    "dataset_to_index",
                    *video_columns,
                ]
            )
            .to_pylist()
        )
        episodes: list[dict[str, Any]] = []
        episode_tasks: dict[int, str] = {}
        for row in rows:
            episode_index = int(row["episode_index"])
            task = row["tasks"]
            if not isinstance(task, str) or not task:
                raise ValueError(f"UMI episode {episode_index} under {root} must contain one task string.")
            episode = {
                "episode_index": episode_index,
                "length": int(row["length"]),
                "dataset_from_index": int(row["dataset_from_index"]),
                "dataset_to_index": int(row["dataset_to_index"]),
                "data_chunk_index": int(row["data/chunk_index"]),
                "data_file_index": int(row["data/file_index"]),
                "videos": {
                    camera: {
                        field: row[f"videos/{camera}/{field}"]
                        for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
                    }
                    for camera in self.camera_keys
                },
            }
            if episode["dataset_to_index"] - episode["dataset_from_index"] != episode["length"]:
                raise ValueError(f"Invalid UMI episode offsets under {root}: {episode}.")
            for camera, video in episode["videos"].items():
                video["chunk_index"] = int(video["chunk_index"])
                video["file_index"] = int(video["file_index"])
                video["from_timestamp"] = float(video["from_timestamp"])
                video["to_timestamp"] = float(video["to_timestamp"])
                if video["to_timestamp"] <= video["from_timestamp"]:
                    raise ValueError(
                        f"Invalid UMI video range for {camera} in episode {episode_index}."
                    )
            episodes.append(episode)
            episode_tasks[episode_index] = task
        return {}, episode_tasks, sorted(episodes, key=lambda item: item["dataset_from_index"])

    @staticmethod
    def _reshape_state_window(absolute_state: torch.Tensor) -> torch.Tensor:
        """Convert stored UMI state16 directly to canonical absolute [T,2,10]."""
        return umi_state_to_tcp(absolute_state)

    def _resize_video(self, video: torch.Tensor) -> torch.Tensor:
        """Resize 640x480 without distortion, then add black vertical letterbox bars."""
        if video.shape[-2:] != (UMI_SOURCE_IMAGE_HEIGHT, UMI_SOURCE_IMAGE_WIDTH):
            raise ValueError(
                "UMI decoded video must be 480x640, "
                f"got {tuple(video.shape[-2:])}."
            )
        resized = F.interpolate(
            video,
            size=(UMI_RESIZED_HEIGHT, UMI_OUTPUT_RESOLUTION),
            mode="bilinear",
            align_corners=False,
        )
        return F.pad(
            resized,
            (0, 0, UMI_VERTICAL_PADDING, UMI_VERTICAL_PADDING),
            value=-1.0,
        )

    def _read_pose_validity(self) -> dict[tuple[int, int], np.ndarray]:
        """Read both-arm pose validity once per Parquet file for window filtering."""
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("UMI pose-valid filtering requires pyarrow.") from error

        selected = {
            (episode["root_index"], episode["episode_index"]): episode
            for episode in self._episodes
        }
        validity: dict[tuple[int, int], np.ndarray] = {}
        for root_index, meta in enumerate(self._root_meta):
            root_episodes = [
                episode for key, episode in selected.items() if key[0] == root_index
            ]
            file_keys = {
                (episode["data_chunk_index"], episode["data_file_index"])
                for episode in root_episodes
            }
            for chunk_index, file_index in sorted(file_keys):
                path = meta["root"] / str(meta["info"]["data_path"]).format(
                    chunk_index=chunk_index,
                    file_index=file_index,
                )
                table = pq.read_table(
                    path,
                    columns=["episode_index", "frame_index", *UMI_POSE_VALID_KEYS],
                )
                rows = table.to_pydict()
                for row_index, episode_index_value in enumerate(rows["episode_index"]):
                    key = (root_index, int(episode_index_value))
                    if key not in selected:
                        continue
                    episode = selected[key]
                    values = validity.setdefault(
                        key, np.zeros(episode["length"], dtype=np.bool_)
                    )
                    frame_index = int(rows["frame_index"][row_index])
                    values[frame_index] = all(
                        float(rows[name][row_index]) >= 0.5 for name in UMI_POSE_VALID_KEYS
                    )
        return validity

    def _valid_target_start_ranges(
        self,
        episode: dict[str, Any],
        source_valid: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Return RGB anchors whose interpolated TCP supervision is pose-valid."""
        rgb_target_length = int(episode["rgb_target_length"])
        action_target_length = int(episode["action_target_length"])
        source_fps = float(self._root_meta[episode["root_index"]]["source_fps"])
        target_slots = np.arange(action_target_length, dtype=np.float64)
        source_positions = target_slots * source_fps / self.target_action_fps
        lower = np.floor(source_positions).astype(np.int64).clip(0, episode["length"] - 1)
        upper = np.ceil(source_positions).astype(np.int64).clip(0, episode["length"] - 1)
        target_valid = source_valid[lower] & source_valid[upper]
        invalid_prefix = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(~target_valid, dtype=np.int64)]
        )
        starts = np.arange(rgb_target_length, dtype=np.int64)
        rate_ratio = int(round(self.target_action_fps / self.target_rgb_fps))
        action_starts = starts * rate_ratio
        action_ends = np.minimum(
            action_starts + self.action_horizon,
            action_target_length - 1,
        )
        valid_starts = starts[
            (invalid_prefix[action_ends + 1] - invalid_prefix[action_starts]) == 0
        ]
        if valid_starts.size == 0:
            return []
        split_points = np.flatnonzero(np.diff(valid_starts) != 1) + 1
        return [
            (int(run[0]), int(run.size))
            for run in np.split(valid_starts, split_points)
            if run.size
        ]

    def _filter_pose_invalid_windows(self, *, max_samples: int | None) -> None:
        """Replace implicit starts with ranges whose two-arm TCP supervision is valid."""
        validity = self._read_pose_validity()
        filtered: list[dict[str, Any]] = []
        window_ends: list[int] = []
        total_windows = 0
        for episode in self._episodes:
            key = (episode["root_index"], episode["episode_index"])
            source_valid = validity.get(key)
            if source_valid is None:
                raise ValueError(f"Missing UMI pose validity for episode {key}.")
            for window_start, window_count in self._valid_target_start_ranges(
                episode, source_valid
            ):
                filtered.append(
                    {
                        **episode,
                        "window_start": window_start,
                        "window_count": window_count,
                    }
                )
                total_windows += window_count
                window_ends.append(total_windows)
        if not filtered:
            raise ValueError("UMI split has no windows with valid two-arm TCP supervision.")
        self._episodes = filtered
        self._episode_window_ends = window_ends
        self.ori_imgs_nums = total_windows
        self._length = min(total_windows, int(max_samples)) if max_samples is not None else total_windows
        self.ratio_nums = {next(iter(self.aspect_ratio)): self._length}

    def _locate_window(self, index: int) -> tuple[dict[str, Any], int]:
        """Map the global index into a pose-valid contiguous target-grid range."""
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        position = bisect_right(self._episode_window_ends, int(index))
        previous_end = 0 if position == 0 else self._episode_window_ends[position - 1]
        episode = self._episodes[position]
        return episode, episode["window_start"] + int(index) - previous_end
