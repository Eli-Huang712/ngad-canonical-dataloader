"""Canonical NGAD TCP dataset and model-facing TCP128 transformation."""

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

from ngad_canonical_dataloader import rotation as rotation_utils
from ngad_canonical_dataloader.tcp import (
    DUAL_ARM_TCP_FEATURE_DIM,
    WAM_FEATURE_DIM,
    arm_mask_to_feature_mask,
    denormalize_dual_arm_relative_tcp,
    dual_arm_tcp_target_relative_to_anchor,
    normalize_dual_arm_absolute_tcp,
    normalize_dual_arm_relative_tcp,
    pack_dual_arm_tcp,
)
from ngad_canonical_dataloader.windows import (
    split_episode_indices,
    wam_window_indices,
)
from ngad_canonical_dataloader.memory import wam_memory_indices


NGAD_CANONICAL_SCHEMA = "ngad_canonical_tcp_v1"
HY_CANONICAL_SCHEMA = "ngad_hy_canonical_lance_v2"
CANONICAL_CAMERA_KEYS = (
    "observation.images.cam_head_left",
    "observation.images.cam_head_right",
    "observation.images.cam_left_wrist_left",
    "observation.images.cam_left_wrist_right",
    "observation.images.cam_right_wrist_left",
    "observation.images.cam_right_wrist_right",
)
CANONICAL_TACTILE_VALUES_KEY = "observation.tactile.values"
CANONICAL_TACTILE_DT_KEY = "observation.tactile.dt"
CANONICAL_IMAGE_SIZE = 256
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
PIXEL_MASKS_FILENAME = "image_pixel_masks.npz"


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _lance_column(key: str) -> str:
    """Map canonical dotted feature names to Hy-compatible Lance columns."""
    return key.replace(".", "_")


class CanonicalTCPTransform:
    """Convert absolute per-arm canonical state windows into the TCP128 ABI."""

    def __init__(
        self,
        state_xyz_min: torch.Tensor,
        state_xyz_max: torch.Tensor,
        action_xyz_scale: torch.Tensor,
    ) -> None:
        self.state_xyz_min = torch.as_tensor(state_xyz_min, dtype=torch.float32)
        self.state_xyz_max = torch.as_tensor(state_xyz_max, dtype=torch.float32)
        self.action_xyz_scale = torch.as_tensor(action_xyz_scale, dtype=torch.float32)

    @staticmethod
    def _flatten_state(absolute_state: torch.Tensor) -> torch.Tensor:
        """Flatten the canonical [left,right] arm axis without interleaving features."""
        if absolute_state.shape[-2:] != (2, 10):
            raise ValueError(
                "Canonical observation.state must end with [2,10], "
                f"got {tuple(absolute_state.shape)}."
            )
        return absolute_state.reshape(*absolute_state.shape[:-2], DUAL_ARM_TCP_FEATURE_DIM)

    def encode_proprio(
        self,
        absolute_state: torch.Tensor,
        arm_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize one absolute state and return its value and validity in TCP128."""
        feature_mask = arm_mask_to_feature_mask(arm_mask)
        normalized = normalize_dual_arm_absolute_tcp(
            self._flatten_state(absolute_state),
            self.state_xyz_min,
            self.state_xyz_max,
        )
        proprio = pack_dual_arm_tcp(normalized)
        return proprio * feature_mask.to(proprio.dtype), feature_mask

    def encode_action_targets(
        self,
        anchor_state: torch.Tensor,
        absolute_state_targets: torch.Tensor,
        arm_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode future states relative to one explicit current-frame anchor."""
        feature_mask = arm_mask_to_feature_mask(arm_mask)
        anchor = self._flatten_state(anchor_state)
        targets = self._flatten_state(absolute_state_targets)
        relative = dual_arm_tcp_target_relative_to_anchor(anchor.unsqueeze(0), targets)
        normalized = normalize_dual_arm_relative_tcp(relative, self.action_xyz_scale)
        action = pack_dual_arm_tcp(normalized)
        return action * feature_mask.to(action.dtype), feature_mask

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Restore the active TCP20 action block to physical relative coordinates."""
        denormalized = torch.zeros_like(action)
        denormalized[..., :DUAL_ARM_TCP_FEATURE_DIM] = denormalize_dual_arm_relative_tcp(
            action[..., :DUAL_ARM_TCP_FEATURE_DIM],
            self.action_xyz_scale.to(action.device, action.dtype),
        )
        return denormalized


def interpolate_canonical_tcp(
    lower: torch.Tensor,
    upper: torch.Tensor,
    fraction: torch.Tensor,
) -> torch.Tensor:
    """Interpolate absolute canonical TCP with XYZ/openness lerp and SO(3) SLERP."""
    if lower.shape != upper.shape or lower.shape[-2:] != (2, 10):
        raise ValueError("Canonical TCP interpolation requires matching [...,2,10] states.")
    fraction = torch.as_tensor(fraction, dtype=lower.dtype, device=lower.device)
    while fraction.ndim < lower.ndim - 1:
        fraction = fraction.unsqueeze(-1)
    weight = fraction.unsqueeze(-1)
    position = torch.lerp(lower[..., :3], upper[..., :3], weight)
    openness = torch.lerp(lower[..., 9:10], upper[..., 9:10], weight)
    lower_rotation = rotation_utils.rotation_6d_rows_to_matrix(lower[..., 3:9])
    upper_rotation = rotation_utils.rotation_6d_rows_to_matrix(upper[..., 3:9])
    quaternion = rotation_utils.quaternion_slerp_xyzw(
        rotation_utils.matrix_to_quaternion_xyzw(lower_rotation),
        rotation_utils.matrix_to_quaternion_xyzw(upper_rotation),
        fraction,
    )
    rotation = rotation_utils.matrix_to_rotation_6d_rows(
        rotation_utils.quaternion_xyzw_to_matrix(quaternion)
    )
    return torch.cat([position, rotation, openness], dim=-1)


class NGADCanonicalDataset(Dataset):
    """Read canonical TCP fields from Lance/JPEG or standard LeRobot video roots."""

    normalization_stats_filename = "ngad_canonical_normalization_stats.json"
    expected_camera_keys = CANONICAL_CAMERA_KEYS

    def __init__(
        self,
        dataset_dirs: list[dict[str, Any]],
        target_rgb_fps: float,
        target_action_fps: float,
        camera_keys: list[str] | tuple[str, ...] = CANONICAL_CAMERA_KEYS,
        num_frames: int = 17,
        action_horizon: int = 32,
        recent_memory_frames: int = 24,
        long_memory_anchor_interval_frames: int = 50,
        long_memory_window_frames: int = 8,
        long_memory_slots: int = 5,
        action_history_horizon: int = 10,
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
        legacy_fields = {
            "action_sample_stride",
            "concat_multi_camera",
            "history_chunks",
            "normalization_stats_path",
            "target_fps",
            "video_sample_stride",
        }.intersection(extra)
        if legacy_fields:
            raise TypeError(
                "NGADCanonicalDataset no longer accepts legacy sampling fields: "
                f"{sorted(legacy_fields)}."
            )
        if tuple(camera_keys) != self.expected_camera_keys:
            raise ValueError(
                f"{type(self).__name__} camera order is fixed to {self.expected_camera_keys}."
            )
        if int(resolution) != CANONICAL_IMAGE_SIZE:
            raise ValueError(
                f"{type(self).__name__} requires the fixed {CANONICAL_IMAGE_SIZE}x"
                f"{CANONICAL_IMAGE_SIZE} image ABI."
            )
        if not dataset_dirs:
            raise ValueError("NGADCanonicalDataset requires at least one named dataset root.")
        if (
            target_rgb_fps is None
            or target_action_fps is None
            or not np.isfinite(float(target_rgb_fps))
            or not np.isfinite(float(target_action_fps))
            or float(target_rgb_fps) <= 0
            or float(target_action_fps) <= 0
        ):
            raise ValueError("NGADCanonicalDataset requires positive RGB and action target rates.")
        rate_ratio = float(target_action_fps) / float(target_rgb_fps)
        if abs(rate_ratio - round(rate_ratio)) > 1.0e-9 or rate_ratio < 1.0:
            raise ValueError("target_action_fps / target_rgb_fps must be a positive integer.")
        if int(action_dim) != WAM_FEATURE_DIM or int(proprio_dim) != WAM_FEATURE_DIM:
            raise ValueError("NGAD canonical model inputs require action_dim=128 and proprio_dim=128.")
        if int(action_horizon) <= 0 or int(action_horizon) % int(round(rate_ratio)):
            raise ValueError("action_horizon must be positive and divisible by the rate ratio.")
        expected_frames = int(action_horizon) // int(round(rate_ratio)) + 1
        if int(num_frames) != expected_frames:
            raise ValueError(
                f"num_frames must be {expected_frames} for the configured rates and horizon."
            )
        memory_sizes = (
            recent_memory_frames,
            long_memory_anchor_interval_frames,
            long_memory_window_frames,
            long_memory_slots,
            action_history_horizon,
        )
        if any(int(value) <= 0 for value in memory_sizes):
            raise ValueError("All canonical memory sizes must be positive.")
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'.")

        configured_roots: list[dict[str, Any]] = []
        dataset_names: set[str] = set()
        for entry in dataset_dirs:
            if not isinstance(entry, dict) or set(entry) != {
                "name",
                "path",
                "normalization_stats_path",
            }:
                raise TypeError(
                    "Each dataset_dirs entry must contain exactly name, path, and "
                    "normalization_stats_path."
                )
            name = str(entry["name"]).strip()
            if not name or name in dataset_names:
                raise ValueError(f"Dataset names must be non-empty and unique, got {name!r}.")
            dataset_names.add(name)
            configured_roots.append(
                {
                    "name": name,
                    "path": Path(os.path.expanduser(str(entry["path"]))).resolve(),
                    "normalization_stats_path": Path(
                        os.path.expanduser(str(entry["normalization_stats_path"]))
                    ).resolve(),
                }
            )

        physical_roots: list[tuple[Path, Path, dict[str, Any]]] = []
        for configured in configured_roots:
            configured_root = configured["path"]
            if (configured_root / "meta" / "info.json").is_file():
                physical_roots.append((configured_root, configured_root, configured))
                continue
            fragment_infos = sorted(
                configured_root.glob("table_*/fragments/*/meta/info.json")
            )
            if not fragment_infos:
                fragment_infos = sorted(configured_root.glob("shard_*/meta/info.json"))
            if not fragment_infos:
                raise ValueError(
                    f"{configured_root} is not a supported canonical root or shard collection."
                )
            physical_roots.extend(
                (info_path.parent.parent, configured_root, configured)
                for info_path in fragment_infos
            )

        self.roots = [physical_root for physical_root, _, _ in physical_roots]
        self.camera_keys = self.expected_camera_keys
        self.target_rgb_fps = float(target_rgb_fps)
        self.target_action_fps = float(target_action_fps)
        self.action_horizon = int(action_horizon)
        self.num_frames = int(num_frames)
        self.recent_memory_frames = int(recent_memory_frames)
        self.long_memory_anchor_interval_frames = int(long_memory_anchor_interval_frames)
        self.long_memory_window_frames = int(long_memory_window_frames)
        self.long_memory_slots = int(long_memory_slots)
        self.action_history_horizon = int(action_history_horizon)
        self.resolution = int(resolution)
        self.load_vae_feat = False
        self.load_text_feat = False
        self.aspect_ratio = {"1.00": [self.resolution, self.resolution]}

        self._root_meta: list[dict[str, Any]] = []
        pixel_masks_by_dataset: dict[Path, torch.Tensor] = {}
        self._episodes: list[dict[str, Any]] = []
        self._episode_window_ends: list[int] = []
        total_windows = 0
        transforms: dict[str, CanonicalTCPTransform] = {}
        normalization_stats: dict[str, dict[str, Any]] = {}
        for configured in configured_roots:
            stats = _read_json_object(configured["normalization_stats_path"])
            transform = self._normalization_transform(stats)
            transforms[configured["name"]] = transform
            normalization_stats[configured["name"]] = stats

        for root_index, (root, dataset_root, configured) in enumerate(physical_roots):
            info = _read_json_object(root / "meta" / "info.json")
            backend = self._detect_backend(root, info)
            self._validate_features(root, info.get("features", {}), backend)
            source_fps = float(info.get("fps", 0.0))
            if not np.isfinite(source_fps) or source_fps <= 0:
                raise ValueError(f"{root} must declare a positive source fps.")
            if source_fps < self.target_rgb_fps:
                raise ValueError(
                    f"{root} source fps {source_fps} is below target_rgb_fps "
                    f"{self.target_rgb_fps}; RGB sampling never synthesizes frames."
                )
            source_rgb_ratio = source_fps / self.target_rgb_fps
            if abs(source_rgb_ratio - round(source_rgb_ratio)) > 1.0e-9:
                raise ValueError(
                    f"{root} source fps {source_fps} is not an integer multiple of "
                    f"target_rgb_fps {self.target_rgb_fps}; RGB anchors must be real frames."
                )
            tasks, episode_tasks, episodes = self._read_metadata(root, backend)
            data_file_starts: dict[tuple[int, int], int] = {}
            if backend == "lerobot_v3":
                for episode in episodes:
                    file_key = (episode["data_chunk_index"], episode["data_file_index"])
                    data_file_starts[file_key] = min(
                        data_file_starts.get(file_key, episode["dataset_from_index"]),
                        episode["dataset_from_index"],
                    )
            train_episodes, validation_episodes = split_episode_indices(
                [episode["episode_index"] for episode in episodes],
                validation_split=float(validation_split),
                seed=int(validation_seed) + root_index,
            )
            selected = train_episodes if split == "train" else validation_episodes
            if dataset_root not in pixel_masks_by_dataset:
                pixel_masks_by_dataset[dataset_root] = self._load_pixel_masks(dataset_root)
            self._root_meta.append(
                {
                    "root": root,
                    "dataset_root": dataset_root,
                    "backend": backend,
                    "info": info,
                    "tasks": tasks,
                    "episode_tasks": episode_tasks,
                    "source_fps": source_fps,
                    "normalization_id": configured["name"],
                    "tcp_transform": transforms[configured["name"]],
                    "data_file_starts": data_file_starts,
                    "lance_path": root,
                    "arm_mask": self._backend_masks(backend)[0],
                    "camera_mask": self._backend_masks(backend)[1],
                    "pixel_masks": pixel_masks_by_dataset[dataset_root],
                }
            )
            for episode in episodes:
                if episode["episode_index"] not in selected:
                    continue
                rgb_target_length = self._target_episode_length(
                    episode["length"], source_fps, self.target_rgb_fps
                )
                action_target_length = self._target_episode_length(
                    episode["length"], source_fps, self.target_action_fps
                )
                self._episodes.append(
                    {
                        "root_index": root_index,
                        "rgb_target_length": rgb_target_length,
                        "action_target_length": action_target_length,
                        **episode,
                    }
                )
                total_windows += rgb_target_length
                self._episode_window_ends.append(total_windows)

        if not self._episodes:
            raise ValueError(f"NGAD canonical split '{split}' has no episodes.")
        self.ori_imgs_nums = total_windows
        if max_samples is not None and int(max_samples) <= 0:
            raise ValueError("max_samples must be positive when provided.")
        self._length = min(total_windows, int(max_samples)) if max_samples is not None else total_windows
        self.ratio_nums = {next(iter(self.aspect_ratio)): self._length}

        self._normalization_stats = {
            "schema_version": "ngad_canonical_normalization_map_v1",
            "datasets": normalization_stats,
        }
        self._tcp_transforms = transforms
        self._lance_handles: dict[int, Any] = {}
        self._lance_pid = os.getpid()
        self._parquet_handles: dict[tuple[int, int, int], Any] = {}
        self._parquet_row_group_ends: dict[tuple[int, int, int], list[int]] = {}
        self._parquet_pid = os.getpid()

    @staticmethod
    def _detect_backend(root: Path, info: dict[str, Any]) -> str:
        if (
            info.get("canonical_schema") == HY_CANONICAL_SCHEMA
            and (root / "_versions").is_dir()
            and len(list((root / "data").glob("*.lance"))) == 1
        ):
            return "lance_jpeg"
        if info.get("data_path") and info.get("video_path"):
            return "lerobot_v3"
        raise ValueError(f"Cannot identify a supported NGAD canonical storage backend under {root}.")

    @staticmethod
    def _backend_masks(backend: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return temporary arm validity and the fixed six-view canonical camera mask."""
        if backend == "lance_jpeg":
            return torch.tensor([True, True]), torch.ones(6, dtype=torch.bool)
        if backend == "lerobot_v3":
            return torch.tensor([True, False]), torch.ones(6, dtype=torch.bool)
        raise ValueError(f"Unsupported canonical backend: {backend}.")

    @staticmethod
    def _reshape_state_window(absolute_state: torch.Tensor) -> torch.Tensor:
        """Interpret the stored flat TCP20 ABI as two ordered TCP10 arms."""
        if absolute_state.ndim != 2 or absolute_state.shape[1] != DUAL_ARM_TCP_FEATURE_DIM:
            raise ValueError(
                "Canonical observation.state window must be [T,20], "
                f"got {tuple(absolute_state.shape)}."
            )
        return absolute_state.reshape(absolute_state.shape[0], 2, 10)

    def _load_pixel_masks(self, dataset_root: Path) -> torch.Tensor:
        """Load one static per-camera validity mask and align it with decoded RGB size."""
        path = dataset_root / PIXEL_MASKS_FILENAME
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(self.camera_keys):
                raise ValueError(
                    f"{path} keys must exactly match canonical cameras {self.camera_keys}."
                )
            masks = []
            for camera in self.camera_keys:
                mask = archive[camera]
                if mask.dtype != np.bool_ or mask.shape != (256, 256):
                    raise ValueError(f"{path} mask {camera} must be bool [256,256].")
                masks.append(torch.from_numpy(mask.copy()))
        pixel_masks = torch.stack(masks).unsqueeze(1).float()
        pixel_masks = F.interpolate(
            pixel_masks,
            size=(self.resolution, self.resolution),
            mode="nearest",
        )
        return pixel_masks[:, 0].bool()

    @staticmethod
    def _target_episode_length(
        source_length: int,
        source_fps: float,
        target_fps: float,
    ) -> int:
        """Return the number of slots on the target grid covered by one episode."""
        if int(source_length) <= 0:
            raise ValueError("Canonical episodes must contain at least one source frame.")
        duration = (int(source_length) - 1) / source_fps
        return int(np.floor(duration * target_fps + 1e-9)) + 1

    @staticmethod
    def _source_indices(
        target_indices: torch.Tensor,
        source_fps: float,
        target_fps: float,
    ) -> torch.Tensor:
        """Map target-grid slots to the nearest source frames without interpolation."""
        source = torch.round(
            target_indices.to(torch.float64) * source_fps / target_fps
        ).to(torch.long)
        target_time = target_indices.to(torch.float64) / target_fps
        source_time = source.to(torch.float64) / source_fps
        tolerance = 0.5 / source_fps + 1e-9
        if torch.any(torch.abs(source_time - target_time) > tolerance):
            raise RuntimeError("Nearest-frame sampling exceeded half a source-frame interval.")
        return source

    @staticmethod
    def _state_interpolation_indices(
        target_indices: torch.Tensor,
        source_fps: float,
        target_fps: float,
        source_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return source brackets and interpolation fractions for one target grid."""
        position = target_indices.to(torch.float64) * source_fps / target_fps
        lower = torch.floor(position).to(torch.long).clamp(0, source_length - 1)
        upper = torch.ceil(position).to(torch.long).clamp(0, source_length - 1)
        fraction = (position - lower.to(torch.float64)).to(torch.float32)
        fraction = torch.where(lower == upper, torch.zeros_like(fraction), fraction)
        return lower, upper, fraction

    def _validate_features(self, root: Path, features: dict[str, Any], backend: str) -> None:
        expected = {
            "observation.state": [20],
            "action": [20],
            CANONICAL_TACTILE_VALUES_KEY: [4, 3, 25, 6],
            CANONICAL_TACTILE_DT_KEY: [4, 3],
            "timestamp": [1],
        }
        for name, shape in expected.items():
            if features.get(name, {}).get("shape") != shape:
                raise ValueError(f"{root} feature {name} must have shape {shape}.")
        expected_image_dtype = "video"
        for camera in self.camera_keys:
            feature = features.get(camera, {})
            if feature.get("dtype") != expected_image_dtype or feature.get("shape") != [256, 256, 3]:
                raise ValueError(
                    f"{root} camera {camera} must be {expected_image_dtype} [256,256,3]."
                )

    def _read_metadata(
        self,
        root: Path,
        backend: str,
    ) -> tuple[dict[int, str], dict[int, str], list[dict[str, Any]]]:
        if backend == "lance_jpeg":
            try:
                import pyarrow.dataset as ds
                import pyarrow.parquet as pq
            except ImportError as error:
                raise ImportError("Lance canonical metadata requires pyarrow.") from error
            task_rows = pq.read_table(root / "meta" / "tasks.parquet").to_pylist()
            tasks = {int(row["task_index"]): str(row["task"]) for row in task_rows}
            episode_rows = ds.dataset(root / "meta" / "episodes", format="parquet").to_table(
                columns=["episode_index", "length", "dataset_from_index", "dataset_to_index"]
            ).to_pylist()
            episodes = [
                {
                    "episode_index": int(row["episode_index"]),
                    "length": int(row["length"]),
                    "dataset_from_index": int(row["dataset_from_index"]),
                    "dataset_to_index": int(row["dataset_to_index"]),
                }
                for row in episode_rows
            ]
            for episode in episodes:
                if episode["dataset_to_index"] - episode["dataset_from_index"] != episode["length"]:
                    raise ValueError(f"Invalid Lance episode offsets under {root}: {episode}.")
            return tasks, {}, sorted(episodes, key=lambda row: row["dataset_from_index"])

        try:
            import pyarrow.dataset as ds
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("LeRobot v3 canonical metadata requires pyarrow.") from error

        task_rows = pq.read_table(root / "meta" / "tasks.parquet").to_pylist()
        tasks = {int(row["task_index"]): str(row["task"]) for row in task_rows}

        video_columns = [
            f"videos/{camera}/{field}"
            for camera in self.camera_keys
            for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
        ]
        columns = [
            "episode_index",
            "tasks",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
            *video_columns,
        ]
        episode_rows = (
            ds.dataset(root / "meta" / "episodes", format="parquet")
            .to_table(columns=columns)
            .to_pylist()
        )
        episodes: list[dict[str, Any]] = []
        for row in episode_rows:
            episode_index = int(row["episode_index"])
            task_values = row["tasks"]
            if not isinstance(task_values, list) or not task_values or not all(task_values):
                raise ValueError(
                    f"LeRobot v3 episode {episode_index} under {root} must list its tasks."
                )
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
                        for field in (
                            "chunk_index",
                            "file_index",
                            "from_timestamp",
                            "to_timestamp",
                        )
                    }
                    for camera in self.camera_keys
                },
            }
            if episode["dataset_to_index"] - episode["dataset_from_index"] != episode["length"]:
                raise ValueError(f"Invalid LeRobot v3 episode offsets under {root}: {episode}.")
            for camera, video in episode["videos"].items():
                video["chunk_index"] = int(video["chunk_index"])
                video["file_index"] = int(video["file_index"])
                video["from_timestamp"] = float(video["from_timestamp"])
                video["to_timestamp"] = float(video["to_timestamp"])
                if video["to_timestamp"] <= video["from_timestamp"]:
                    raise ValueError(
                        f"Invalid LeRobot v3 video range for {camera} in episode {episode_index}."
                    )
            episodes.append(episode)
        return tasks, {}, sorted(
            episodes, key=lambda record: record["dataset_from_index"]
        )

    @staticmethod
    def _normalization_transform(stats: dict[str, Any]) -> CanonicalTCPTransform:
        """Validate one external stats object and construct its source transform."""
        if stats.get("schema_version") != NGAD_CANONICAL_SCHEMA:
            raise ValueError(
                f"Expected normalization schema {NGAD_CANONICAL_SCHEMA}, got {stats.get('schema_version')}."
            )
        state_xyz_min = torch.tensor(stats["state_xyz_min"], dtype=torch.float32)
        state_xyz_max = torch.tensor(stats["state_xyz_max"], dtype=torch.float32)
        action_xyz_scale = torch.tensor(stats["action_xyz_scale"], dtype=torch.float32)
        for name, value in (
            ("state_xyz_min", state_xyz_min),
            ("state_xyz_max", state_xyz_max),
            ("action_xyz_scale", action_xyz_scale),
        ):
            if value.shape != (2, 3) or not torch.isfinite(value).all():
                raise ValueError(f"Canonical normalization {name} must be finite [2,3].")
        if torch.any(state_xyz_max <= state_xyz_min) or torch.any(action_xyz_scale <= 0):
            raise ValueError("Canonical xyz normalization ranges must be positive.")
        return CanonicalTCPTransform(state_xyz_min, state_xyz_max, action_xyz_scale)

    def __len__(self) -> int:
        return self._length

    def normalization_stats(self) -> dict[str, Any]:
        """Return the exact canonical statistics serialized with a checkpoint."""
        return dict(self._normalization_stats)

    def denormalize_action(
        self,
        action: torch.Tensor,
        normalization_id: str,
    ) -> torch.Tensor:
        """Denormalize with the explicitly selected dataset statistics."""
        return self._tcp_transforms[normalization_id].denormalize_action(action)

    def _locate_window(self, index: int) -> tuple[dict[str, Any], int]:
        if not 0 <= int(index) < self._length:
            raise IndexError(index)
        episode_position = bisect_right(self._episode_window_ends, int(index))
        previous_end = 0 if episode_position == 0 else self._episode_window_ends[episode_position - 1]
        return self._episodes[episode_position], int(index) - previous_end

    def _lance_dataset(self, root_index: int):
        pid = os.getpid()
        if pid != self._lance_pid:
            self._lance_handles = {}
            self._lance_pid = pid
        if root_index not in self._lance_handles:
            try:
                import lance
            except ImportError as error:
                raise ImportError("Lance canonical roots require the pylance package.") from error
            self._lance_handles[root_index] = lance.dataset(
                str(self._root_meta[root_index]["lance_path"])
            )
        return self._lance_handles[root_index]

    def _take_lance_rows(
        self,
        episode: dict[str, int],
        relative_indices: torch.Tensor,
    ) -> dict[int, dict[str, Any]]:
        try:
            import pyarrow as pa
        except ImportError as error:
            raise ImportError("Lance canonical roots require pyarrow.") from error
        offsets = {episode["dataset_from_index"]}
        offsets.update(
            episode["dataset_from_index"] + int(relative_index)
            for relative_index in relative_indices.tolist()
        )
        offsets = sorted(offsets)
        columns = [
            "index",
            "episode_index",
            "frame_index",
            "task_index",
            "timestamp",
            _lance_column("observation.state"),
            _lance_column(CANONICAL_TACTILE_VALUES_KEY),
            _lance_column(CANONICAL_TACTILE_DT_KEY),
            *[_lance_column(camera) for camera in self.camera_keys],
        ]
        table = self._lance_dataset(episode["root_index"]).take(
            pa.array(offsets, type=pa.int64()), columns=columns
        )
        rows = table.to_pylist()
        # Lance take() consumes fragment-local physical offsets, while canonical
        # index remains global across fragments and episodes.
        anchor_position = offsets.index(episode["dataset_from_index"])
        global_index_start = int(rows[anchor_position]["index"])
        by_relative_index: dict[int, dict[str, Any]] = {}
        for offset, row in zip(offsets, rows):
            relative_index = offset - episode["dataset_from_index"]
            if (
                int(row["index"]) != global_index_start + relative_index
                or int(row["episode_index"]) != episode["episode_index"]
                or int(row["frame_index"]) != relative_index
            ):
                raise RuntimeError(f"Canonical Lance row identity mismatch at offset {offset}.")
            by_relative_index[relative_index] = row
        return by_relative_index

    def _lerobot_v3_data_file(self, episode: dict[str, Any]):
        """Open the shared Parquet shard containing one LeRobot v3 episode."""
        pid = os.getpid()
        if pid != self._parquet_pid:
            self._parquet_handles = {}
            self._parquet_row_group_ends = {}
            self._parquet_pid = pid
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("LeRobot canonical roots require pyarrow.") from error

        root_index = episode["root_index"]
        chunk_index = episode["data_chunk_index"]
        file_index = episode["data_file_index"]
        cache_key = (root_index, chunk_index, file_index)
        if cache_key in self._parquet_handles:
            return self._parquet_handles[cache_key], self._parquet_row_group_ends[cache_key]

        meta = self._root_meta[root_index]
        info = meta["info"]
        path = meta["root"] / str(info["data_path"]).format(
            chunk_index=chunk_index,
            file_index=file_index,
        )
        parquet_file = pq.ParquetFile(path)
        row_group_ends: list[int] = []
        row_count = 0
        for row_group_index in range(parquet_file.num_row_groups):
            row_count += parquet_file.metadata.row_group(row_group_index).num_rows
            row_group_ends.append(row_count)
        self._parquet_handles[cache_key] = parquet_file
        self._parquet_row_group_ends[cache_key] = row_group_ends
        return parquet_file, row_group_ends

    def _take_lerobot_v3_rows(
        self,
        episode: dict[str, Any],
        relative_indices: torch.Tensor,
    ) -> dict[int, dict[str, Any]]:
        """Read only requested episode rows from a shared LeRobot v3 Parquet shard."""
        try:
            import pyarrow as pa
        except ImportError as error:
            raise ImportError("LeRobot canonical roots require pyarrow.") from error

        requested = {0}
        requested.update(int(index) for index in relative_indices.tolist())
        meta = self._root_meta[episode["root_index"]]
        file_key = (episode["data_chunk_index"], episode["data_file_index"])
        file_start = meta["data_file_starts"][file_key]
        local_rows = {
            episode["dataset_from_index"] + relative_index - file_start: relative_index
            for relative_index in requested
        }
        parquet_file, row_group_ends = self._lerobot_v3_data_file(episode)
        if not local_rows or min(local_rows) < 0 or max(local_rows) >= row_group_ends[-1]:
            raise IndexError(f"LeRobot v3 episode offsets exceed their data shard: {episode}.")

        columns = [
            "index",
            "episode_index",
            "frame_index",
            "task_index",
            "timestamp",
            "observation.state",
            CANONICAL_TACTILE_VALUES_KEY,
            CANONICAL_TACTILE_DT_KEY,
        ]
        by_row_group: dict[int, list[int]] = {}
        for local_row in sorted(local_rows):
            row_group_index = bisect_right(row_group_ends, local_row)
            by_row_group.setdefault(row_group_index, []).append(local_row)

        rows_by_relative_index: dict[int, dict[str, Any]] = {}
        for row_group_index, group_rows in by_row_group.items():
            group_start = 0 if row_group_index == 0 else row_group_ends[row_group_index - 1]
            table = parquet_file.read_row_group(row_group_index, columns=columns)
            table = table.take(pa.array([row - group_start for row in group_rows], type=pa.int64()))
            for local_row, row in zip(group_rows, table.to_pylist()):
                relative_index = local_rows[local_row]
                global_index = episode["dataset_from_index"] + relative_index
                if (
                    int(row["index"]) != global_index
                    or int(row["episode_index"]) != episode["episode_index"]
                    or int(row["frame_index"]) != relative_index
                ):
                    raise RuntimeError(
                        f"Canonical LeRobot v3 row identity mismatch at index {global_index}."
                    )
                rows_by_relative_index[relative_index] = row
        return rows_by_relative_index

    def _validate_sample_timestamps(
        self,
        episode: dict[str, int],
        source_indices: torch.Tensor,
        actual_timestamps: torch.Tensor,
        timestamp_start: torch.Tensor,
    ) -> None:
        """Check source rows against their declared uniform episode timeline."""
        source_fps = self._root_meta[episode["root_index"]]["source_fps"]
        if actual_timestamps.numel() > 1 and torch.any(torch.diff(actual_timestamps) <= 0):
            raise ValueError(
                f"Canonical timestamps are not strictly increasing in episode {episode['episode_index']}."
            )
        expected = timestamp_start + source_indices.to(torch.float64) / source_fps
        tolerance = 1.0e-4
        if torch.any(torch.abs(actual_timestamps - expected) > tolerance):
            raise ValueError(
                f"Canonical source timestamps do not match the declared fps in "
                f"episode {episode['episode_index']}."
            )

    def _decode_lance_camera(
        self,
        rows: dict[int, dict[str, Any]],
        camera: str,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        frames = []
        column = _lance_column(camera)
        for index in indices.tolist():
            payload = rows[int(index)][column]
            with Image.open(BytesIO(payload)) as image:
                image.load()
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if rgb.shape != (256, 256, 3):
                raise ValueError(f"Canonical JPEG {camera} frame {index} has shape {rgb.shape}.")
            frames.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1))
        video = torch.stack(frames).float() / 127.5 - 1.0
        return self._resize_video(video)

    def _resize_video(self, video: torch.Tensor) -> torch.Tensor:
        """Resize decoded camera frames to the model-facing square resolution."""
        return F.interpolate(
            video,
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _video_frame_rate(stream: Any, fallback: float) -> float:
        """Use the nominal fixed rate instead of duration-skewed MP4 average rate."""
        return float(stream.base_rate or stream.average_rate or fallback)

    def _decode_video_camera(
        self,
        episode: dict[str, Any],
        camera: str,
        source_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Decode episode-relative frames from a shared LeRobot v3 MP4 shard."""
        try:
            import av
        except ImportError as error:
            raise ImportError("LeRobot canonical roots require PyAV.") from error
        meta = self._root_meta[episode["root_index"]]
        info = meta["info"]
        video_meta = episode["videos"][camera]
        video_path = meta["root"] / str(info["video_path"]).format(
            video_key=camera,
            chunk_index=video_meta["chunk_index"],
            file_index=video_meta["file_index"],
        )
        requested_relative = [int(index) for index in source_indices.tolist()]
        decoded: dict[int, torch.Tensor] = {}
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            frame_rate = self._video_frame_rate(stream, float(info["fps"]))
            source_fps = meta["source_fps"]
            if abs(frame_rate - source_fps) > 1e-3:
                raise ValueError(
                    f"Video fps {frame_rate} does not match data fps {source_fps} in {video_path}."
                )
            time_base = float(stream.time_base)
            start_pts = int(stream.start_time or 0)
            requested = [
                int(round((video_meta["from_timestamp"] + index / source_fps) * frame_rate))
                for index in requested_relative
            ]
            if any(
                video_meta["from_timestamp"] + index / source_fps
                >= video_meta["to_timestamp"] + 1e-9
                for index in requested_relative
            ):
                raise IndexError(
                    f"Requested frame exceeds the episode video range for {camera} in "
                    f"episode {episode['episode_index']}."
                )
            unique = set(requested)
            first, last = min(unique), max(unique)
            container.seek(
                start_pts + int((first / frame_rate) / time_base),
                stream=stream,
                backward=True,
                any_frame=False,
            )
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                frame_index = int(round((int(frame.pts) - start_pts) * time_base * frame_rate))
                if frame_index < first:
                    continue
                if frame_index > last:
                    break
                if frame_index in unique and frame_index not in decoded:
                    decoded[frame_index] = torch.from_numpy(frame.to_ndarray(format="rgb24"))
                    if len(decoded) == len(unique):
                        break
        missing = sorted(unique.difference(decoded))
        if missing:
            raise RuntimeError(f"Failed to decode canonical frames {missing} from {video_path}.")
        video = torch.stack([decoded[index] for index in requested]).permute(0, 3, 1, 2).float()
        video = video / 127.5 - 1.0
        return self._resize_video(video)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, start = self._locate_window(index)
        target_observation_indices, target_action_indices, image_is_pad, action_is_pad = wam_window_indices(
            start,
            rgb_episode_length=episode["rgb_target_length"],
            action_episode_length=episode["action_target_length"],
            action_horizon=self.action_horizon,
            target_rgb_fps=self.target_rgb_fps,
            target_action_fps=self.target_action_fps,
        )
        memory = wam_memory_indices(
            start,
            rgb_episode_length=episode["rgb_target_length"],
            action_episode_length=episode["action_target_length"],
            target_rgb_fps=self.target_rgb_fps,
            target_action_fps=self.target_action_fps,
            recent_memory_frames=self.recent_memory_frames,
            long_memory_anchor_interval_frames=self.long_memory_anchor_interval_frames,
            long_memory_window_frames=self.long_memory_window_frames,
            long_memory_slots=self.long_memory_slots,
            action_history_horizon=self.action_history_horizon,
        )
        meta = self._root_meta[episode["root_index"]]
        source_fps = meta["source_fps"]
        all_rgb_target_indices = torch.cat(
            [
                target_observation_indices,
                memory.recent_rgb,
                memory.long_rgb.reshape(-1),
            ]
        )
        all_observation_indices = self._source_indices(
            all_rgb_target_indices, source_fps, self.target_rgb_fps
        )
        observation_count = target_observation_indices.numel()
        recent_count = memory.recent_rgb.numel()
        observation_indices = all_observation_indices[:observation_count]
        action_steps_per_rgb = int(round(self.target_action_fps / self.target_rgb_fps))
        anchor_action_index = torch.tensor([start * action_steps_per_rgb], dtype=torch.long)
        state_target_indices = torch.cat(
            [anchor_action_index, target_action_indices, memory.action_history]
        )
        state_lower, state_upper, state_fraction = self._state_interpolation_indices(
            state_target_indices,
            source_fps,
            self.target_action_fps,
            episode["length"],
        )
        requested = torch.unique(
            torch.cat([all_observation_indices, state_lower, state_upper]), sorted=True
        )
        if meta["backend"] == "lance_jpeg":
            rows = self._take_lance_rows(episode, requested)
            source_timestamps = torch.tensor(
                [rows[int(frame)]["timestamp"] for frame in requested.tolist()],
                dtype=torch.float64,
            ).reshape(-1)
            self._validate_sample_timestamps(
                episode,
                requested,
                source_timestamps,
                torch.as_tensor(rows[0]["timestamp"], dtype=torch.float64).reshape(()),
            )
            lower_state = self._reshape_state_window(torch.tensor(
                [rows[int(frame)][_lance_column("observation.state")] for frame in state_lower.tolist()],
                dtype=torch.float32,
            ))
            upper_state = self._reshape_state_window(torch.tensor(
                [rows[int(frame)][_lance_column("observation.state")] for frame in state_upper.tolist()],
                dtype=torch.float32,
            ))
            current_row = rows[int(observation_indices[0])]
            tactile_values = torch.as_tensor(
                current_row[_lance_column(CANONICAL_TACTILE_VALUES_KEY)],
                dtype=torch.float32,
            )
            tactile_dt = torch.as_tensor(
                current_row[_lance_column(CANONICAL_TACTILE_DT_KEY)],
                dtype=torch.float32,
            )
            arm_mask = meta["arm_mask"].clone()
            camera_mask = meta["camera_mask"].clone()
            all_cameras = [
                self._decode_lance_camera(rows, camera, all_observation_indices)
                for camera in self.camera_keys
            ]
            task_index = int(current_row["task_index"])
        else:
            rows = self._take_lerobot_v3_rows(episode, requested)
            source_timestamps = torch.tensor(
                [rows[int(frame)]["timestamp"] for frame in requested.tolist()],
                dtype=torch.float64,
            ).reshape(-1)
            self._validate_sample_timestamps(
                episode,
                requested,
                source_timestamps,
                torch.as_tensor(rows[0]["timestamp"], dtype=torch.float64).reshape(()),
            )
            lower_state = self._reshape_state_window(torch.tensor(
                [rows[int(frame)]["observation.state"] for frame in state_lower.tolist()],
                dtype=torch.float32,
            ))
            upper_state = self._reshape_state_window(torch.tensor(
                [rows[int(frame)]["observation.state"] for frame in state_upper.tolist()],
                dtype=torch.float32,
            ))
            current_row = rows[int(observation_indices[0])]
            tactile_values = torch.as_tensor(
                current_row[CANONICAL_TACTILE_VALUES_KEY], dtype=torch.float32
            )
            tactile_dt = torch.as_tensor(
                current_row[CANONICAL_TACTILE_DT_KEY], dtype=torch.float32
            )
            arm_mask = meta["arm_mask"].clone()
            camera_mask = meta["camera_mask"].clone()
            all_cameras = [
                self._decode_video_camera(episode, camera, all_observation_indices)
                for camera in self.camera_keys
            ]
            task_index = int(current_row["task_index"])

        absolute_state_grid = interpolate_canonical_tcp(
            lower_state, upper_state, state_fraction
        )
        tcp_transform = meta["tcp_transform"]
        action, action_feature_mask = tcp_transform.encode_action_targets(
            absolute_state_grid[0],
            absolute_state_grid[1 : 1 + self.action_horizon],
            arm_mask,
        )
        proprio, proprio_feature_mask = tcp_transform.encode_proprio(
            absolute_state_grid[0], arm_mask
        )
        action_history, action_history_feature_mask = tcp_transform.encode_action_targets(
            absolute_state_grid[0],
            absolute_state_grid[1 + self.action_horizon :],
            arm_mask,
        )
        action_history = action_history * memory.action_history_valid[:, None].to(
            action_history.dtype
        )

        camera_tensor = torch.stack(all_cameras, dim=0)
        main_cameras = camera_tensor[:, :observation_count]
        recent_cameras = camera_tensor[
            :, observation_count : observation_count + recent_count
        ]
        long_cameras = camera_tensor[:, observation_count + recent_count :].reshape(
            len(self.camera_keys),
            self.long_memory_slots,
            self.long_memory_window_frames,
            3,
            self.resolution,
            self.resolution,
        )
        main_cameras = torch.where(
            image_is_pad[None, :, None, None, None],
            torch.full_like(main_cameras, -1.0),
            main_cameras,
        )
        recent_cameras = torch.where(
            memory.recent_valid[None, :, None, None, None],
            recent_cameras,
            torch.full_like(recent_cameras, -1.0),
        )
        long_cameras = torch.where(
            memory.long_valid[None, :, :, None, None, None],
            long_cameras,
            torch.full_like(long_cameras, -1.0),
        )
        # Keep the canonical camera axis explicit for shared-VAE tokenization.
        video = main_cameras.permute(0, 2, 1, 3, 4).contiguous()
        recent_memory = recent_cameras.permute(0, 2, 1, 3, 4).contiguous()
        long_memory = long_cameras.permute(1, 0, 3, 2, 4, 5).contiguous()

        base_pixel_mask = meta["pixel_masks"] & camera_mask[:, None, None]
        image_pixel_mask = base_pixel_mask[:, None].expand(
            -1, video.shape[2], -1, -1
        ) & ~image_is_pad[None, :, None, None]
        recent_memory_pixel_mask = base_pixel_mask[:, None].expand(
            -1, self.recent_memory_frames, -1, -1
        ) & memory.recent_valid[None, :, None, None]
        long_memory_pixel_mask = base_pixel_mask[None, :, None].expand(
            self.long_memory_slots,
            -1,
            self.long_memory_window_frames,
            -1,
            -1,
        ) & memory.long_valid[:, None, :, None, None]
        task = meta["episode_tasks"].get(episode["episode_index"]) or meta["tasks"][task_index]
        episode_timestamp_start = torch.as_tensor(
            rows[0]["timestamp"], dtype=torch.float64
        ).reshape(())
        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "recent_memory": recent_memory,
            "long_memory": long_memory,
            "action_history": action_history,
            "action_feature_mask": action_feature_mask,
            "proprio_feature_mask": proprio_feature_mask,
            "action_history_feature_mask": action_history_feature_mask,
            "arm_mask": arm_mask,
            CANONICAL_TACTILE_VALUES_KEY: tactile_values,
            CANONICAL_TACTILE_DT_KEY: tactile_dt,
            "camera_view_mask": camera_mask,
            "image_pixel_mask": image_pixel_mask,
            "recent_memory_pixel_mask": recent_memory_pixel_mask,
            "long_memory_pixel_mask": long_memory_pixel_mask,
            "prompt": DEFAULT_PROMPT.format(task=task),
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "recent_memory_valid": memory.recent_valid,
            "long_memory_valid": memory.long_valid,
            "action_history_valid": memory.action_history_valid,
            "data_info": {
                "img_hw": torch.tensor([video.shape[-2], video.shape[-1]], dtype=torch.float32),
                "aspect_ratio": torch.tensor(video.shape[-1] / video.shape[-2], dtype=torch.float32),
                "root_index": episode["root_index"],
                "episode_index": episode["episode_index"],
                "task_index": task_index,
                "normalization_id": meta["normalization_id"],
                "source_fps": source_fps,
                "target_rgb_fps": self.target_rgb_fps,
                "target_action_fps": self.target_action_fps,
                "anchor_timestamp": episode_timestamp_start
                + start / self.target_rgb_fps,
                "observation_timestamps": episode_timestamp_start
                + target_observation_indices.to(torch.float64) / self.target_rgb_fps,
                "action_timestamps": episode_timestamp_start
                + target_action_indices.to(torch.float64) / self.target_action_fps,
                "recent_memory_timestamps": episode_timestamp_start
                + memory.recent_rgb.to(torch.float64) / self.target_rgb_fps,
                "long_memory_timestamps": episode_timestamp_start
                + memory.long_rgb.to(torch.float64) / self.target_rgb_fps,
                "action_history_timestamps": episode_timestamp_start
                + memory.action_history.to(torch.float64) / self.target_action_fps,
                "start": start,
            },
        }
