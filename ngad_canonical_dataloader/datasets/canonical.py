"""Canonical NGAD TCP dataset and model-facing TCP128 transformation."""

from __future__ import annotations

from bisect import bisect_right
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ngad_canonical_dataloader.backends import create_storage_backends
from ngad_canonical_dataloader.action import (
    DUAL_ARM_TCP_FEATURE_DIM,
    WAM_FEATURE_DIM,
    denormalize_dual_arm_relative_tcp,
    dual_arm_tcp_target_relative_to_anchor,
    element_mask_to_feature_mask,
    matrix_to_quaternion_xyzw,
    matrix_to_rotation_6d_rows,
    normalize_dual_arm_absolute_tcp,
    normalize_dual_arm_relative_tcp,
    pack_dual_arm_tcp,
    quaternion_slerp_xyzw,
    quaternion_xyzw_to_matrix,
    rotation_6d_rows_to_matrix,
)
from ngad_canonical_dataloader.windows import (
    TimelineLayout,
    build_timeline_layout,
    split_episode_indices,
    timeline_sample_indices,
)


NGAD_CANONICAL_SCHEMA = "ngad_canonical_tcp_v1"
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
CANONICAL_STATE_KEY = "observation.state"
CANONICAL_ACTION_KEY = "action"
CANONICAL_IDENTITY_KEYS = (
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
CANONICAL_IMAGE_SIZE = 256
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
TABLE_DIRECTORY_PATTERN = re.compile(r"table_(\d{3})")


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _table_record_from_root(
    table_root: Path,
    *,
    flat_table_index: int | None = None,
) -> dict[str, Any]:
    """Build one validated physical-table record from its direct root."""
    match = TABLE_DIRECTORY_PATTERN.fullmatch(table_root.name)
    if not table_root.is_dir() or (match is None and flat_table_index is None):
        raise ValueError(f"Canonical table root must be a table_NNN directory: {table_root}.")
    info_path = table_root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"Canonical table {table_root} must contain meta/info.json.")
    info = _read_json_object(info_path)
    num_episodes = int(info.get("total_episodes", 0))
    num_frames = int(info.get("total_frames", 0))
    if num_episodes <= 0 or num_frames <= 0:
        raise ValueError(
            f"{info_path} must declare positive total_episodes and total_frames."
        )
    return {
        "table_index": (
            int(match.group(1)) if flat_table_index is None else int(flat_table_index)
        ),
        "table_name": table_root.name,
        "table_root": table_root.resolve(),
        "num_episodes": num_episodes,
        "num_frames": num_frames,
    }


def _discover_published_tables(dataset_path: Path) -> list[dict[str, Any]]:
    """Resolve one dataset-root, table-root, or flat LeRobot v3 input path."""
    if not dataset_path.is_dir():
        raise ValueError(f"Canonical dataset path does not exist: {dataset_path}.")
    if TABLE_DIRECTORY_PATTERN.fullmatch(dataset_path.name):
        return [_table_record_from_root(dataset_path)]

    table_roots = [
        table_root
        for table_root in dataset_path.iterdir()
        if TABLE_DIRECTORY_PATTERN.fullmatch(table_root.name)
    ]
    is_flat_lerobot = (
        (dataset_path / "meta" / "info.json").is_file()
        and (dataset_path / "data").is_dir()
        and (dataset_path / "videos").is_dir()
    )
    if is_flat_lerobot and table_roots:
        raise ValueError(
            f"Canonical dataset path is ambiguous: {dataset_path} contains both "
            "a flat LeRobot v3 payload and direct table_NNN children."
        )
    if is_flat_lerobot:
        return [_table_record_from_root(dataset_path, flat_table_index=0)]

    records = [_table_record_from_root(table_root) for table_root in table_roots]
    if not records:
        raise ValueError(
            f"{dataset_path} is neither a table_NNN root, a flat LeRobot v3 root, "
            "nor a dataset root with direct table_NNN children."
        )
    return sorted(records, key=lambda row: row["table_index"])


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

    def encode_state_targets(
        self,
        absolute_state_targets: torch.Tensor,
        state_element_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize time-indexed absolute states and encode them in TCP128."""
        feature_mask = element_mask_to_feature_mask(state_element_mask)
        normalized = normalize_dual_arm_absolute_tcp(
            self._flatten_state(absolute_state_targets),
            self.state_xyz_min,
            self.state_xyz_max,
        )
        state = pack_dual_arm_tcp(normalized)
        return state * feature_mask.to(state.dtype), feature_mask

    def encode_action_targets(
        self,
        anchor_state: torch.Tensor,
        absolute_state_targets: torch.Tensor,
        action_element_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode time-indexed states relative to one explicit RGB anchor."""
        feature_mask = element_mask_to_feature_mask(action_element_mask)
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
    lower_rotation = rotation_6d_rows_to_matrix(lower[..., 3:9])
    upper_rotation = rotation_6d_rows_to_matrix(upper[..., 3:9])
    quaternion = quaternion_slerp_xyzw(
        matrix_to_quaternion_xyzw(lower_rotation),
        matrix_to_quaternion_xyzw(upper_rotation),
        fraction,
    )
    rotation = matrix_to_rotation_6d_rows(
        quaternion_xyzw_to_matrix(quaternion)
    )
    return torch.cat([position, rotation, openness], dim=-1)


class NGADCanonicalDataset(Dataset):
    """Read canonical TCP fields from Lance/JPEG or standard LeRobot video roots."""

    expected_camera_keys = CANONICAL_CAMERA_KEYS

    def __init__(
        self,
        dataset_dirs: list[dict[str, Any]],
        normalization_stats_path: str | None,
        rgb_rate_hz: float,
        action_steps_per_rgb_frame: int,
        anchor_offset: int,
        frame_ranges: tuple[tuple[int, int], ...],
        max_samples: int | None = None,
        validation_split: float = 0.0,
        validation_seed: int = 3407,
        split: str = "train",
    ) -> None:
        if not dataset_dirs:
            raise ValueError("NGADCanonicalDataset requires at least one named dataset root.")
        if normalization_stats_path is not None and (
            not isinstance(normalization_stats_path, str)
            or not normalization_stats_path.strip()
        ):
            raise ValueError(
                "normalization_stats_path must be a non-empty string or None."
            )
        if (
            rgb_rate_hz is None
            or not np.isfinite(float(rgb_rate_hz))
            or float(rgb_rate_hz) <= 0
        ):
            raise ValueError("NGADCanonicalDataset requires a positive RGB rate.")
        if int(anchor_offset) != 0:
            raise ValueError("The canonical timeline anchor_offset must be 0.")
        timeline_layout = build_timeline_layout(
            frame_ranges,
            action_steps_per_rgb_frame,
        )
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'.")

        configured_roots: list[dict[str, Any]] = []
        dataset_names: set[str] = set()
        for entry in dataset_dirs:
            if not isinstance(entry, dict) or set(entry) != {
                "name",
                "path",
                "mask_and_mapping_path",
            }:
                raise TypeError(
                    "Each dataset_dirs entry must contain exactly name, path, "
                    "and mask_and_mapping_path."
                )
            name = str(entry["name"]).strip()
            if not name or name in dataset_names:
                raise ValueError(f"Dataset names must be non-empty and unique, got {name!r}.")
            dataset_names.add(name)
            configured_roots.append(
                {
                    "name": name,
                    "path": Path(os.path.expanduser(str(entry["path"]))).resolve(),
                    "mask_and_mapping_path": Path(
                        os.path.expanduser(str(entry["mask_and_mapping_path"]))
                    ).resolve(),
                }
            )

        physical_tables: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for configured in configured_roots:
            configured_root = configured["path"]
            physical_tables.extend(
                (table_record, configured)
                for table_record in _discover_published_tables(configured_root)
            )

        self.camera_keys = self.expected_camera_keys
        self.rgb_rate_hz = float(rgb_rate_hz)
        self.action_steps_per_rgb_frame = int(action_steps_per_rgb_frame)
        self.action_rate_hz = (
            self.rgb_rate_hz * self.action_steps_per_rgb_frame
        )
        self.timeline_layout: TimelineLayout = timeline_layout
        self.resolution = CANONICAL_IMAGE_SIZE
        self.video_only = normalization_stats_path is None

        self._root_meta: list[dict[str, Any]] = []
        self._episodes: list[dict[str, Any]] = []
        self._episode_window_ends: list[int] = []
        total_windows = 0
        self._normalization_stats: dict[str, Any] | None = None
        self._tcp_transform: CanonicalTCPTransform | None = None
        if normalization_stats_path is not None:
            normalization_path = Path(
                os.path.expanduser(normalization_stats_path)
            ).resolve()
            self._normalization_stats = _read_json_object(normalization_path)
            self._tcp_transform = self._normalization_transform(
                self._normalization_stats
            )
        mask_contracts: dict[str, dict[str, Any]] = {}
        for configured in configured_roots:
            mask_contract = self._load_mask_and_mapping_contract(
                configured["mask_and_mapping_path"], configured["name"]
            )
            mask_contract["pixel_mask"] = self._load_pixel_mask(mask_contract)
            mask_contracts[configured["name"]] = mask_contract

        for root_index, (table_record, configured) in enumerate(physical_tables):
            root = table_record["table_root"]
            info = _read_json_object(root / "meta" / "info.json")
            table_backend, image_backend = create_storage_backends(
                root,
                table_record["table_name"],
                info,
            )
            mask_contract = mask_contracts[configured["name"]]
            self._validate_features(
                root,
                info.get("features", {}),
                mask_contract,
                image_backend.feature_dtype,
            )
            source_fps = float(info.get("fps", 0.0))
            if not np.isfinite(source_fps) or source_fps <= 0:
                raise ValueError(f"{root} must declare a positive source fps.")
            if source_fps < self.rgb_rate_hz:
                raise ValueError(
                    f"{root} source fps {source_fps} is below rgb_rate_hz "
                    f"{self.rgb_rate_hz}; RGB sampling never synthesizes frames."
                )
            source_rgb_ratio = source_fps / self.rgb_rate_hz
            if abs(source_rgb_ratio - round(source_rgb_ratio)) > 1.0e-9:
                raise ValueError(
                    f"{root} source fps {source_fps} is not an integer multiple of "
                    f"rgb_rate_hz {self.rgb_rate_hz}; RGB anchors must be real frames."
                )
            tasks, episodes = table_backend.read_catalog(
                self.camera_keys,
                mask_contract["camera_mask"],
                mask_contract["field_mapping"],
            )
            if len(episodes) != table_record["num_episodes"]:
                raise ValueError(
                    f"{root} has {len(episodes)} episodes but info.json declares "
                    f"{table_record['num_episodes']}."
                )
            frame_count = sum(int(episode["length"]) for episode in episodes)
            if frame_count != table_record["num_frames"]:
                raise ValueError(
                    f"{root} has {frame_count} frames but info.json declares "
                    f"{table_record['num_frames']}."
                )
            expected_episode_start = 0
            for episode in episodes:
                if episode["dataset_from_index"] != expected_episode_start:
                    raise ValueError(
                        f"{root} episode offsets must be contiguous table-local indices; "
                        f"expected {expected_episode_start}, got "
                        f"{episode['dataset_from_index']}."
                    )
                expected_episode_start = episode["dataset_to_index"]
            if expected_episode_start != table_record["num_frames"]:
                raise ValueError(
                    f"{root} episode offsets end at {expected_episode_start}, "
                    f"expected {table_record['num_frames']}."
                )
            train_episodes, validation_episodes = split_episode_indices(
                [episode["episode_index"] for episode in episodes],
                validation_split=float(validation_split),
                seed=int(validation_seed) + root_index,
            )
            selected = train_episodes if split == "train" else validation_episodes
            self._root_meta.append(
                {
                    "table_backend": table_backend,
                    "image_backend": image_backend,
                    "tasks": tasks,
                    "source_fps": source_fps,
                    "field_mask": mask_contract["field_mask"],
                    "field_mapping": mask_contract["field_mapping"],
                    "state_element_mask": mask_contract["state_element_mask"],
                    "action_element_mask": mask_contract["action_element_mask"],
                    "camera_mask": mask_contract["camera_mask"],
                    "tactile_mask": mask_contract["tactile_mask"],
                    "pixel_mask": mask_contract["pixel_mask"],
                }
            )
            for episode in episodes:
                if episode["episode_index"] not in selected:
                    continue
                rgb_target_length = self._target_episode_length(
                    episode["length"], source_fps, self.rgb_rate_hz
                )
                action_target_length = self._target_episode_length(
                    episode["length"], source_fps, self.action_rate_hz
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
        if max_samples is not None and int(max_samples) <= 0:
            raise ValueError("max_samples must be positive when provided.")
        self._length = min(total_windows, int(max_samples)) if max_samples is not None else total_windows

    def _load_mask_and_mapping_contract(
        self,
        path: Path,
        dataset_name: str,
    ) -> dict[str, Any]:
        """Load one strict canonical validity and physical-field contract."""
        value = _read_json_object(path)
        expected_sections = {
            "dataset",
            "field_mapping",
            "field_mask",
            "element_mask",
            "image_pixel_mask",
        }
        if set(value) != expected_sections:
            raise ValueError(f"{path} must contain exactly {sorted(expected_sections)}.")
        if value["dataset"] != dataset_name:
            raise ValueError(
                f"{path} dataset must be {dataset_name!r}, got {value['dataset']!r}."
            )

        expected_fields = {
            *self.camera_keys,
            CANONICAL_STATE_KEY,
            CANONICAL_ACTION_KEY,
            CANONICAL_TACTILE_VALUES_KEY,
            CANONICAL_TACTILE_DT_KEY,
            *CANONICAL_IDENTITY_KEYS,
        }
        field_mask = value["field_mask"]
        if not isinstance(field_mask, dict) or set(field_mask) != expected_fields:
            raise ValueError(
                f"{path} field_mask keys must exactly match {sorted(expected_fields)}."
            )
        if any(type(enabled) is not bool for enabled in field_mask.values()):
            raise TypeError(f"{path} field_mask values must be bool.")
        required_fields = {CANONICAL_STATE_KEY, *CANONICAL_IDENTITY_KEYS}
        disabled_required = sorted(key for key in required_fields if not field_mask[key])
        if disabled_required:
            raise ValueError(
                f"{path} disables fields required for window construction: {disabled_required}."
            )

        field_mapping = value["field_mapping"]
        if not isinstance(field_mapping, dict):
            raise TypeError(f"{path} field_mapping must be an object.")
        unknown_mapping_keys = sorted(set(field_mapping).difference(expected_fields))
        if unknown_mapping_keys:
            raise ValueError(
                f"{path} field_mapping contains non-canonical keys: "
                f"{unknown_mapping_keys}."
            )
        disabled_mapping_keys = sorted(
            key for key in field_mapping if not field_mask[key]
        )
        if disabled_mapping_keys:
            raise ValueError(
                f"{path} field_mapping contains disabled fields: "
                f"{disabled_mapping_keys}."
            )
        invalid_physical_keys = sorted(
            canonical
            for canonical, physical in field_mapping.items()
            if not isinstance(physical, str) or not physical.strip()
        )
        if invalid_physical_keys:
            raise ValueError(
                f"{path} field_mapping values must be non-empty strings: "
                f"{invalid_physical_keys}."
            )

        element_mask = value["element_mask"]
        expected_elements = {CANONICAL_STATE_KEY, CANONICAL_ACTION_KEY}
        if not isinstance(element_mask, dict) or set(element_mask) != expected_elements:
            raise ValueError(
                f"{path} element_mask keys must be {sorted(expected_elements)}."
            )
        element_tensors: dict[str, torch.Tensor] = {}
        for key in (CANONICAL_STATE_KEY, CANONICAL_ACTION_KEY):
            elements = element_mask[key]
            if (
                not isinstance(elements, list)
                or len(elements) != DUAL_ARM_TCP_FEATURE_DIM
                or any(type(enabled) is not bool for enabled in elements)
            ):
                raise ValueError(f"{path} element_mask[{key!r}] must be bool [20].")
            if not field_mask[key] and any(elements):
                raise ValueError(
                    f"{path} marks {key} unavailable but enables some of its elements."
                )
            element_tensors[key] = torch.tensor(elements, dtype=torch.bool)
        if torch.any(
            element_tensors[CANONICAL_ACTION_KEY]
            & ~element_tensors[CANONICAL_STATE_KEY]
        ):
            raise ValueError(
                f"{path} enables action elements that cannot be reconstructed from state."
            )

        pixel = value["image_pixel_mask"]
        expected_pixel_fields = {"path", "key", "shape", "applies_to_all_available_images"}
        if not isinstance(pixel, dict) or set(pixel) != expected_pixel_fields:
            raise ValueError(
                f"{path} image_pixel_mask must contain exactly {sorted(expected_pixel_fields)}."
            )
        if pixel["shape"] != [CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE]:
            raise ValueError(
                f"{path} image_pixel_mask shape must be "
                f"[{CANONICAL_IMAGE_SIZE},{CANONICAL_IMAGE_SIZE}]."
            )
        if pixel["applies_to_all_available_images"] is not True:
            raise ValueError(
                f"{path} image_pixel_mask must apply to all available images."
            )
        if not isinstance(pixel["path"], str) or not pixel["path"]:
            raise ValueError(f"{path} image_pixel_mask.path must be a non-empty string.")
        if not isinstance(pixel["key"], str) or not pixel["key"]:
            raise ValueError(f"{path} image_pixel_mask.key must be a non-empty string.")

        camera_mask = torch.tensor(
            [field_mask[camera] for camera in self.camera_keys], dtype=torch.bool
        )
        if not torch.any(camera_mask):
            raise ValueError(f"{path} must enable at least one canonical camera.")
        return {
            "field_mask": field_mask,
            "field_mapping": dict(field_mapping),
            "camera_mask": camera_mask,
            "tactile_mask": torch.tensor(
                [
                    field_mask[CANONICAL_TACTILE_VALUES_KEY],
                    field_mask[CANONICAL_TACTILE_DT_KEY],
                ],
                dtype=torch.bool,
            ),
            "state_element_mask": element_tensors[CANONICAL_STATE_KEY],
            "action_element_mask": element_tensors[CANONICAL_ACTION_KEY],
            "pixel_mask_path": (path.parent / pixel["path"]).resolve(),
            "pixel_mask_key": pixel["key"],
        }

    @staticmethod
    def _reshape_state_window(absolute_state: torch.Tensor) -> torch.Tensor:
        """Interpret the stored flat TCP20 ABI as two ordered TCP10 arms."""
        if absolute_state.ndim != 2 or absolute_state.shape[1] != DUAL_ARM_TCP_FEATURE_DIM:
            raise ValueError(
                "Canonical observation.state window must be [T,20], "
                f"got {tuple(absolute_state.shape)}."
            )
        return absolute_state.reshape(absolute_state.shape[0], 2, 10)

    def _load_pixel_mask(self, contract: dict[str, Any]) -> torch.Tensor:
        """Load the single pixel mask shared by every available canonical camera."""
        path = contract["pixel_mask_path"]
        key = contract["pixel_mask_key"]
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise ValueError(f"{path} does not contain image pixel mask key {key!r}.")
            mask = archive[key]
            if mask.dtype != np.bool_ or mask.shape != (
                CANONICAL_IMAGE_SIZE,
                CANONICAL_IMAGE_SIZE,
            ):
                raise ValueError(
                    f"{path} mask {key!r} must be bool "
                    f"[{CANONICAL_IMAGE_SIZE},{CANONICAL_IMAGE_SIZE}]."
                )
        return torch.from_numpy(mask.copy())

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

    def _validate_features(
        self,
        root: Path,
        features: dict[str, Any],
        contract: dict[str, Any],
        expected_image_dtype: str,
    ) -> None:
        """Validate mapped physical fields declared available by the sidecar."""
        expected = {
            CANONICAL_STATE_KEY: [20],
            CANONICAL_ACTION_KEY: [20],
            CANONICAL_TACTILE_VALUES_KEY: [4, 3, 25, 6],
            CANONICAL_TACTILE_DT_KEY: [4, 3],
            "timestamp": [1],
            "frame_index": [1],
            "episode_index": [1],
            "index": [1],
            "task_index": [1],
        }
        for canonical_name, shape in expected.items():
            if not contract["field_mask"][canonical_name]:
                continue
            physical_name = contract["field_mapping"].get(
                canonical_name, canonical_name
            )
            if physical_name not in features:
                raise ValueError(
                    f"{root} mapping for canonical field {canonical_name!r} points "
                    f"to missing physical field {physical_name!r}."
                )
            if features[physical_name].get("shape") != shape:
                raise ValueError(
                    f"{root} physical feature {physical_name!r} mapped from "
                    f"{canonical_name!r} must have shape {shape}."
                )
        for camera, available in zip(
            self.camera_keys, contract["camera_mask"].tolist()
        ):
            if not available:
                continue
            physical_camera = contract["field_mapping"].get(camera, camera)
            if physical_camera not in features:
                raise ValueError(
                    f"{root} mapping for canonical camera {camera!r} points to "
                    f"missing physical field {physical_camera!r}."
                )
            feature = features[physical_camera]
            if feature.get("dtype") != expected_image_dtype or feature.get("shape") != [256, 256, 3]:
                raise ValueError(
                    f"{root} physical camera {physical_camera!r} mapped from "
                    f"{camera!r} must be {expected_image_dtype} [256,256,3]."
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

    def normalization_stats(self) -> dict[str, Any] | None:
        """Return the exact canonical statistics serialized with a checkpoint."""
        if self._normalization_stats is None:
            return None
        return dict(self._normalization_stats)

    def denormalize_action(
        self,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Denormalize with the global mixed-training statistics."""
        if self._tcp_transform is None:
            raise RuntimeError("Action denormalization is unavailable in video-only mode.")
        return self._tcp_transform.denormalize_action(action)

    def _locate_window(self, index: int) -> tuple[dict[str, Any], int]:
        if not 0 <= int(index) < self._length:
            raise IndexError(index)
        episode_position = bisect_right(self._episode_window_ends, int(index))
        previous_end = 0 if episode_position == 0 else self._episode_window_ends[episode_position - 1]
        return self._episodes[episode_position], int(index) - previous_end

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

    def _prepare_video(self, video: torch.Tensor) -> torch.Tensor:
        """Validate backend-neutral uint8 RGB frames and normalize to [-1,1]."""
        expected = (3, CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE)
        if video.dtype != torch.uint8 or video.ndim != 4 or tuple(video.shape[1:]) != expected:
            raise ValueError(
                "Canonical camera backend must return uint8 [T,3,256,256], "
                f"got dtype={video.dtype}, shape={tuple(video.shape)}."
            )
        return video.float() / 127.5 - 1.0

    def _blank_video(self, frame_count: int) -> torch.Tensor:
        """Create the canonical black value for a camera disabled by field_mask."""
        return torch.full(
            (frame_count, 3, self.resolution, self.resolution),
            -1.0,
            dtype=torch.float32,
        )

    def _build_video_sample(
        self,
        episode: dict[str, Any],
        anchor_rgb_index: int,
        timeline: Any,
        meta: dict[str, Any],
        source_frame_indices: torch.Tensor,
        rows: dict[int, dict[str, Any]],
        current_row: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the shared six-camera and frame-metadata sample fields."""
        source_fps = meta["source_fps"]
        all_cameras = [
            (
                self._prepare_video(
                    meta["image_backend"].read_camera(
                        rows,
                        episode,
                        camera,
                        source_frame_indices,
                        source_fps,
                    )
                )
                if available
                else self._blank_video(source_frame_indices.numel())
            )
            for camera, available in zip(
                self.camera_keys, meta["camera_mask"].tolist()
            )
        ]
        camera_mask = meta["camera_mask"].clone()
        camera_tensor = torch.stack(all_cameras, dim=0)
        camera_tensor = torch.where(
            timeline.frame_valid[None, :, None, None, None],
            camera_tensor,
            torch.full_like(camera_tensor, -1.0),
        )
        video = camera_tensor.permute(1, 0, 2, 3, 4).contiguous()
        base_pixel_mask = meta["pixel_mask"][None].expand(
            len(self.camera_keys), -1, -1
        ) & camera_mask[:, None, None]
        image_pixel_mask = base_pixel_mask[None].expand(
            video.shape[0], -1, -1, -1
        ) & timeline.frame_valid[:, None, None, None]
        sample_camera_mask = camera_mask[None].expand(
            video.shape[0], -1
        ) & timeline.frame_valid[:, None]

        task_index = int(current_row["task_index"])
        task = meta["tasks"][task_index]
        episode_timestamp_start = torch.as_tensor(
            rows[0]["timestamp"], dtype=torch.float64
        ).reshape(())
        anchor_timestamp = (
            episode_timestamp_start + anchor_rgb_index / self.rgb_rate_hz
        )
        frame_timestamps = (
            anchor_timestamp
            + self.timeline_layout.frame_offsets.to(torch.float64)
            / self.rgb_rate_hz
        )
        return {
            "video": video,
            "frame_offsets": self.timeline_layout.frame_offsets,
            "source_frame_indices": source_frame_indices,
            "frame_timestamps": frame_timestamps,
            "frame_valid": timeline.frame_valid,
            "camera_mask": sample_camera_mask,
            "image_pixel_mask": image_pixel_mask,
            "prompt": DEFAULT_PROMPT.format(task=task),
            "data_info": {
                "sample_mode": "video_only" if self.video_only else "canonical",
                "img_hw": torch.tensor(
                    [video.shape[-2], video.shape[-1]], dtype=torch.float32
                ),
                "aspect_ratio": torch.tensor(
                    video.shape[-1] / video.shape[-2], dtype=torch.float32
                ),
                "root_index": episode["root_index"],
                "episode_index": episode["episode_index"],
                "task_index": task_index,
                "source_fps": source_fps,
                "rgb_rate_hz": self.rgb_rate_hz,
                "action_steps_per_rgb_frame": self.action_steps_per_rgb_frame,
                "action_rate_hz": self.action_rate_hz,
                "anchor_timestamp": anchor_timestamp,
                "anchor_rgb_index": anchor_rgb_index,
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, anchor_rgb_index = self._locate_window(index)
        timeline = timeline_sample_indices(
            anchor_rgb_index,
            rgb_episode_length=episode["rgb_target_length"],
            action_episode_length=episode["action_target_length"],
            layout=self.timeline_layout,
        )
        meta = self._root_meta[episode["root_index"]]
        source_fps = meta["source_fps"]
        source_frame_indices = self._source_indices(
            timeline.frame_indices,
            source_fps,
            self.rgb_rate_hz,
        )
        anchor_source_index = self._source_indices(
            torch.tensor([anchor_rgb_index], dtype=torch.long),
            source_fps,
            self.rgb_rate_hz,
        )
        if self.video_only:
            requested = torch.unique(
                torch.cat([source_frame_indices, anchor_source_index]), sorted=True
            )
        else:
            anchor_action_index = torch.tensor(
                [anchor_rgb_index * self.action_steps_per_rgb_frame],
                dtype=torch.long,
            )
            state_target_indices = torch.cat(
                [anchor_action_index, timeline.action_indices.reshape(-1)]
            )
            state_lower, state_upper, state_fraction = self._state_interpolation_indices(
                state_target_indices,
                source_fps,
                self.action_rate_hz,
                episode["length"],
            )
            requested = torch.unique(
                torch.cat([source_frame_indices, state_lower, state_upper]), sorted=True
            )
        rows = meta["table_backend"].read_rows(
            episode,
            requested,
            meta["field_mask"],
            meta["field_mapping"],
            self.camera_keys,
            meta["camera_mask"],
        )
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
        current_row = rows[int(anchor_source_index.item())]
        sample = self._build_video_sample(
            episode,
            anchor_rgb_index,
            timeline,
            meta,
            source_frame_indices,
            rows,
            current_row,
        )
        if self.video_only:
            return sample

        lower_state = self._reshape_state_window(torch.tensor(
            [rows[int(frame)][CANONICAL_STATE_KEY] for frame in state_lower.tolist()],
            dtype=torch.float32,
        ))
        upper_state = self._reshape_state_window(torch.tensor(
            [rows[int(frame)][CANONICAL_STATE_KEY] for frame in state_upper.tolist()],
            dtype=torch.float32,
        ))
        tactile_values = (
            torch.as_tensor(current_row[CANONICAL_TACTILE_VALUES_KEY], dtype=torch.float32)
            if meta["tactile_mask"][0]
            else torch.zeros((4, 3, 25, 6), dtype=torch.float32)
        )
        tactile_dt = (
            torch.as_tensor(current_row[CANONICAL_TACTILE_DT_KEY], dtype=torch.float32)
            if meta["tactile_mask"][1]
            else torch.zeros((4, 3), dtype=torch.float32)
        )
        absolute_state_grid = interpolate_canonical_tcp(
            lower_state, upper_state, state_fraction
        )
        state_element_mask = meta["state_element_mask"].clone()
        action_element_mask = meta["action_element_mask"].clone()
        tactile_mask = meta["tactile_mask"].clone()
        tcp_transform = self._tcp_transform
        if tcp_transform is None:
            raise RuntimeError("Canonical sample construction requires global stats.")
        # Both outputs reuse this absolute interpolation grid; only Action is
        # converted into the fixed-anchor frame before normalization.
        action, action_feature_mask = tcp_transform.encode_action_targets(
            absolute_state_grid[0],
            absolute_state_grid[1:],
            action_element_mask,
        )
        action = action.reshape(
            self.timeline_layout.frame_offsets.numel(),
            self.action_steps_per_rgb_frame,
            WAM_FEATURE_DIM,
        )
        action = action * timeline.action_valid[..., None].to(
            action.dtype
        )
        state, state_feature_mask = tcp_transform.encode_state_targets(
            absolute_state_grid[1:], state_element_mask
        )
        state = state.reshape(
            self.timeline_layout.frame_offsets.numel(),
            self.action_steps_per_rgb_frame,
            WAM_FEATURE_DIM,
        )
        state = state * timeline.action_valid[..., None].to(
            state.dtype
        )

        action_timestamps = (
            sample["data_info"]["anchor_timestamp"]
            + self.timeline_layout.action_step_offsets.to(torch.float64)
            / self.action_rate_hz
        )
        sample.update({
            "state": state,
            "action": action,
            "action_step_offsets": self.timeline_layout.action_step_offsets,
            "action_timestamps": action_timestamps,
            "action_valid": timeline.action_valid,
            "state_feature_mask": state_feature_mask,
            "action_feature_mask": action_feature_mask,
            "observation_state_element_mask": state_element_mask,
            "action_element_mask": action_element_mask,
            CANONICAL_TACTILE_VALUES_KEY: tactile_values,
            CANONICAL_TACTILE_DT_KEY: tactile_dt,
            "tactile_field_mask": tactile_mask,
        })
        return sample
