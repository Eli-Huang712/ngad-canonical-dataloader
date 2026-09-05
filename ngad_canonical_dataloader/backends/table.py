"""Canonical row readers for Lance and LeRobot v3 Parquet containers."""

from __future__ import annotations

from bisect import bisect_right
import os
from pathlib import Path
from typing import Any

import torch


IDENTITY_KEYS = ("index", "episode_index", "frame_index", "task_index", "timestamp")
STATE_KEY = "observation.state"
TACTILE_KEYS = ("observation.tactile.values", "observation.tactile.dt")


def _lance_column(key: str) -> str:
    """Map canonical dotted feature names to Hy-compatible Lance columns."""
    return key.replace(".", "_")


def _physical_key(canonical_key: str, field_mapping: dict[str, str]) -> str:
    """Resolve the required canonical-to-physical mapping contract."""
    return field_mapping.get(canonical_key, canonical_key)


class LanceTableBackend:
    """Read canonical metadata and rows from one Lance/JPEG physical table."""

    def __init__(
        self,
        table_root: Path,
        lance_root: Path,
    ) -> None:
        self.table_root = table_root
        self.lance_root = lance_root
        self._handle: Any | None = None
        self._pid = os.getpid()

    def read_catalog(
        self,
        camera_keys: tuple[str, ...],
        camera_mask: torch.Tensor,
        field_mapping: dict[str, str],
    ) -> tuple[dict[int, str], list[dict[str, Any]]]:
        """Return tasks and episode offsets in the backend-neutral episode schema."""
        del camera_keys, camera_mask, field_mapping
        try:
            import pyarrow.dataset as ds
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("Lance canonical metadata requires pyarrow.") from error

        task_rows = pq.read_table(
            self.table_root / "meta" / "tasks.parquet"
        ).to_pylist()
        tasks = {int(row["task_index"]): str(row["task"]) for row in task_rows}
        episode_rows = ds.dataset(
            self.table_root / "meta" / "episodes", format="parquet"
        ).to_table(
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
                raise ValueError(
                    f"Invalid Lance episode offsets under {self.table_root}: {episode}."
                )
        return tasks, sorted(episodes, key=lambda row: row["dataset_from_index"])

    def _dataset(self):
        """Open one worker-local Lance handle."""
        pid = os.getpid()
        if pid != self._pid:
            self._handle = None
            self._pid = pid
        if self._handle is None:
            try:
                import lance
            except ImportError as error:
                raise ImportError("Lance canonical roots require the pylance package.") from error
            self._handle = lance.dataset(str(self.lance_root))
        return self._handle

    def read_rows(
        self,
        episode: dict[str, Any],
        relative_indices: torch.Tensor,
        field_mask: dict[str, bool],
        field_mapping: dict[str, str],
        camera_keys: tuple[str, ...],
        camera_mask: torch.Tensor,
    ) -> dict[int, dict[str, Any]]:
        """Read selected episode rows and normalize Lance column names to canonical keys."""
        try:
            import pyarrow as pa
        except ImportError as error:
            raise ImportError("Lance canonical roots require pyarrow.") from error

        episode_local_start = episode["dataset_from_index"]
        offsets = {episode_local_start}
        offsets.update(
            episode_local_start + int(relative_index)
            for relative_index in relative_indices.tolist()
        )
        offsets = sorted(offsets)
        value_keys = [STATE_KEY] if field_mask[STATE_KEY] else []
        value_keys.extend(key for key in TACTILE_KEYS if field_mask[key])
        value_keys.extend(
            camera
            for camera, available in zip(camera_keys, camera_mask.tolist())
            if available
        )
        canonical_keys = [*IDENTITY_KEYS, *value_keys]
        columns = list(
            dict.fromkeys(
                _lance_column(_physical_key(key, field_mapping))
                for key in canonical_keys
            )
        )
        rows = self._dataset().take(
            pa.array(offsets, type=pa.int64()), columns=columns
        ).to_pylist()

        by_relative_index: dict[int, dict[str, Any]] = {}
        for offset, physical_row in zip(offsets, rows):
            relative_index = offset - episode_local_start
            canonical_row = {
                key: physical_row[_lance_column(_physical_key(key, field_mapping))]
                for key in canonical_keys
            }
            # Episode metadata describes the compacted physical Lance layout. HY
            # conversion may trim invalid boundary rows while retaining the
            # pre-filter global ``index`` as source provenance, so that stored
            # value can be sparse and must not be used as a physical row address.
            if (
                int(canonical_row["episode_index"]) != episode["episode_index"]
                or int(canonical_row["frame_index"]) != relative_index
            ):
                raise RuntimeError(
                    f"Canonical Lance episode/frame identity mismatch at offset {offset}."
                )
            by_relative_index[relative_index] = canonical_row
        return by_relative_index


class ParquetTableBackend:
    """Read canonical metadata and rows from one LeRobot v3 Parquet root."""

    def __init__(
        self,
        root: Path,
        info: dict[str, Any],
        row_addressing: str = "global_contiguous",
    ) -> None:
        if row_addressing not in {"global_contiguous", "episode_indexed"}:
            raise ValueError(
                "Parquet row_addressing must be 'global_contiguous' or "
                "'episode_indexed'."
            )
        self.root = root
        self.info = info
        self.row_addressing = row_addressing
        self._data_file_starts: dict[tuple[int, int], int] = {}
        self._handles: dict[tuple[int, int], Any] = {}
        self._row_group_ends: dict[tuple[int, int], list[int]] = {}
        self._episode_row_ranges: dict[
            tuple[int, int], dict[int, tuple[int, int]]
        ] = {}
        self._pid = os.getpid()

    def read_catalog(
        self,
        camera_keys: tuple[str, ...],
        camera_mask: torch.Tensor,
        field_mapping: dict[str, str],
    ) -> tuple[dict[int, str], list[dict[str, Any]]]:
        """Return tasks, episodes, Parquet offsets and per-camera video ranges."""
        try:
            import pyarrow.dataset as ds
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("LeRobot v3 canonical metadata requires pyarrow.") from error

        task_rows = pq.read_table(self.root / "meta" / "tasks.parquet").to_pylist()
        tasks = {int(row["task_index"]): str(row["task"]) for row in task_rows}
        camera_pairs = [
            (camera, _physical_key(camera, field_mapping))
            for camera, available in zip(camera_keys, camera_mask.tolist())
            if available
        ]
        video_columns = [
            f"videos/{physical_camera}/{field}"
            for _, physical_camera in camera_pairs
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
            ds.dataset(self.root / "meta" / "episodes", format="parquet")
            .to_table(columns=columns)
            .to_pylist()
        )
        episodes: list[dict[str, Any]] = []
        for row in episode_rows:
            episode_index = int(row["episode_index"])
            task_values = row["tasks"]
            if not isinstance(task_values, list) or not task_values or not all(task_values):
                raise ValueError(
                    f"LeRobot v3 episode {episode_index} under {self.root} must list its tasks."
                )
            episode = {
                "episode_index": episode_index,
                "tasks": tuple(str(task) for task in task_values),
                "length": int(row["length"]),
                "dataset_from_index": int(row["dataset_from_index"]),
                "dataset_to_index": int(row["dataset_to_index"]),
                "data_chunk_index": int(row["data/chunk_index"]),
                "data_file_index": int(row["data/file_index"]),
                "videos": {
                    camera: {
                        "physical_key": physical_camera,
                        **{
                            field: row[f"videos/{physical_camera}/{field}"]
                            for field in (
                                "chunk_index",
                                "file_index",
                                "from_timestamp",
                                "to_timestamp",
                            )
                        },
                    }
                    for camera, physical_camera in camera_pairs
                },
            }
            if episode["dataset_to_index"] - episode["dataset_from_index"] != episode["length"]:
                raise ValueError(
                    f"Invalid LeRobot v3 episode offsets under {self.root}: {episode}."
                )
            for camera, video in episode["videos"].items():
                video["chunk_index"] = int(video["chunk_index"])
                video["file_index"] = int(video["file_index"])
                video["from_timestamp"] = float(video["from_timestamp"])
                video["to_timestamp"] = float(video["to_timestamp"])
                if video["to_timestamp"] <= video["from_timestamp"]:
                    raise ValueError(
                        f"Invalid LeRobot v3 video range for {camera} in episode {episode_index}."
                    )
            file_key = (episode["data_chunk_index"], episode["data_file_index"])
            self._data_file_starts[file_key] = min(
                self._data_file_starts.get(file_key, episode["dataset_from_index"]),
                episode["dataset_from_index"],
            )
            episodes.append(episode)
        if self.row_addressing == "episode_indexed":
            episodes_by_file: dict[tuple[int, int], list[dict[str, Any]]] = {}
            for episode in episodes:
                file_key = (
                    episode["data_chunk_index"],
                    episode["data_file_index"],
                )
                episodes_by_file.setdefault(file_key, []).append(episode)
            self._episode_row_ranges = {}
            for file_key, file_episodes in episodes_by_file.items():
                local_start = 0
                ranges: dict[int, tuple[int, int]] = {}
                for episode in sorted(
                    file_episodes,
                    key=lambda record: record["dataset_from_index"],
                ):
                    episode_index = episode["episode_index"]
                    if episode_index in ranges:
                        raise ValueError(
                            f"Duplicate episode {episode_index} in Parquet shard "
                            f"{file_key}."
                        )
                    local_end = local_start + episode["length"]
                    ranges[episode_index] = (local_start, local_end)
                    local_start = local_end
                self._episode_row_ranges[file_key] = ranges
        return tasks, sorted(episodes, key=lambda record: record["dataset_from_index"])

    def _data_file(self, episode: dict[str, Any]):
        """Open the worker-local Parquet shard containing one episode."""
        pid = os.getpid()
        if pid != self._pid:
            self._handles = {}
            self._row_group_ends = {}
            self._pid = pid
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("LeRobot canonical roots require pyarrow.") from error

        cache_key = (episode["data_chunk_index"], episode["data_file_index"])
        if cache_key in self._handles:
            return self._handles[cache_key], self._row_group_ends[cache_key]
        path = self.root / str(self.info["data_path"]).format(
            chunk_index=cache_key[0],
            file_index=cache_key[1],
        )
        parquet_file = pq.ParquetFile(path)
        row_group_ends: list[int] = []
        row_count = 0
        for row_group_index in range(parquet_file.num_row_groups):
            row_count += parquet_file.metadata.row_group(row_group_index).num_rows
            row_group_ends.append(row_count)
        self._handles[cache_key] = parquet_file
        self._row_group_ends[cache_key] = row_group_ends
        return parquet_file, row_group_ends

    def _episode_rows(
        self,
        file_key: tuple[int, int],
        parquet_file: Any,
        field_mapping: dict[str, str],
    ) -> dict[int, tuple[int, int]]:
        """Index physical episode blocks in one non-contiguous Parquet shard."""
        if file_key in self._episode_row_ranges:
            return self._episode_row_ranges[file_key]
        episode_column = _physical_key("episode_index", field_mapping)
        frame_column = _physical_key("frame_index", field_mapping)
        identity = parquet_file.read(columns=[episode_column, frame_column]).to_pydict()
        episode_values = identity[episode_column]
        frame_values = identity[frame_column]
        if not episode_values:
            raise ValueError(f"Canonical Parquet shard {file_key} contains no rows.")

        ranges: dict[int, tuple[int, int]] = {}
        block_start = 0
        current_episode = int(episode_values[0])
        for local_row, (episode_value, frame_value) in enumerate(
            zip(episode_values, frame_values)
        ):
            episode_index = int(episode_value)
            frame_index = int(frame_value)
            if episode_index != current_episode:
                if current_episode in ranges:
                    raise ValueError(
                        f"Episode {current_episode} is not contiguous in Parquet shard "
                        f"{file_key}."
                    )
                ranges[current_episode] = (block_start, local_row)
                block_start = local_row
                current_episode = episode_index
            expected_frame = local_row - block_start
            if frame_index != expected_frame:
                raise ValueError(
                    f"Episode {episode_index} has frame_index {frame_index} at local "
                    f"row {local_row}, expected {expected_frame}, in Parquet shard "
                    f"{file_key}."
                )
        if current_episode in ranges:
            raise ValueError(
                f"Episode {current_episode} is not contiguous in Parquet shard {file_key}."
            )
        ranges[current_episode] = (block_start, len(episode_values))
        self._episode_row_ranges[file_key] = ranges
        return ranges

    def read_rows(
        self,
        episode: dict[str, Any],
        relative_indices: torch.Tensor,
        field_mask: dict[str, bool],
        field_mapping: dict[str, str],
        camera_keys: tuple[str, ...],
        camera_mask: torch.Tensor,
    ) -> dict[int, dict[str, Any]]:
        """Read requested episode rows from a shared LeRobot v3 Parquet shard."""
        del camera_keys, camera_mask
        try:
            import pyarrow as pa
        except ImportError as error:
            raise ImportError("LeRobot canonical roots require pyarrow.") from error

        requested = {0}
        requested.update(int(index) for index in relative_indices.tolist())
        file_key = (episode["data_chunk_index"], episode["data_file_index"])
        parquet_file, row_group_ends = self._data_file(episode)
        if self.row_addressing == "episode_indexed":
            episode_rows = self._episode_rows(file_key, parquet_file, field_mapping)
            try:
                episode_start, episode_end = episode_rows[episode["episode_index"]]
            except KeyError as error:
                raise IndexError(
                    f"Episode {episode['episode_index']} is absent from Parquet shard "
                    f"{file_key}."
                ) from error
            if episode_end - episode_start != episode["length"]:
                raise ValueError(
                    f"Episode {episode['episode_index']} metadata length "
                    f"{episode['length']} does not match its {episode_end - episode_start} "
                    f"physical rows in Parquet shard {file_key}."
                )
        else:
            file_start = self._data_file_starts[file_key]
            episode_start = episode["dataset_from_index"] - file_start
        local_rows = {
            episode_start + relative_index: relative_index
            for relative_index in requested
        }
        if not local_rows or min(local_rows) < 0 or max(local_rows) >= row_group_ends[-1]:
            raise IndexError(f"LeRobot v3 episode offsets exceed their data shard: {episode}.")

        canonical_columns = [*IDENTITY_KEYS]
        if field_mask[STATE_KEY]:
            canonical_columns.append(STATE_KEY)
        canonical_columns.extend(key for key in TACTILE_KEYS if field_mask[key])
        columns = list(
            dict.fromkeys(
                _physical_key(key, field_mapping) for key in canonical_columns
            )
        )
        by_row_group: dict[int, list[int]] = {}
        for local_row in sorted(local_rows):
            row_group_index = bisect_right(row_group_ends, local_row)
            by_row_group.setdefault(row_group_index, []).append(local_row)

        rows_by_relative_index: dict[int, dict[str, Any]] = {}
        for row_group_index, group_rows in by_row_group.items():
            group_start = 0 if row_group_index == 0 else row_group_ends[row_group_index - 1]
            table = parquet_file.read_row_group(row_group_index, columns=columns)
            table = table.take(
                pa.array([row - group_start for row in group_rows], type=pa.int64())
            )
            for local_row, physical_row in zip(group_rows, table.to_pylist()):
                relative_index = local_rows[local_row]
                global_index = episode["dataset_from_index"] + relative_index
                row = {
                    key: physical_row[_physical_key(key, field_mapping)]
                    for key in canonical_columns
                }
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
