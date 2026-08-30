"""Canonical camera readers for JPEG payloads and H.264 video shards."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch


class JpegImageBackend:
    """Decode canonical JPEG payloads stored in table rows."""

    def read_camera(
        self,
        rows: dict[int, dict[str, Any]],
        episode: dict[str, Any],
        camera: str,
        source_indices: torch.Tensor,
        source_fps: float,
    ) -> torch.Tensor:
        """Return uint8 RGB frames in [T,3,H,W] without model preprocessing."""
        del episode, source_fps
        frames = []
        for index in source_indices.tolist():
            payload = rows[int(index)][camera]
            with Image.open(BytesIO(payload)) as image:
                image.load()
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if rgb.shape != (256, 256, 3):
                raise ValueError(f"Canonical JPEG {camera} frame {index} has shape {rgb.shape}.")
            frames.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1))
        return torch.stack(frames)


class H264ImageBackend:
    """Decode episode-relative RGB frames from LeRobot v3 H.264 shards."""

    def __init__(self, root: Path, info: dict[str, Any]) -> None:
        self.root = root
        self.info = info

    @staticmethod
    def _frame_rate(stream: Any, fallback: float) -> float:
        """Use the nominal fixed rate instead of duration-skewed MP4 average rate."""
        return float(stream.base_rate or stream.average_rate or fallback)

    def read_camera(
        self,
        rows: dict[int, dict[str, Any]],
        episode: dict[str, Any],
        camera: str,
        source_indices: torch.Tensor,
        source_fps: float,
    ) -> torch.Tensor:
        """Return uint8 RGB frames in [T,3,H,W] without model preprocessing."""
        del rows
        try:
            import av
        except ImportError as error:
            raise ImportError("LeRobot canonical roots require PyAV.") from error

        video_meta = episode["videos"][camera]
        video_path = self.root / str(self.info["video_path"]).format(
            video_key=video_meta["physical_key"],
            chunk_index=video_meta["chunk_index"],
            file_index=video_meta["file_index"],
        )
        requested_relative = [int(index) for index in source_indices.tolist()]
        decoded: dict[int, torch.Tensor] = {}
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            frame_rate = self._frame_rate(stream, float(self.info["fps"]))
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
        return torch.stack([decoded[index] for index in requested]).permute(0, 3, 1, 2)
