#!/usr/bin/env python3
"""Write one Dataset episode to an H100-side temporary Rerun recording."""

from __future__ import annotations

import argparse
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch

from ngad_canonical_dataloader import build_dataset_from_yaml


TACTILE_VALUES_KEY = "observation.tactile.values"
TACTILE_DT_KEY = "observation.tactile.dt"
TACTILE_SENSOR_NAMES = (
    "left_finger_0",
    "left_finger_1",
    "right_finger_0",
    "right_finger_1",
)
TCP_FEATURE_NAMES = (
    "x",
    "y",
    "z",
    "rot6_0",
    "rot6_1",
    "rot6_2",
    "rot6_3",
    "rot6_4",
    "rot6_5",
    "gripper",
)


def _entity_name(canonical_camera_key: str) -> str:
    return canonical_camera_key.rsplit(".", 1)[-1]


def _display_rgb(frame: torch.Tensor) -> np.ndarray:
    """Invert only the Dataset's display normalization, not its semantics."""
    image = ((frame.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
    return image.round().to(torch.uint8).permute(1, 2, 0).numpy()


def _force_activation(force: np.ndarray) -> np.ndarray:
    """Map non-negative force magnitudes to a bounded visual intensity."""
    force = np.maximum(force, 0.0)
    return force / (force + 1.0)


def _force_colors(normal_force: np.ndarray) -> np.ndarray:
    activation = _force_activation(normal_force)
    red = np.round(255.0 * activation)
    green = np.round(80.0 + 100.0 * activation)
    blue = np.round(255.0 * (1.0 - activation))
    alpha = np.full_like(red, 255.0)
    return np.stack([red, green, blue, alpha], axis=-1).astype(np.uint8)


def _bounded_shear_vectors(force_xy: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(force_xy, axis=-1, keepdims=True)
    direction = np.divide(
        force_xy,
        magnitude,
        out=np.zeros_like(force_xy),
        where=magnitude > 0.0,
    )
    return direction * (2.0 * _force_activation(magnitude))


def _build_blueprint(camera_keys: tuple[str, ...]) -> rrb.Blueprint:
    camera_views = [
        rrb.Spatial2DView(
            origin=f"/cameras/{_entity_name(camera)}",
            name=_entity_name(camera),
        )
        for camera in camera_keys
    ]
    tactile_views = [
        rrb.Spatial2DView(
            origin=f"/tactile/{sensor}",
            name=sensor,
        )
        for sensor in TACTILE_SENSOR_NAMES
    ]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Grid(*camera_views, grid_columns=3, name="Six camera views"),
                rrb.Vertical(
                    rrb.TextDocumentView(
                        origin="/episode/prompt",
                        name="Prompt",
                    ),
                    rrb.TextDocumentView(
                        origin="/episode/metadata",
                        name="Metadata",
                    ),
                ),
                column_shares=[3, 1],
            ),
            rrb.Horizontal(
                rrb.Grid(*tactile_views, grid_columns=2, name="Tactile taxels"),
                rrb.Vertical(
                    rrb.TimeSeriesView(
                        origin="/tactile_metrics",
                        name="Tactile metrics",
                    ),
                    rrb.TimeSeriesView(origin="/state", name="State TCP128"),
                    rrb.TimeSeriesView(origin="/action", name="Action TCP128"),
                ),
                column_shares=[1, 1],
            ),
            rrb.TextLogView(
                origin="/annotations",
                name="Task annotations",
            ),
            row_shares=[4, 2, 1],
        ),
        collapse_panels=True,
    )


def _log_masks(sample: dict[str, Any], camera_keys: tuple[str, ...]) -> None:
    for camera_index, camera in enumerate(camera_keys):
        mask = sample["image_pixel_mask"][0, camera_index]
        rr.log(
            f"/pixel_masks/{_entity_name(camera)}",
            rr.Image(mask.detach().cpu().to(torch.uint8).numpy() * 255),
            static=True,
        )


def _log_tcp_series(sample: dict[str, Any], episode_start: float) -> None:
    if "state" not in sample or "action" not in sample:
        return
    timestamps = sample["action_timestamps"][0].detach().cpu().numpy()
    valid = sample["action_valid"][0].detach().cpu().numpy()
    state = sample["state"][0].detach().cpu().numpy()
    action = sample["action"][0].detach().cpu().numpy()
    state_mask = sample["state_feature_mask"].detach().cpu().numpy()
    action_mask = sample["action_feature_mask"].detach().cpu().numpy()
    for step, timestamp in enumerate(timestamps):
        rr.set_time("episode_time", duration=float(timestamp) - episode_start)
        for arm_index, arm_name in enumerate(("left", "right")):
            for feature_index, feature_name in enumerate(TCP_FEATURE_NAMES):
                packed_index = arm_index * len(TCP_FEATURE_NAMES) + feature_index
                if state_mask[packed_index]:
                    rr.log(
                        f"/state/{arm_name}/{feature_name}",
                        rr.Scalars(float(state[step, packed_index]) if valid[step] else np.nan),
                    )
                if action_mask[packed_index]:
                    rr.log(
                        f"/action/{arm_name}/{feature_name}",
                        rr.Scalars(float(action[step, packed_index]) if valid[step] else np.nan),
                    )


def _log_tactile(sample: dict[str, Any], episode_start: float) -> None:
    if TACTILE_VALUES_KEY not in sample:
        return
    field_mask = sample["tactile_field_mask"].detach().cpu().numpy()
    if not bool(np.all(field_mask)):
        rr.log(
            "/episode/tactile_status",
            rr.TextDocument("Tactile fields are unavailable for this dataset."),
            static=True,
        )
        return

    values = sample[TACTILE_VALUES_KEY][0].detach().cpu().numpy()
    tactile_dt = sample[TACTILE_DT_KEY][0].detach().cpu().numpy()
    valid = sample["tactile_valid"][0].detach().cpu().numpy()
    frame_timestamp = float(sample["frame_timestamps"][0])
    rgb_rate_hz = float(sample["data_info"]["rgb_rate_hz"])
    tactile_rate_hz = float(sample["data_info"]["tactile_rate_hz"])

    for sensor_index, sensor_name in enumerate(TACTILE_SENSOR_NAMES):
        for slot in range(values.shape[1]):
            if not valid[sensor_index, slot]:
                slot_time = (
                    frame_timestamp
                    - 1.0 / rgb_rate_hz
                    + (slot + 0.5) / tactile_rate_hz
                )
                rr.set_time("episode_time", duration=slot_time - episode_start)
                rr.log(f"/tactile/{sensor_name}", rr.Clear(recursive=True))
                continue

            event_time = frame_timestamp + float(tactile_dt[sensor_index, slot])
            rr.set_time("episode_time", duration=event_time - episode_start)
            taxels = values[sensor_index, slot]
            position_xy = taxels[:, :2]
            force = taxels[:, 3:6]
            normal_force = np.maximum(force[:, 2], 0.0)
            force_norm = np.linalg.norm(force, axis=-1)
            activation = _force_activation(normal_force)
            rr.log(
                f"/tactile/{sensor_name}/taxels",
                rr.Points2D(
                    position_xy,
                    radii=0.25 + 0.9 * activation,
                    colors=_force_colors(normal_force),
                ),
            )
            active = np.linalg.norm(force[:, :2], axis=-1) > 0.05
            shear_path = f"/tactile/{sensor_name}/shear"
            if np.any(active):
                arrow_colors = np.tile(
                    np.array([[255, 220, 0, 255]], dtype=np.uint8),
                    (int(active.sum()), 1),
                )
                rr.log(
                    shear_path,
                    rr.Arrows2D(
                        origins=position_xy[active],
                        vectors=_bounded_shear_vectors(force[active, :2]),
                        colors=arrow_colors,
                    ),
                )
            else:
                rr.log(shear_path, rr.Clear(recursive=False))
            rr.log(
                f"/tactile_metrics/{sensor_name}/total_normal_force",
                rr.Scalars(float(normal_force.sum())),
            )
            rr.log(
                f"/tactile_metrics/{sensor_name}/total_shear",
                rr.Scalars(float(np.linalg.norm(force[:, :2], axis=-1).sum())),
            )
            rr.log(
                f"/tactile_metrics/{sensor_name}/max_force_norm",
                rr.Scalars(float(force_norm.max())),
            )
            rr.log(
                f"/tactile_metrics/{sensor_name}/active_taxels",
                rr.Scalars(int(np.count_nonzero(force_norm > 0.05))),
            )


def log_episode(dataset_config: Path, episode_index: int, output: Path) -> None:
    """Load one complete episode through Dataset samples and save it as RRD."""
    dataset = build_dataset_from_yaml(dataset_config)
    frame_offsets = dataset.timeline_layout.frame_offsets.tolist()
    if frame_offsets != [0]:
        raise ValueError(
            "Episode visualization requires a Dataset YAML with frame_ranges: [[0, 0]]."
        )

    samples = dataset.iter_episode_samples(episode_index)
    first = next(samples, None)
    if first is None:
        raise RuntimeError(f"Episode {episode_index} has no Dataset samples.")
    camera_keys = tuple(dataset.camera_keys)
    blueprint = _build_blueprint(camera_keys)
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.init(
        "ngad-canonical-episode",
        recording_id=f"episode-{episode_index}",
        spawn=False,
        default_blueprint=blueprint,
    )
    rr.save(output, default_blueprint=blueprint)
    _log_masks(first, camera_keys)

    episode_start = float(first["data_info"]["anchor_timestamp"])
    previous_prompt: str | None = None
    for sample in chain((first,), samples):
        frame_index = int(sample["data_info"]["anchor_rgb_index"])
        timestamp = float(sample["data_info"]["anchor_timestamp"])
        rr.set_time("frame", sequence=frame_index)
        rr.set_time("episode_time", duration=timestamp - episode_start)

        for camera_index, camera in enumerate(camera_keys):
            rr.log(
                f"/cameras/{_entity_name(camera)}",
                rr.Image(_display_rgb(sample["video"][0, camera_index])),
            )

        prompt = str(sample["prompt"])
        if prompt != previous_prompt:
            rr.log(
                "/episode/prompt",
                rr.TextDocument(prompt, media_type="text/markdown"),
            )
            rr.log(
                "/annotations/task_transitions",
                rr.TextLog(
                    f"task_index={int(sample['data_info']['task_index'])}: {prompt}",
                    level="INFO",
                ),
            )
            previous_prompt = prompt
        camera_mask = sample["camera_mask"][0].detach().cpu().tolist()
        info = sample["data_info"]
        metadata = (
            f"### Episode {int(info['episode_index'])}\n\n"
            f"- frame: `{frame_index}`\n"
            f"- task_index: `{int(info['task_index'])}`\n"
            f"- timestamp: `{timestamp:.6f}`\n"
            f"- camera_mask: `{camera_mask}`\n"
            f"- sample_mode: `{info['sample_mode']}`"
        )
        rr.log(
            "/episode/metadata",
            rr.TextDocument(metadata, media_type="text/markdown"),
        )
        _log_tcp_series(sample, episode_start)
        _log_tactile(sample, episode_start)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one full canonical Dataset episode as a Rerun RRD."
    )
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.suffix != ".rrd":
        parser.error("--output must end with .rrd")
    return args


def main() -> None:
    args = _parse_args()
    log_episode(args.dataset_config, args.episode_index, args.output)


if __name__ == "__main__":
    main()
