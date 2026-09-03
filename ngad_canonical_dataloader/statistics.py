"""State-only offline statistics for the canonical TCP z-score ABI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ngad_canonical_dataloader.action import (
    dual_arm_tcp_target_relative_to_anchor,
)
from ngad_canonical_dataloader.config import load_dataset_config
from ngad_canonical_dataloader.datasets.canonical import (
    CANONICAL_STATE_KEY,
    NGAD_CANONICAL_SCHEMA,
    NGADCanonicalDataset,
    interpolate_canonical_tcp,
)


class RunningMoments:
    """Featurewise parallel-Welford accumulator with independent masks."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.count = torch.zeros(shape, dtype=torch.float64)
        self.mean = torch.zeros(shape, dtype=torch.float64)
        self.m2 = torch.zeros(shape, dtype=torch.float64)

    def update(self, values: torch.Tensor, valid: torch.Tensor) -> None:
        values = torch.as_tensor(values, dtype=torch.float64)
        valid = torch.as_tensor(valid, dtype=torch.bool)
        if values.shape != valid.shape or values.shape[-2:] != self.mean.shape:
            raise ValueError(
                f"Moment values/mask must match and end with {tuple(self.mean.shape)}, "
                f"got {tuple(values.shape)} and {tuple(valid.shape)}."
            )
        reduce_dims = tuple(range(values.ndim - self.mean.ndim))
        batch_count = valid.sum(dim=reduce_dims, dtype=torch.float64)
        safe_count = batch_count.clamp_min(1.0)
        batch_sum = torch.where(valid, values, 0.0).sum(dim=reduce_dims)
        batch_mean = batch_sum / safe_count
        centered = values - batch_mean
        batch_m2 = torch.where(valid, centered.square(), 0.0).sum(dim=reduce_dims)

        total = self.count + batch_count
        delta = batch_mean - self.mean
        weight = torch.where(total > 0, batch_count / total.clamp_min(1.0), 0.0)
        self.mean = self.mean + delta * weight
        self.m2 = self.m2 + batch_m2 + torch.where(
            total > 0,
            delta.square() * self.count * batch_count / total.clamp_min(1.0),
            0.0,
        )
        self.count = total

    def result(
        self,
        std_floor: float,
        required_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        required = (
            torch.ones_like(self.count, dtype=torch.bool)
            if required_mask is None
            else torch.as_tensor(required_mask, dtype=torch.bool)
        )
        if required.shape != self.count.shape:
            raise ValueError(
                f"required_mask must have shape {tuple(self.count.shape)}, "
                f"got {tuple(required.shape)}."
            )
        if torch.any(required & (self.count <= 0)):
            raise RuntimeError(
                f"No observations for required TCP features: {self.count.tolist()}."
            )
        variance = self.m2 / self.count
        raw_std = torch.where(
            required, variance.clamp_min(0.0).sqrt(), torch.zeros_like(variance)
        )
        safe_std = torch.where(
            required & (raw_std >= float(std_floor)), raw_std, torch.ones_like(raw_std)
        )
        return {
            "count": self.count,
            "mean": torch.where(required, self.mean, torch.zeros_like(self.mean)),
            "raw_std": raw_std,
            "std": safe_std,
            "std_floor_applied": required & (raw_std < float(std_floor)),
        }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _source_gripper_endpoints(config_path: Path) -> tuple[list[float], list[float]]:
    config = load_dataset_config(config_path)
    if config.normalization_stats_path is None:
        raise ValueError("Statistics require a source normalization JSON with gripper endpoints.")
    source = _read_json(Path(config.normalization_stats_path).expanduser().resolve())
    try:
        open_value = [float(value) for value in source["gripper_open_value"]]
        closed_value = [float(value) for value in source["gripper_closed_value"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Source normalization must provide numeric gripper endpoints.") from error
    if len(open_value) != 2 or len(closed_value) != 2:
        raise ValueError("Source gripper endpoints must both have shape [2].")
    return open_value, closed_value


def build_state_only_dataset(config_path: str | Path) -> NGADCanonicalDataset:
    """Build catalog/backends without loading old normalization or decoding images."""
    config = load_dataset_config(config_path)
    kwargs = config.to_dataset_kwargs()
    kwargs["normalization_stats_path"] = None
    kwargs["max_samples"] = None
    return NGADCanonicalDataset(**kwargs)


def compute_zscore_statistics(
    dataset: NGADCanonicalDataset,
    *,
    std_floor: float = 1.0e-5,
    anchor_batch_size: int = 2048,
    episode_start: int = 0,
    episode_stop: int | None = None,
    max_episodes: int | None = None,
    max_anchors: int | None = None,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Compute exact sample-distribution moments from absolute state only."""
    if std_floor <= 0 or anchor_batch_size <= 0:
        raise ValueError("std_floor and anchor_batch_size must be positive.")
    if episode_start < 0:
        raise ValueError("episode_start must be non-negative.")
    resolved_stop = len(dataset._episodes) if episode_stop is None else episode_stop
    if resolved_stop <= episode_start or resolved_stop > len(dataset._episodes):
        raise ValueError(
            "episode_stop must be greater than episode_start and no larger than "
            "the number of Dataset episodes."
        )
    if max_episodes is not None:
        resolved_stop = min(resolved_stop, episode_start + max_episodes)
    state_moments = RunningMoments((2, 9))
    action_moments = RunningMoments((2, 9))
    action_offsets = dataset.timeline_layout.action_step_offsets.reshape(-1)
    total_anchors = 0
    total_valid_targets = 0
    processed_episodes = 0
    required_state_features = torch.zeros((2, 9), dtype=torch.bool)

    for episode_number, episode in enumerate(
        dataset._episodes[episode_start:resolved_stop],
        start=episode_start,
    ):
        if max_anchors is not None and total_anchors >= max_anchors:
            break
        meta = dataset._root_meta[episode["root_index"]]
        state_feature_mask = meta["state_element_mask"].reshape(2, 10)[..., :9]
        required_state_features |= state_feature_mask

        source_indices = torch.arange(episode["length"], dtype=torch.long)
        state_only_mask = {key: False for key in meta["field_mask"]}
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            state_only_mask[key] = True
        state_only_mask[CANONICAL_STATE_KEY] = True
        rows = meta["table_backend"].read_rows(
            episode,
            source_indices,
            state_only_mask,
            meta["field_mapping"],
            dataset.camera_keys,
            torch.zeros_like(meta["camera_mask"]),
        )
        source_state = dataset._reshape_state_window(torch.tensor(
            [rows[index][CANONICAL_STATE_KEY] for index in range(episode["length"])],
            dtype=torch.float32,
        ))
        target_grid = torch.arange(episode["action_target_length"], dtype=torch.long)
        lower, upper, fraction = dataset._state_interpolation_indices(
            target_grid,
            meta["source_fps"],
            dataset.action_rate_hz,
            episode["length"],
        )
        absolute_grid = interpolate_canonical_tcp(
            source_state[lower], source_state[upper], fraction
        )

        episode_anchor_count = int(episode["rgb_target_length"])
        if max_anchors is not None:
            episode_anchor_count = min(
                episode_anchor_count, max_anchors - total_anchors
            )
        for start in range(0, episode_anchor_count, anchor_batch_size):
            stop = min(start + anchor_batch_size, episode_anchor_count)
            anchor_rgb = torch.arange(start, stop, dtype=torch.long)
            anchor_action = anchor_rgb * dataset.action_steps_per_rgb_frame
            raw_targets = anchor_action[:, None] + action_offsets[None, :]
            valid = (raw_targets >= 0) & (
                raw_targets < episode["action_target_length"]
            )
            target_indices = raw_targets.clamp(
                0, episode["action_target_length"] - 1
            )
            targets = absolute_grid[target_indices]
            anchors = absolute_grid[anchor_action]
            relative = dual_arm_tcp_target_relative_to_anchor(
                anchors.reshape(stop - start, 1, 20),
                targets.reshape(stop - start, action_offsets.numel(), 20),
            ).reshape(stop - start, action_offsets.numel(), 2, 10)
            feature_valid = valid[..., None, None].expand(-1, -1, 2, 9)
            feature_valid = feature_valid & state_feature_mask
            state_moments.update(targets[..., :9], feature_valid)
            action_moments.update(relative[..., :9], feature_valid)
            total_valid_targets += int(valid.sum())

        total_anchors += episode_anchor_count
        processed_episodes += 1
        if progress_every > 0 and (
            processed_episodes % progress_every == 0
            or processed_episodes == 1
        ):
            print(
                f"episodes={processed_episodes} anchors={total_anchors} "
                f"valid_targets={total_valid_targets}",
                flush=True,
            )

    state = state_moments.result(std_floor, required_state_features)
    action = action_moments.result(std_floor, required_state_features)
    return {
        "schema_version": NGAD_CANONICAL_SCHEMA,
        "state_tcp_mean": state["mean"].tolist(),
        "state_tcp_std": state["std"].tolist(),
        "action_tcp_mean": action["mean"].tolist(),
        "action_tcp_std": action["std"].tolist(),
        "statistics": {
            "method": "population_zscore_parallel_welford",
            "scope": "all valid canonical timeline state/action targets",
            "std_floor": float(std_floor),
            "episode_start": int(episode_start),
            "episode_stop": int(episode_start + processed_episodes),
            "episodes": processed_episodes,
            "anchors": total_anchors,
            "valid_targets": total_valid_targets,
            "state_count": state["count"].to(torch.int64).tolist(),
            "action_count": action["count"].to(torch.int64).tolist(),
            "state_raw_std": state["raw_std"].tolist(),
            "action_raw_std": action["raw_std"].tolist(),
            "state_std_floor_applied": state["std_floor_applied"].tolist(),
            "action_std_floor_applied": action["std_floor_applied"].tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute canonical state/relative-action z-score statistics from state only."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--std-floor", type=float, default=1.0e-5)
    parser.add_argument("--anchor-batch-size", type=int, default=2048)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-stop", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-anchors", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    open_value, closed_value = _source_gripper_endpoints(args.config)
    dataset = build_state_only_dataset(args.config)
    result = compute_zscore_statistics(
        dataset,
        std_floor=args.std_floor,
        anchor_batch_size=args.anchor_batch_size,
        episode_start=args.episode_start,
        episode_stop=args.episode_stop,
        max_episodes=args.max_episodes,
        max_anchors=args.max_anchors,
        progress_every=args.progress_every,
    )
    result["gripper_open_value"] = open_value
    result["gripper_closed_value"] = closed_value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote={args.output}", flush=True)


if __name__ == "__main__":
    main()
