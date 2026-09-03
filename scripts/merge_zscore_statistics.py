"""Merge exact canonical z-score statistic shards with parallel Welford math."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA_VERSION = "ngad_canonical_tcp_v2"
MOMENT_SHAPE = (2, 9)
MOMENT_NAMES = ("state", "action")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read statistics JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _finite_float(value: object, label: str, path: Path) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} in {path} must be numeric, not bool.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} in {path} must be numeric.") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} in {path} must be finite, got {result}.")
    return result


def _nonnegative_int(value: object, label: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} in {path} must be an integer.")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{label} in {path} must be a nonnegative integer.")
    return result


def _matrix(
    value: object,
    label: str,
    path: Path,
    *,
    counts: bool = False,
) -> list[list[float]] | list[list[int]]:
    if not isinstance(value, list) or len(value) != MOMENT_SHAPE[0]:
        raise ValueError(f"{label} in {path} must have shape {MOMENT_SHAPE}.")
    rows: list[list[float]] | list[list[int]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != MOMENT_SHAPE[1]:
            raise ValueError(f"{label} in {path} must have shape {MOMENT_SHAPE}.")
        if counts:
            parsed = [
                _nonnegative_int(item, f"{label}[{row_index}][{column}]", path)
                for column, item in enumerate(row)
            ]
        else:
            parsed = [
                _finite_float(item, f"{label}[{row_index}][{column}]", path)
                for column, item in enumerate(row)
            ]
        rows.append(parsed)
    return rows


def _endpoint(value: object, label: str, path: Path) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} in {path} must have shape [2].")
    return (
        _finite_float(value[0], f"{label}[0]", path),
        _finite_float(value[1], f"{label}[1]", path),
    )


def _validated_shard(path: Path) -> dict[str, Any]:
    shard = _read_json(path)
    if shard.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Expected schema_version={SCHEMA_VERSION!r} in {path}, "
            f"got {shard.get('schema_version')!r}."
        )
    statistics = shard.get("statistics")
    if not isinstance(statistics, dict):
        raise TypeError(f"statistics in {path} must be an object.")
    if statistics.get("method") != "population_zscore_parallel_welford":
        raise ValueError(
            f"Unsupported statistics.method in {path}: "
            f"{statistics.get('method')!r}."
        )
    scope = statistics.get("scope")
    if not isinstance(scope, str) or not scope:
        raise ValueError(f"statistics.scope in {path} must be a nonempty string.")

    std_floor = _finite_float(statistics.get("std_floor"), "statistics.std_floor", path)
    if std_floor <= 0:
        raise ValueError(f"statistics.std_floor in {path} must be positive.")
    episode_start = _nonnegative_int(
        statistics.get("episode_start"), "statistics.episode_start", path
    )
    episode_stop = _nonnegative_int(
        statistics.get("episode_stop"), "statistics.episode_stop", path
    )
    if episode_stop <= episode_start:
        raise ValueError(
            f"Invalid empty or reversed episode range [{episode_start}, {episode_stop}) "
            f"in {path}."
        )
    episodes = _nonnegative_int(statistics.get("episodes"), "statistics.episodes", path)
    if episodes != episode_stop - episode_start:
        raise ValueError(
            f"statistics.episodes={episodes} in {path} does not match range "
            f"[{episode_start}, {episode_stop})."
        )

    moments: dict[str, dict[str, Any]] = {}
    for name in MOMENT_NAMES:
        count = _matrix(
            statistics.get(f"{name}_count"),
            f"statistics.{name}_count",
            path,
            counts=True,
        )
        mean = _matrix(
            shard.get(f"{name}_tcp_mean"), f"{name}_tcp_mean", path
        )
        raw_std = _matrix(
            statistics.get(f"{name}_raw_std"),
            f"statistics.{name}_raw_std",
            path,
        )
        safe_std = _matrix(
            shard.get(f"{name}_tcp_std"), f"{name}_tcp_std", path
        )
        for arm in range(MOMENT_SHAPE[0]):
            for feature in range(MOMENT_SHAPE[1]):
                if count[arm][feature] <= 0:
                    raise ValueError(
                        f"statistics.{name}_count[{arm}][{feature}] in {path} "
                        "must be positive."
                    )
                if raw_std[arm][feature] < 0:
                    raise ValueError(
                        f"statistics.{name}_raw_std[{arm}][{feature}] in {path} "
                        "must be nonnegative."
                    )
                expected = (
                    1.0
                    if raw_std[arm][feature] < std_floor
                    else raw_std[arm][feature]
                )
                if not math.isclose(
                    safe_std[arm][feature], expected, rel_tol=1.0e-12, abs_tol=1.0e-15
                ):
                    raise ValueError(
                        f"{name}_tcp_std[{arm}][{feature}] in {path} is inconsistent "
                        "with raw_std and std_floor."
                    )
        moments[name] = {"count": count, "mean": mean, "raw_std": raw_std}

    return {
        "path": path,
        "std_floor": std_floor,
        "episode_start": episode_start,
        "episode_stop": episode_stop,
        "episodes": episodes,
        "anchors": _nonnegative_int(
            statistics.get("anchors"), "statistics.anchors", path
        ),
        "valid_targets": _nonnegative_int(
            statistics.get("valid_targets"), "statistics.valid_targets", path
        ),
        "scope": scope,
        "gripper_open_value": _endpoint(
            shard.get("gripper_open_value"), "gripper_open_value", path
        ),
        "gripper_closed_value": _endpoint(
            shard.get("gripper_closed_value"), "gripper_closed_value", path
        ),
        "moments": moments,
    }


def _merge_moment_shards(
    shards: list[dict[str, Any]], name: str
) -> dict[str, list[list[float]] | list[list[int]]]:
    count = [[0 for _ in range(MOMENT_SHAPE[1])] for _ in range(MOMENT_SHAPE[0])]
    mean = [[0.0 for _ in range(MOMENT_SHAPE[1])] for _ in range(MOMENT_SHAPE[0])]
    m2 = [[0.0 for _ in range(MOMENT_SHAPE[1])] for _ in range(MOMENT_SHAPE[0])]

    for shard in shards:
        incoming = shard["moments"][name]
        for arm in range(MOMENT_SHAPE[0]):
            for feature in range(MOMENT_SHAPE[1]):
                old_count = count[arm][feature]
                new_count = incoming["count"][arm][feature]
                total_count = old_count + new_count
                old_mean = mean[arm][feature]
                new_mean = incoming["mean"][arm][feature]
                new_m2 = (
                    incoming["raw_std"][arm][feature] ** 2 * new_count
                )
                delta = new_mean - old_mean
                mean[arm][feature] = old_mean + delta * new_count / total_count
                m2[arm][feature] = (
                    m2[arm][feature]
                    + new_m2
                    + delta * delta * old_count * new_count / total_count
                )
                count[arm][feature] = total_count

    raw_std = [
        [
            math.sqrt(max(0.0, m2[arm][feature] / count[arm][feature]))
            for feature in range(MOMENT_SHAPE[1])
        ]
        for arm in range(MOMENT_SHAPE[0])
    ]
    return {"count": count, "mean": mean, "raw_std": raw_std}


def merge_statistics(paths: list[Path]) -> dict[str, Any]:
    """Validate and merge contiguous shard JSONs into one formal v2 object."""
    if not paths:
        raise ValueError("At least one input statistics shard is required.")
    resolved = [path.expanduser().resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Input statistics shard paths must be unique.")
    shards = sorted(
        (_validated_shard(path) for path in resolved),
        key=lambda shard: shard["episode_start"],
    )

    if shards[0]["episode_start"] != 0:
        raise ValueError(
            f"Merged statistics must start at episode 0, got "
            f"{shards[0]['episode_start']}."
        )
    for previous, current in zip(shards, shards[1:]):
        if current["episode_start"] != previous["episode_stop"]:
            raise ValueError(
                "Statistics episode ranges must be contiguous and non-overlapping; "
                f"{previous['path']} ends at {previous['episode_stop']}, but "
                f"{current['path']} starts at {current['episode_start']}."
            )

    reference = shards[0]
    for shard in shards[1:]:
        if shard["std_floor"] != reference["std_floor"]:
            raise ValueError(
                f"Mismatched std_floor in {shard['path']}: {shard['std_floor']} != "
                f"{reference['std_floor']}."
            )
        for endpoint in ("gripper_open_value", "gripper_closed_value"):
            if shard[endpoint] != reference[endpoint]:
                raise ValueError(
                    f"Mismatched {endpoint} in {shard['path']}: "
                    f"{shard[endpoint]} != {reference[endpoint]}."
                )
        if shard["scope"] != reference["scope"]:
            raise ValueError(
                f"Mismatched statistics.scope in {shard['path']}: "
                f"{shard['scope']!r} != {reference['scope']!r}."
            )

    merged = {name: _merge_moment_shards(shards, name) for name in MOMENT_NAMES}
    std_floor = reference["std_floor"]
    safe_std = {
        name: [
            [1.0 if value < std_floor else value for value in row]
            for row in merged[name]["raw_std"]
        ]
        for name in MOMENT_NAMES
    }
    floor_applied = {
        name: [
            [value < std_floor for value in row]
            for row in merged[name]["raw_std"]
        ]
        for name in MOMENT_NAMES
    }

    episode_stop = shards[-1]["episode_stop"]
    return {
        "schema_version": SCHEMA_VERSION,
        "state_tcp_mean": merged["state"]["mean"],
        "state_tcp_std": safe_std["state"],
        "action_tcp_mean": merged["action"]["mean"],
        "action_tcp_std": safe_std["action"],
        "gripper_open_value": list(reference["gripper_open_value"]),
        "gripper_closed_value": list(reference["gripper_closed_value"]),
        "statistics": {
            "method": "population_zscore_parallel_welford",
            "scope": reference["scope"],
            "std_floor": std_floor,
            "episode_start": 0,
            "episode_stop": episode_stop,
            "episodes": sum(shard["episodes"] for shard in shards),
            "anchors": sum(shard["anchors"] for shard in shards),
            "valid_targets": sum(shard["valid_targets"] for shard in shards),
            "shards": len(shards),
            "state_count": merged["state"]["count"],
            "action_count": merged["action"]["count"],
            "state_raw_std": merged["state"]["raw_std"],
            "action_raw_std": merged["action"]["raw_std"],
            "state_std_floor_applied": floor_applied["state"],
            "action_std_floor_applied": floor_applied["action"],
        },
    }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge contiguous canonical z-score statistic shards."
    )
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    inputs = [path.expanduser().resolve() for path in args.inputs]
    if output in inputs:
        raise ValueError("Output path must not overwrite an input shard.")
    result = merge_statistics(inputs)
    _atomic_write_json(output, result)
    print(
        f"wrote={output} shards={len(inputs)} "
        f"episodes={result['statistics']['episodes']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
