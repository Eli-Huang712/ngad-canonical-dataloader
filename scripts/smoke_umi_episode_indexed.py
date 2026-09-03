#!/usr/bin/env python3
"""Read known UMI shard-gap boundaries through episode-indexed addressing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ngad_canonical_dataloader.backends.table import ParquetTableBackend


TARGET_EPISODES = (0, 210, 225, 242, 321, 45087, 45122, 90149, 90173)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    backend = ParquetTableBackend(root, info, row_addressing="episode_indexed")
    _, episodes = backend.read_catalog((), torch.zeros(0, dtype=torch.bool), {})
    by_index = {episode["episode_index"]: episode for episode in episodes}
    field_mask = {
        "observation.state": False,
        "observation.tactile.values": False,
        "observation.tactile.dt": False,
    }

    for episode_index in TARGET_EPISODES:
        episode = by_index[episode_index]
        relative_indices = sorted(
            {
                0,
                min(1, episode["length"] - 1),
                min(31, episode["length"] - 1),
                min(80, episode["length"] - 1),
                episode["length"] - 1,
            }
        )
        rows = backend.read_rows(
            episode,
            torch.tensor(relative_indices, dtype=torch.int64),
            field_mask,
            {},
            (),
            torch.zeros(0, dtype=torch.bool),
        )
        assert sorted(rows) == relative_indices
        for relative_index, row in rows.items():
            assert int(row["index"]) == episode["dataset_from_index"] + relative_index
            assert int(row["episode_index"]) == episode_index
            assert int(row["frame_index"]) == relative_index
        file_key = (episode["data_chunk_index"], episode["data_file_index"])
        local_start, local_end = backend._episode_row_ranges[file_key][episode_index]
        print(
            f"episode={episode_index} file={file_key} local=[{local_start},{local_end}) "
            f"global=[{episode['dataset_from_index']},{episode['dataset_to_index']}) PASS",
            flush=True,
        )

    print(f"episodes={len(episodes)} targets={len(TARGET_EPISODES)} PASS", flush=True)


if __name__ == "__main__":
    main()
