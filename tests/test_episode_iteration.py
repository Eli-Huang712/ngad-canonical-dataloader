import pytest

from ngad_canonical_dataloader import NGADCanonicalDataset


def _dataset_with_episode_boundaries() -> NGADCanonicalDataset:
    dataset = NGADCanonicalDataset.__new__(NGADCanonicalDataset)
    dataset._episodes = [
        {"episode_index": 10, "root_index": 0, "tasks": ("task ten",)},
        {"episode_index": 11, "root_index": 0},
        {"episode_index": 12, "root_index": 0, "tasks": ("task twelve",)},
    ]
    dataset._root_meta = [
        {
            "dataset_name": "demo",
            "table_name": "table_000",
            "tasks": {10: "task ten", 12: "task twelve"},
        }
    ]
    dataset._episode_window_ends = [2, 5, 9]
    dataset._length = 7
    return dataset


def test_iter_episode_samples_reuses_global_getitem_indices(monkeypatch) -> None:
    dataset = _dataset_with_episode_boundaries()
    monkeypatch.setattr(
        NGADCanonicalDataset,
        "__getitem__",
        lambda self, index: {"sample_index": index},
    )

    samples = list(dataset.iter_episode_samples(11))

    assert samples == [
        {"sample_index": 2},
        {"sample_index": 3},
        {"sample_index": 4},
    ]
    assert dataset._episode_window_ends == [2, 5, 9]
    assert len(dataset) == 7


def test_iter_episode_samples_respects_max_samples(monkeypatch) -> None:
    dataset = _dataset_with_episode_boundaries()
    monkeypatch.setattr(
        NGADCanonicalDataset,
        "__getitem__",
        lambda self, index: {"sample_index": index},
    )

    assert list(dataset.iter_episode_samples(12)) == [
        {"sample_index": 5},
        {"sample_index": 6},
    ]


def test_iter_episode_samples_rejects_missing_or_ambiguous_episode() -> None:
    dataset = _dataset_with_episode_boundaries()
    with pytest.raises(KeyError, match="not part of this Dataset split"):
        list(dataset.iter_episode_samples(999))

    dataset._episodes[2]["episode_index"] = 10
    with pytest.raises(ValueError, match="ambiguous across configured tables"):
        list(dataset.iter_episode_samples(10))


def test_episode_catalog_uses_metadata_without_loading_samples(monkeypatch) -> None:
    dataset = _dataset_with_episode_boundaries()
    monkeypatch.setattr(
        NGADCanonicalDataset,
        "__getitem__",
        lambda self, index: pytest.fail(f"unexpected sample read: {index}"),
    )

    assert dataset.episode_catalog() == [
        {
            "dataset_name": "demo",
            "table_name": "table_000",
            "episode_index": 10,
            "sample_count": 2,
            "task_indices": [10],
            "prompts": ["task ten"],
        },
        {
            "dataset_name": "demo",
            "table_name": "table_000",
            "episode_index": 11,
            "sample_count": 3,
            "task_indices": [],
            "prompts": [],
        },
        {
            "dataset_name": "demo",
            "table_name": "table_000",
            "episode_index": 12,
            "sample_count": 2,
            "task_indices": [12],
            "prompts": ["task twelve"],
        },
    ]
