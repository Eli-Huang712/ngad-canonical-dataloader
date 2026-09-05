from pathlib import Path

import pytest

from tools.episode_browser import EpisodeBrowserState, load_viewer_catalog


def test_viewer_catalog_resolves_dataset_yaml_relative_to_catalog(tmp_path: Path) -> None:
    (tmp_path / "dataset.yaml").write_text("dataset: {}\n", encoding="utf-8")
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        """datasets:
  - id: demo
    label: Demo Dataset
    config: dataset.yaml
""",
        encoding="utf-8",
    )

    entry = load_viewer_catalog(catalog)[0]
    assert entry.dataset_id == "demo"
    assert entry.label == "Demo Dataset"
    assert entry.config_path == (tmp_path / "dataset.yaml").resolve()


@pytest.mark.parametrize(
    "catalog_text",
    [
        "datasets: []\n",
        "datasets:\n  - id: BAD ID\n    label: Demo\n    config: dataset.yaml\n",
        "datasets:\n  - id: demo\n    label: Demo\n    config: ../dataset.yaml\n",
        "datasets:\n  - id: demo\n    label: Demo\n    config: dataset.yaml\n    extra: true\n",
    ],
)
def test_viewer_catalog_rejects_invalid_entries(
    tmp_path: Path,
    catalog_text: str,
) -> None:
    (tmp_path / "dataset.yaml").write_text("dataset: {}\n", encoding="utf-8")
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(catalog_text, encoding="utf-8")

    with pytest.raises(ValueError):
        load_viewer_catalog(catalog)


def test_browser_close_deletes_current_rrd(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "dataset.yaml").write_text("dataset: {}\n", encoding="utf-8")
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "datasets:\n  - id: demo\n    label: Demo\n    config: dataset.yaml\n",
        encoding="utf-8",
    )
    visualizer = tmp_path / "visualize_episode.py"
    visualizer.write_text("", encoding="utf-8")
    monkeypatch.setattr("tools.episode_browser.shutil.which", lambda name: "/bin/true")
    state = EpisodeBrowserState(catalog, tmp_path / "temporary", 19001, 19002, visualizer)
    recording = tmp_path / "temporary" / "episode.rrd"
    recording.write_bytes(b"recording")
    state._rrd_path = recording

    state.close()

    assert not recording.exists()
    assert state.status() == {"state": "stopped"}
