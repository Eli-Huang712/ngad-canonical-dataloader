import importlib.util
from pathlib import Path
import signal
import subprocess
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/remote/episode_browser.py"
MODULE_SPEC = importlib.util.spec_from_file_location("episode_browser", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
EPISODE_BROWSER = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = EPISODE_BROWSER
MODULE_SPEC.loader.exec_module(EPISODE_BROWSER)
EpisodeBrowserState = EPISODE_BROWSER.EpisodeBrowserState
load_viewer_catalog = EPISODE_BROWSER.load_viewer_catalog


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
    monkeypatch.setattr(EPISODE_BROWSER.shutil, "which", lambda name: "/bin/true")
    state = EpisodeBrowserState(catalog, tmp_path / "temporary", 19001, 19002, visualizer)
    recording = tmp_path / "temporary" / "episode.rrd"
    recording.write_bytes(b"recording")
    state._rrd_path = recording

    state.close()

    assert not recording.exists()
    assert state.status() == {"state": "stopped"}


def test_browser_requests_only_one_episode_page(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "dataset.yaml").write_text("dataset: {}\n", encoding="utf-8")
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "datasets:\n  - id: demo\n    label: Demo\n    config: dataset.yaml\n",
        encoding="utf-8",
    )
    visualizer = tmp_path / "visualize_episode.py"
    visualizer.write_text("", encoding="utf-8")
    calls = []

    class FakeDataset:
        def episode_catalog(self, *, page, page_size):
            calls.append((page, page_size))
            return {
                "items": [],
                "page": page,
                "page_size": page_size,
                "total": 250,
                "total_pages": 3,
            }

    monkeypatch.setattr(EPISODE_BROWSER.shutil, "which", lambda name: "/bin/true")
    monkeypatch.setattr(
        EPISODE_BROWSER,
        "build_dataset_from_yaml",
        lambda path: FakeDataset(),
    )
    state = EpisodeBrowserState(catalog, tmp_path / "temporary", 19001, 19002, visualizer)

    page = state.episode_choices("demo", 2)

    assert page["page"] == 2
    assert calls == [(2, 100)]


def test_rendered_browser_javascript_preserves_newline_escape() -> None:
    assert "row.prompts.join('\\n')" in EPISODE_BROWSER.INDEX_HTML


def test_viewer_process_group_is_terminated_even_after_parent_exit(monkeypatch) -> None:
    calls = []

    class FinishedViewer:
        pid = 1234

        @staticmethod
        def wait(*, timeout):
            calls.append(("wait", timeout))

    monkeypatch.setattr(
        EPISODE_BROWSER.os,
        "killpg",
        lambda process_group, sig: calls.append((process_group, sig)),
    )

    EpisodeBrowserState._stop_viewer_process_group(FinishedViewer())

    assert calls == [(1234, signal.SIGTERM), ("wait", 10)]


def test_viewer_process_group_escalates_after_timeout(monkeypatch) -> None:
    calls = []

    class StuckViewer:
        pid = 5678
        waits = 0

        @classmethod
        def wait(cls, *, timeout):
            cls.waits += 1
            calls.append(("wait", timeout))
            if cls.waits == 1:
                raise subprocess.TimeoutExpired("rerun", timeout)

    monkeypatch.setattr(
        EPISODE_BROWSER.os,
        "killpg",
        lambda process_group, sig: calls.append((process_group, sig)),
    )

    EpisodeBrowserState._stop_viewer_process_group(StuckViewer())

    assert calls == [
        (5678, signal.SIGTERM),
        ("wait", 10),
        (5678, signal.SIGKILL),
        ("wait", 10),
    ]


def test_reload_waits_until_both_viewer_ports_are_closed(monkeypatch) -> None:
    state = object.__new__(EpisodeBrowserState)
    open_port_results = iter([[19001, 19002], [19002], []])
    sleeps = []
    monkeypatch.setattr(state, "_open_viewer_ports", lambda: next(open_port_results))
    monkeypatch.setattr(EPISODE_BROWSER.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(EPISODE_BROWSER.time, "sleep", sleeps.append)

    state._wait_for_viewer_ports_to_close()

    assert sleeps == [0.25, 0.25]
