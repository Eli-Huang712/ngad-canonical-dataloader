#!/usr/bin/env python3
"""Serve H100-side Dataset/Episode selection and one temporary Rerun recording."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import yaml

from ngad_canonical_dataloader import build_dataset_from_yaml


CATALOG_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")
EPISODE_PAGE_SIZE = 100


@dataclass(frozen=True)
class ViewerDataset:
    """One user-facing dataset and its strict Dataset YAML."""

    dataset_id: str
    label: str
    config_path: Path


def load_viewer_catalog(path: Path) -> tuple[ViewerDataset, ...]:
    """Read the strict Dataset selector catalog."""
    catalog_path = path.resolve()
    with catalog_path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or set(value) != {"datasets"}:
        raise ValueError("Viewer catalog root must contain exactly datasets.")
    rows = value["datasets"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Viewer catalog datasets must be a non-empty list.")

    entries: list[ViewerDataset] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "label", "config"}:
            raise ValueError("Each viewer dataset must contain exactly id, label, config.")
        dataset_id = str(row["id"])
        label = str(row["label"]).strip()
        relative_config = Path(str(row["config"]))
        if (
            CATALOG_ID.fullmatch(dataset_id) is None
            or dataset_id in seen
            or not label
            or relative_config.is_absolute()
            or ".." in relative_config.parts
        ):
            raise ValueError(f"Invalid viewer dataset entry: {row!r}.")
        config_path = (catalog_path.parent / relative_config).resolve()
        if not config_path.is_file():
            raise ValueError(f"Viewer Dataset YAML does not exist: {config_path}.")
        seen.add(dataset_id)
        entries.append(ViewerDataset(dataset_id, label, config_path))
    return tuple(entries)


class EpisodeBrowserState:
    """Own Dataset catalogs and the single disposable Rerun subprocess."""

    def __init__(
        self,
        catalog_path: Path,
        temporary_directory: Path,
        web_port: int,
        grpc_port: int,
        visualizer_path: Path,
    ) -> None:
        self.entries = load_viewer_catalog(catalog_path)
        self._entry_by_id = {entry.dataset_id: entry for entry in self.entries}
        self._temporary_directory = temporary_directory.resolve()
        self._temporary_directory.mkdir(parents=True, exist_ok=True)
        self._web_port = int(web_port)
        self._grpc_port = int(grpc_port)
        self._visualizer_path = visualizer_path.resolve()
        if not self._visualizer_path.is_file():
            raise ValueError(f"Episode visualizer does not exist: {self._visualizer_path}.")
        rerun = shutil.which("rerun")
        if rerun is None:
            raise RuntimeError("The Rerun CLI is not installed in this environment.")
        self._rerun = rerun
        self._datasets: dict[str, Any] = {}
        self._generation: subprocess.Popen[str] | None = None
        self._viewer: subprocess.Popen[str] | None = None
        self._rrd_path: Path | None = None
        self._status: dict[str, Any] = {"state": "idle"}
        self._closing = False
        self._lock = threading.RLock()

    def dataset_choices(self) -> list[dict[str, str]]:
        return [
            {"id": entry.dataset_id, "label": entry.label}
            for entry in self.entries
        ]

    def episode_choices(self, dataset_id: str, page: int) -> dict[str, Any]:
        """Return one metadata-only page while caching only the Dataset instance."""
        with self._lock:
            self._require_entry(dataset_id)
            if dataset_id not in self._datasets:
                self._datasets[dataset_id] = build_dataset_from_yaml(
                    self._entry_by_id[dataset_id].config_path
                )
            return self._datasets[dataset_id].episode_catalog(
                page=page,
                page_size=EPISODE_PAGE_SIZE,
            )

    def start_render(self, dataset_id: str, episode_index: int, page: int) -> None:
        """Replace the current preview and generate one full episode asynchronously."""
        episode_index = int(episode_index)
        choices = self.episode_choices(dataset_id, page)["items"]
        if episode_index not in {row["episode_index"] for row in choices}:
            raise KeyError(f"Unknown episode {episode_index} for dataset {dataset_id!r}.")
        with self._lock:
            if self._generation is not None:
                raise RuntimeError("An episode is already being generated.")
            self._discard_preview_locked()
            self._status = {
                "state": "generating",
                "dataset_id": dataset_id,
                "episode_index": episode_index,
            }
            thread = threading.Thread(
                target=self._render_worker,
                args=(dataset_id, episode_index),
                daemon=True,
            )
            thread.start()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def close(self) -> None:
        """Stop children and delete the generated RRD before the Slurm job exits."""
        with self._lock:
            self._closing = True
            self._stop_process(self._generation)
            self._generation = None
            self._discard_preview_locked()
            self._status = {"state": "stopped"}

    def _require_entry(self, dataset_id: str) -> ViewerDataset:
        try:
            return self._entry_by_id[dataset_id]
        except KeyError as error:
            raise KeyError(f"Unknown dataset {dataset_id!r}.") from error

    @staticmethod
    def _stop_process(process: subprocess.Popen[str] | None) -> None:
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    def _discard_preview_locked(self) -> None:
        self._stop_process(self._viewer)
        self._viewer = None
        if self._rrd_path is not None:
            self._rrd_path.unlink(missing_ok=True)
            self._rrd_path = None

    def _render_worker(self, dataset_id: str, episode_index: int) -> None:
        entry = self._require_entry(dataset_id)
        rrd_path = self._temporary_directory / f"{dataset_id}_episode_{episode_index}.rrd"
        command = [
            sys.executable,
            str(self._visualizer_path),
            "--dataset-config",
            str(entry.config_path),
            "--episode-index",
            str(episode_index),
            "--output",
            str(rrd_path),
        ]
        try:
            generation = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            with self._lock:
                if self._closing:
                    self._stop_process(generation)
                    return
                self._generation = generation
            output, _ = generation.communicate()
            with self._lock:
                self._generation = None
                if self._closing:
                    rrd_path.unlink(missing_ok=True)
                    return
                if generation.returncode != 0:
                    rrd_path.unlink(missing_ok=True)
                    self._status = {
                        "state": "error",
                        "message": output.strip() or "Episode generation failed.",
                    }
                    return
                self._rrd_path = rrd_path
                self._viewer = subprocess.Popen(
                    [
                        self._rerun,
                        str(rrd_path),
                        "--serve-web",
                        "--web-viewer-port",
                        str(self._web_port),
                        "--port",
                        str(self._grpc_port),
                    ],
                    text=True,
                )
            self._wait_for_viewer()
            with self._lock:
                if self._closing:
                    return
                self._status = {
                    "state": "ready",
                    "dataset_id": dataset_id,
                    "episode_index": episode_index,
                    "viewer_url": self._viewer_url(),
                }
        except Exception as error:
            with self._lock:
                if not self._closing:
                    self._status = {"state": "error", "message": str(error)}
                self._discard_preview_locked()

    def _wait_for_viewer(self) -> None:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            with self._lock:
                viewer = self._viewer
                if self._closing:
                    raise RuntimeError("Episode browser is stopping.")
            if viewer is None or viewer.poll() is not None:
                raise RuntimeError("Rerun viewer exited before becoming ready.")
            try:
                with socket.create_connection(("127.0.0.1", self._web_port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.25)
        raise TimeoutError("Rerun viewer did not open its web port within 30 seconds.")

    def _viewer_url(self) -> str:
        recording = f"rerun+http://127.0.0.1:{self._grpc_port}/proxy"
        return f"http://127.0.0.1:{self._web_port}/?url={quote(recording, safe='')}"


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NGAD Dataset Viewer</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #111318; color: #f2f4f8; }
  main { display: grid; grid-template-columns: 320px 1fr; height: 100vh; }
  aside { padding: 20px; border-right: 1px solid #343842; overflow: auto; }
  h1 { margin: 0 0 18px; font-size: 20px; }
  label { display: block; margin-top: 14px; color: #b9c0cc; font-size: 13px; }
  select, button { width: 100%; margin-top: 6px; padding: 10px; border-radius: 7px; border: 1px solid #474d59; background: #1d2027; color: #fff; }
  button { margin-top: 18px; background: #246bfd; border-color: #246bfd; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .55; cursor: wait; }
  .pager { display: grid; grid-template-columns: 1fr auto 1fr; gap: 8px; align-items: center; margin-top: 8px; }
  .pager button { margin-top: 0; background: #2a2e37; border-color: #474d59; }
  #page-label { min-width: 72px; text-align: center; color: #b9c0cc; font-size: 12px; }
  #details, #status { margin-top: 14px; padding: 12px; border-radius: 7px; background: #1a1d23; white-space: pre-wrap; font-size: 12px; line-height: 1.5; }
  #viewer { width: 100%; height: 100%; border: 0; background: #0b0c0f; }
  #empty { display: grid; place-items: center; height: 100%; color: #8f98a8; }
</style>
</head>
<body><main>
<aside>
  <h1>NGAD Dataset Viewer</h1>
  <label for="dataset">Dataset</label><select id="dataset"></select>
  <label for="episode">Episode</label><select id="episode"></select>
  <div class="pager">
    <button id="previous-page">上一页</button>
    <span id="page-label">- / -</span>
    <button id="next-page">下一页</button>
  </div>
  <div id="details">请选择 Dataset 和 Episode。</div>
  <button id="render">生成并预览</button>
  <div id="status">等待选择。</div>
</aside>
<section id="stage"><div id="empty">选择 Episode 后生成完整 RRD。</div></section>
</main>
<script>
const datasetSelect = document.querySelector('#dataset');
const episodeSelect = document.querySelector('#episode');
const details = document.querySelector('#details');
const statusBox = document.querySelector('#status');
const renderButton = document.querySelector('#render');
const previousPageButton = document.querySelector('#previous-page');
const nextPageButton = document.querySelector('#next-page');
const pageLabel = document.querySelector('#page-label');
const stage = document.querySelector('#stage');
let episodes = [];
let currentPage = 1;
let totalPages = 1;

async function getJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}
function showEpisode() {
  const row = episodes[episodeSelect.selectedIndex];
  if (!row) { details.textContent = '没有可选 Episode。'; return; }
  const taskIndices = row.task_indices.length ? row.task_indices.join(', ') : '按 anchor 在 Rerun 中显示';
  const prompts = row.prompts.length ? row.prompts.join('\n') : '按 anchor 在 Rerun 中显示';
  details.textContent = `Episode: ${row.episode_index}\nFrames: ${row.sample_count}\nTask index: ${taskIndices}\nPrompt: ${prompts}`;
}
function setControlsDisabled(disabled) {
  renderButton.disabled = disabled;
  previousPageButton.disabled = disabled || currentPage <= 1;
  nextPageButton.disabled = disabled || currentPage >= totalPages;
}
async function loadEpisodes(page = 1) {
  setControlsDisabled(true);
  statusBox.textContent = '正在读取 Episode metadata…';
  try {
    const payload = await getJson(`/api/episodes?dataset=${encodeURIComponent(datasetSelect.value)}&page=${page}`);
    episodes = payload.items;
    currentPage = payload.page;
    totalPages = payload.total_pages;
    episodeSelect.replaceChildren(...episodes.map(row => {
      const option = document.createElement('option');
      option.value = row.episode_index;
      const hint = row.prompts.length ? ` — ${row.prompts[0].slice(0, 72)}` : '';
      option.textContent = `${row.episode_index}${hint}`;
      return option;
    }));
    showEpisode();
    pageLabel.textContent = `${currentPage} / ${totalPages}`;
    statusBox.textContent = `共 ${payload.total} 个 Episode；当前显示 ${episodes.length} 个。`;
  } catch (error) { statusBox.textContent = `错误：${error.message}`; }
  setControlsDisabled(false);
  renderButton.disabled = episodes.length === 0;
}
async function pollStatus() {
  const payload = await getJson('/api/status');
  if (payload.state === 'generating') {
    statusBox.textContent = '正在完整解码 Episode 并生成临时 RRD…';
    setTimeout(pollStatus, 1000);
  } else if (payload.state === 'ready') {
    statusBox.textContent = '预览已就绪；切换 Episode 或结束任务时会删除临时 RRD。';
    stage.innerHTML = `<iframe id="viewer" src="${payload.viewer_url}"></iframe>`;
    setControlsDisabled(false);
  } else if (payload.state === 'error') {
    statusBox.textContent = `错误：${payload.message}`;
    setControlsDisabled(false);
  }
}
datasetSelect.addEventListener('change', () => loadEpisodes(1));
episodeSelect.addEventListener('change', showEpisode);
previousPageButton.addEventListener('click', () => loadEpisodes(currentPage - 1));
nextPageButton.addEventListener('click', () => loadEpisodes(currentPage + 1));
renderButton.addEventListener('click', async () => {
  setControlsDisabled(true);
  stage.innerHTML = '<div id="empty">正在生成完整 RRD…</div>';
  try {
    await getJson('/api/render', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({dataset_id: datasetSelect.value, episode_index: Number(episodeSelect.value), page: currentPage})
    });
    pollStatus();
  } catch (error) { statusBox.textContent = `错误：${error.message}`; setControlsDisabled(false); }
});
(async () => {
  const datasets = await getJson('/api/datasets');
  datasetSelect.replaceChildren(...datasets.map(row => {
    const option = document.createElement('option'); option.value = row.id; option.textContent = row.label; return option;
  }));
  await loadEpisodes(1);
})();
</script></body></html>
"""


def make_handler(state: EpisodeBrowserState) -> type[BaseHTTPRequestHandler]:
    """Bind one request handler class to a browser state instance."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_bytes("text/html; charset=utf-8", INDEX_HTML.encode())
                elif parsed.path == "/api/datasets":
                    self._send_json(HTTPStatus.OK, state.dataset_choices())
                elif parsed.path == "/api/episodes":
                    query = parse_qs(parsed.query)
                    dataset_id = query.get("dataset", [""])[0]
                    page = int(query.get("page", ["1"])[0])
                    self._send_json(
                        HTTPStatus.OK,
                        state.episode_choices(dataset_id, page),
                    )
                elif parsed.path == "/api/status":
                    self._send_json(HTTPStatus.OK, state.status())
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            except (KeyError, ValueError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

        def do_POST(self) -> None:  # noqa: N802
            try:
                if urlparse(self.path).path != "/api/render":
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or set(payload) != {
                    "dataset_id",
                    "episode_index",
                    "page",
                }:
                    raise ValueError(
                        "Render payload must contain dataset_id, episode_index, and page."
                    )
                state.start_render(
                    str(payload["dataset_id"]),
                    int(payload["episode_index"]),
                    int(payload["page"]),
                )
                self._send_json(HTTPStatus.ACCEPTED, state.status())
            except (KeyError, ValueError, RuntimeError, json.JSONDecodeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def log_message(self, format: str, *args: Any) -> None:
            print(f"BROWSER_HTTP {self.address_string()} {format % args}", flush=True)

        def _send_json(self, status: HTTPStatus, payload: Any) -> None:
            self._send_bytes(
                "application/json; charset=utf-8",
                json.dumps(payload, ensure_ascii=False).encode(),
                status,
            )

        def _send_bytes(
            self,
            content_type: str,
            payload: bytes,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the NGAD episode selector.")
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--temporary-directory", required=True, type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--rerun-web-port", required=True, type=int)
    parser.add_argument("--rerun-grpc-port", required=True, type=int)
    parser.add_argument(
        "--visualizer",
        type=Path,
        default=Path(__file__).with_name("visualize_episode.py"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    state = EpisodeBrowserState(
        args.catalog,
        args.temporary_directory,
        args.rerun_web_port,
        args.rerun_grpc_port,
        args.visualizer,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    def stop_server(signum: int, frame: Any) -> None:
        del signum, frame
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    print(f"SELECTOR_READY={args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        state.close()


if __name__ == "__main__":
    main()
