from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_episode_viewer_shell_scripts_have_valid_syntax() -> None:
    for relative_path in (
        "tools/remote/serve_episode_slurm.sh",
        "tools/local/view-h100",
    ):
        subprocess.run(
            ["bash", "-n", str(REPOSITORY_ROOT / relative_path)],
            check=True,
        )


def test_slurm_service_uses_disposable_selector_path() -> None:
    script = (REPOSITORY_ROOT / "tools/remote/serve_episode_slurm.sh").read_text(
        encoding="utf-8"
    )
    assert "tools/remote/episode_browser.py" in script
    assert "ngad_episode_viewer_${job_id}" in script
    assert 'export PATH="${repo_root}/.venv/bin:${PATH}"' in script
    assert 'trap cleanup EXIT INT TERM' in script
    assert "<dataset-yaml> <episode-index>" not in script


def test_local_launcher_accepts_only_personal_project_checkouts() -> None:
    script = (REPOSITORY_ROOT / "tools/local/view-h100").read_text(encoding="utf-8")
    assert script.splitlines()[1].startswith("#")
    assert '""":' not in script
    assert "NGAD_H100_REPO" in script
    assert '"/gpfs/jiuquyun/projects/${remote_user}/"*' in script
