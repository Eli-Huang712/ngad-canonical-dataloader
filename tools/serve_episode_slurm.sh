#!/usr/bin/env bash
#SBATCH --partition=h100
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch tools/serve_episode_slurm.sh <viewer-catalog-yaml>" >&2
    exit 2
fi

viewer_catalog=$1
repo_root=${SLURM_SUBMIT_DIR:?Submit this job from the repository root.}
job_id=${SLURM_JOB_ID:?This script must run inside a Slurm job.}
node_name=$(hostname -s)
selector_port=$((20000 + (job_id % 10000) * 3))
grpc_port=$((selector_port + 1))
web_port=$((selector_port + 2))
temporary_root=${SLURM_TMPDIR:-/tmp}
temporary_directory="${temporary_root}/ngad_episode_viewer_${job_id}"
browser_pid=""

umask 0027
mkdir -p "$temporary_directory"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

cleanup() {
    if [[ -n "$browser_pid" ]] && kill -0 "$browser_pid" 2>/dev/null; then
        kill "$browser_pid" 2>/dev/null || true
        wait "$browser_pid" 2>/dev/null || true
    fi
    case "$temporary_directory" in
        "${temporary_root}/ngad_episode_viewer_${job_id}")
            rm -rf -- "$temporary_directory"
            ;;
        *)
            echo "Refusing to remove unexpected temporary path: $temporary_directory" >&2
            ;;
    esac
}
trap cleanup EXIT INT TERM

echo "COMPUTE_NODE=$node_name"
echo "SELECTOR_PORT=$selector_port"
echo "WEB_PORT=$web_port"
echo "GRPC_PORT=$grpc_port"

python "$repo_root/tools/episode_browser.py" \
    --catalog "$viewer_catalog" \
    --temporary-directory "$temporary_directory" \
    --port "$selector_port" \
    --rerun-web-port "$web_port" \
    --rerun-grpc-port "$grpc_port" &
browser_pid=$!
wait "$browser_pid"
