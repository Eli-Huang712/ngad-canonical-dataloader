#!/usr/bin/env bash
#SBATCH --partition=h100
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: sbatch tools/serve_episode_slurm.sh <dataset-yaml> <episode-index> <output-dir>" >&2
    exit 2
fi

dataset_config=$1
episode_index=$2
output_dir=$3
repo_root=${SLURM_SUBMIT_DIR:?Submit this job from the repository root.}
job_id=${SLURM_JOB_ID:?This script must run inside a Slurm job.}
node_name=$(hostname -s)
grpc_port=$((20000 + (job_id % 10000) * 2))
web_port=$((grpc_port + 1))
rrd_path="${output_dir}/episode_${episode_index}_job_${job_id}.rrd"

umask 0027
mkdir -p "$output_dir"

python "$repo_root/tools/visualize_episode.py" \
    --dataset-config "$dataset_config" \
    --episode-index "$episode_index" \
    --output "$rrd_path"

echo "RRD_READY=$rrd_path"
echo "COMPUTE_NODE=$node_name"
echo "WEB_PORT=$web_port"
echo "GRPC_PORT=$grpc_port"
echo "Run this command on your local machine:"
echo "ssh -N -L ${web_port}:${node_name}:${web_port} -L ${grpc_port}:${node_name}:${grpc_port} h100"
echo "Then open:"
echo "http://127.0.0.1:${web_port}/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A${grpc_port}%2Fproxy"

exec rerun "$rrd_path" \
    --serve-web \
    --web-viewer-port "$web_port" \
    --port "$grpc_port"
