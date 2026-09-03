#!/usr/bin/env bash
#SBATCH --job-name=hy-zstats
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

set -euo pipefail

repo=/gpfs/jiuquyun/projects/xuancan/ngad-canonical-dataloader
dataset_root=/gpfs/jiuquyun/datasets/PRETRAIN_DATA/Hy-Embodied-0.5-VLA-Data
config_root="$dataset_root/dataset_configs/configs_canonical_state"
python_bin=/gpfs/jiuquyun/projects/xuancan/envs/dataloader/bin/python
staging_root="${STAGING_ROOT:-$repo/outputs/hy-zscore-v2-staging}"

mapfile -t configs < <(find "$config_root" -maxdepth 1 -type f -name 'hy_table_*.yaml' | sort)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_id < 0 || task_id >= ${#configs[@]} )); then
  echo "Invalid array task $task_id for ${#configs[@]} configs" >&2
  exit 2
fi

config="${configs[$task_id]}"
table_name="$(basename "$config" .yaml)"
output="$staging_root/${table_name}.json"
args=(
  --config "$config"
  --output "$output"
  --anchor-batch-size "${ANCHOR_BATCH_SIZE:-2048}"
  --std-floor "${STD_FLOOR:-1e-5}"
  --progress-every "${PROGRESS_EVERY:-100}"
)
if [[ -n "${MAX_EPISODES:-}" ]]; then
  args+=(--max-episodes "$MAX_EPISODES")
fi
if [[ -n "${MAX_ANCHORS:-}" ]]; then
  args+=(--max-anchors "$MAX_ANCHORS")
fi

mkdir -p "$staging_root"
cd "$repo"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} task=$task_id config=$config output=$output"
"$python_bin" -m ngad_canonical_dataloader.statistics "${args[@]}"
