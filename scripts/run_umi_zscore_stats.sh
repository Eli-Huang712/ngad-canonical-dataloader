#!/usr/bin/env bash
#SBATCH --job-name=umi-zstats
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%4

set -euo pipefail

repo=/gpfs/jiuquyun/projects/xuancan/ngad-canonical-dataloader
python_bin=/gpfs/jiuquyun/projects/xuancan/envs/dataloader/bin/python
config="$repo/configs/umi_selfcollect_stats_source.yaml"
output_root="$repo/outputs/umi-zscore-v2-parts"
total_episodes=90174
shard_count=8
shard_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

episode_start=$((total_episodes * shard_index / shard_count))
episode_stop=$((total_episodes * (shard_index + 1) / shard_count))
part_name=$(printf "part-%03d.json" "$shard_index")

mkdir -p "$output_root"
cd "$repo"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "host=$(hostname) job=$SLURM_JOB_ID task=$shard_index range=[$episode_start,$episode_stop)"
"$python_bin" -m ngad_canonical_dataloader.statistics \
  --config "$config" \
  --output "$output_root/$part_name" \
  --episode-start "$episode_start" \
  --episode-stop "$episode_stop" \
  --anchor-batch-size 2048 \
  --std-floor 1e-5 \
  --progress-every 500
