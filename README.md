# ngad-canonical-dataloader

独立的 canonical Loading / Sample Construction package。它从 canonical 数据集读取样本，
以真实 RGB frame 为 anchor，输出统一的六视角 video、时间坐标、validity mask 和
metadata；提供 global normalization 时额外输出 absolute State TCP128 与 anchor-relative
Action TCP128。

本仓库只负责 `Dataset.__init__()` 与 `Dataset.__getitem__()`。PyTorch `DataLoader`、
GPU transfer、Tokenizer、VAE、flow construction 和模型不在本仓库范围内。完整架构和边界见
[Input Pipeline Architecture](docs/architecture.md)，详细字段和数值变换见
[Canonical Data Contract](docs/data-contract.md)。

## 快速开始

### 1. 配置数据集和统一时间轴

```yaml
dataset:
  normalization_stats_path: /path/to/stats/canonical_global_normalization.json

  dataset_dirs:
    - name: libero
      path: /path/to/canonical/libero
      mask_and_mapping_path: /path/to/canonical/libero/mask_and_mapping.json

    - name: hy_embodied
      path: /path/to/canonical/hy_embodied
      mask_and_mapping_path: /path/to/canonical/hy_embodied/mask_and_mapping.json

  timeline:
    rgb_rate_hz: 10
    action_steps_per_rgb_frame: 2
    anchor_offset: 0
    frame_ranges:
      - [-232, -225]
      - [-182, -175]
      - [-132, -125]
      - [-82, -75]
      - [-32, 16]

  split: train
  validation_split: 0.0
  validation_seed: 3407
  max_samples: null
```

上述配置展开为 81 个具有真实时间意义的 RGB offset。`action_steps_per_rgb_frame: 2`
表示每个 RGB frame 对齐两个 State/Action 子步，因此输出：

```text
video  : [81,6,3,256,256]
state  : [81,2,128]
action : [81,2,128]
```

完整模板见 [configs/canonical.yaml](configs/canonical.yaml)。

`normalization_stats_path` 是必填全局字段。非空路径进入 canonical mode；设为 `null` 时
进入临时的 video-only mode，只加载真实 video/mask/prompt/frame metadata，不读取统计量，
也不生成 state/action、dummy stats 或伪造 TCP128。

### 2. 构造 Dataset

```python
from ngad_canonical_dataloader import build_dataset_from_yaml

dataset = build_dataset_from_yaml("configs/canonical.yaml")
sample = dataset[0]
```

### 3. 读取指定 offset

offset 是相对当前 anchor 的语义时间坐标，不是 tensor position。例如 10 Hz 下，offset
`-1` 表示 anchor 之前 `0.1s`：

```python
position = dataset.timeline_layout.position(-1)

if not bool(sample["frame_valid"][position]):
    raise IndexError("offset -1 is outside the current episode")

frame = sample["video"][position]                       # [6,3,256,256]
frame_timestamp = sample["frame_timestamps"][position] # scalar float64
source_index = sample["source_frame_indices"][position]

state = sample["state"][position]                      # canonical mode only
action = sample["action"][position]                    # canonical mode only
action_timestamp = sample["action_timestamps"][position]
action_valid = sample["action_valid"][position]

valid_state = state[action_valid]
valid_action = action[action_valid]
```

Tokenizer 或模型适配层也可以通过 offset 选择时间角色：

```python
past = sample["frame_offsets"] < 0
anchor = sample["frame_offsets"] == 0
future = sample["frame_offsets"] > 0
```

Canonical mode 保留 `state[N,K,128]` 和 `action[N,K,128]`；如果模型需要
`[N*K,128]`，由模型适配层 reshape。Video-only mode 不返回这两个字段。

## 输入 ABI 简述

支持两种 physical backend，输出相同：

- Canonical Lance table + JPEG payload；
- LeRobot v3 Parquet + H.264 MP4。

### 期望的数据集文件拓扑

`dataset_dirs[].path` 接受两种确定性正式拓扑：包含多个直接 `table_NNN` 子目录的
dataset root，或直接指向一张 `table_NNN` 的 single-table root。前者只枚举直接子目录并
按数字排序；后者只读取自身，不在 table 内继续搜索。两者都不递归，也不存在
`table_name` 配置字段。

```text
<dataset-root>/
├── table_000/
│   ├── meta/
│   │   ├── info.json
│   │   ├── stats.json
│   │   ├── tasks.parquet
│   │   └── episodes/
│   │       └── *.parquet
│   │
│   └── table_000.lance/             # Lance + inline JPEG
│       ├── _transactions/
│       ├── _versions/
│       └── data/
│           └── *.lance
│
├── table_001/
│   ├── meta/
│   │   ├── info.json
│   │   ├── tasks.parquet
│   │   └── episodes/
│   │       └── *.parquet
│   │
│   ├── data/                         # Parquet
│   │   └── .../*.parquet
│   └── videos/                       # H.264
│       └── <camera-key>/*.mp4
│
└── ...
```

Loader 按目录名中的数字排序，例如 `table_000`、`table_001`。每张 table 的
`meta/info.json` 必须声明 `total_episodes` 和 `total_frames`；Episode metadata 中的
`dataset_from_index`、`dataset_to_index` 以及数据行 `index` 都是 table-local。Dataset
再把所有 table 的合法窗口串成一个对外的全局 `__len__` / `__getitem__` 索引空间。

backend 由每张 table 实际发布的唯一 payload 确定：`<table-name>.lance/` 表示 Lance +
inline JPEG，`data/` 与 `videos/` 表示 Parquet + H.264。Loader 不接受根级 manifest、
fragment、shard、`.work` 或 `.publishing` 等旧拓扑。

六路相机固定顺序：

```text
0  observation.images.cam_head_left
1  observation.images.cam_head_right
2  observation.images.cam_left_wrist_left
3  observation.images.cam_left_wrist_right
4  observation.images.cam_right_wrist_left
5  observation.images.cam_right_wrist_right
```

核心 canonical 字段：

| 字段 | dtype / shape | 语义 |
|---|---|---|
| 六路 `observation.images.*` | `image[256,256,3]` 或 `video[256,256,3]` | Lance inline JPEG 使用 `image`，Parquet/H.264 使用 `video`；缺失视角由 mask 声明 |
| `observation.state` | `float32[20]` | 双臂 absolute TCP，reshape 为 `[2,10]` |
| `action` | `float32[20]` | 落盘字段；不作为训练监督，Action 由 state window 重算 |
| `observation.tactile.values` | `float32[4,3,25,6]` | 可选触觉值 |
| `observation.tactile.dt` | `float32[4,3]` | 可选触觉时间差 |
| identity fields | scalar | `timestamp/frame_index/episode_index/index/task_index` |

每臂 TCP10 为 XYZ `[3]`、row-major Rot6D `[6]`、absolute gripper openness `[1]`。
每个 `dataset_dirs` 条目只提供 `name`、`path` 和 `mask_and_mapping_path`。一次混合训练在
`dataset.normalization_stats_path` 提供唯一的 global normalization；所有 physical table
共用它。Dataset 不在训练时统计，也不接受各 table 的 `meta/stats.json` 代替正式的
anchor-relative canonical global stats。

物理目录拓扑、mask JSON、时间轴、插值、relative pose、normalization 和 TCP128D 规划见
[Canonical Data Contract](docs/data-contract.md)。

## 输出 ABI 简述

设 `N = len(frame_offsets)`、`K = action_steps_per_rgb_frame`：

两种模式都返回：

```python
{
    "video":                 float32[N, 6, 3, 256, 256],
    "frame_offsets":         int64[N],
    "source_frame_indices":  int64[N],
    "frame_timestamps":      float64[N],
    "frame_valid":           bool[N],
    "camera_mask":           bool[N, 6],
    "image_pixel_mask":      bool[N, 6, 256, 256],
    "prompt":                str,
    "data_info":             dict,  # sample_mode="canonical" | "video_only"
}
```

Canonical mode 另外返回：

```python
{
    "state":                 float32[N, K, 128],  # absolute TCP
    "action":                float32[N, K, 128],  # fixed-anchor relative TCP
    "action_step_offsets":   int64[N, K],
    "action_timestamps":     float64[N, K],
    "action_valid":          bool[N, K],

    "state_feature_mask":    bool[128],
    "action_feature_mask":   bool[128],
}
```

Video-only mode 不返回 state/action、feature/element mask、Action offsets/timestamps 或
`action_valid`。

State 和 Action 共享时间点、timestamps 和 validity。两者均使用左臂 `0:10`、右臂
`10:20`、保留/屏蔽 `20:128` 的 TCP128D 布局；State 保留 absolute pose，Action 的
XYZ/rotation 相对于 offset `0` anchor，gripper 保留 absolute openness。

完整 element/tactile mask 和 `data_info` 字段见
[Canonical Data Contract — 完整输出 ABI](docs/data-contract.md#6-完整输出-abi)。

## 详细文档

- [Input Pipeline Architecture](docs/architecture.md)：完整链路、仓库边界、内部数据流和文件职责。
- [Canonical Data Contract](docs/data-contract.md)：物理后端、输入字段、时间轴、插值、relative pose、normalization、TCP128D 和完整输出 ABI。
