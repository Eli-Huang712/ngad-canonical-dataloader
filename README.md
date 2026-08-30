# ngad-canonical-dataloader

独立的 canonical Loading / Sample Construction package。它从 canonical 数据集读取样本，
以真实 RGB frame 为 anchor，输出统一的六视角 video、absolute State TCP128、
anchor-relative Action TCP128、时间坐标、validity mask 和 metadata。

本仓库只负责 `Dataset.__init__()` 与 `Dataset.__getitem__()`。PyTorch `DataLoader`、
GPU transfer、Tokenizer、VAE、flow construction 和模型不在本仓库范围内。完整架构和边界见
[Input Pipeline Architecture](docs/architecture.md)，详细字段和数值变换见
[Canonical Data Contract](docs/data-contract.md)。

## 快速开始

### 1. 配置数据集和统一时间轴

```yaml
schema_version: ngad_canonical_dataloader_v2

dataset:
  dataset_dirs:
    - name: libero
      path: /path/to/canonical/libero
      mask_and_mapping_path: /path/to/canonical/libero/mask_and_mapping.json
      normalization_stats_path: /path/to/stats/libero.json

    - name: hy_embodied
      path: /path/to/canonical/hy_embodied
      mask_and_mapping_path: /path/to/canonical/hy_embodied/mask_and_mapping.json
      normalization_stats_path: /path/to/stats/hy_embodied.json

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

state = sample["state"][position]                      # [K,128], absolute TCP
action = sample["action"][position]                    # [K,128], relative TCP
action_timestamp = sample["action_timestamps"][position]  # [K]
action_valid = sample["action_valid"][position]        # [K]

valid_state = state[action_valid]
valid_action = action[action_valid]
```

Tokenizer 或模型适配层也可以通过 offset 选择时间角色：

```python
past = sample["frame_offsets"] < 0
anchor = sample["frame_offsets"] == 0
future = sample["frame_offsets"] > 0
```

Loading ABI 始终保留 `state[N,K,128]` 和 `action[N,K,128]`。如果模型需要
`[N*K,128]`，由模型适配层 reshape。

## 输入 ABI 简述

支持两种 physical backend，输出相同：

- Canonical Lance table + JPEG payload；
- LeRobot v3 Parquet + H.264 MP4。

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
| 六路 `observation.images.*` | `video[256,256,3]` | RGB；缺失视角由 mask 声明 |
| `observation.state` | `float32[20]` | 双臂 absolute TCP，reshape 为 `[2,10]` |
| `action` | `float32[20]` | 落盘字段；不作为训练监督，Action 由 state window 重算 |
| `observation.tactile.values` | `float32[4,3,25,6]` | 可选触觉值 |
| `observation.tactile.dt` | `float32[4,3]` | 可选触觉时间差 |
| identity fields | scalar | `timestamp/frame_index/episode_index/index/task_index` |

每臂 TCP10 为 XYZ `[3]`、row-major Rot6D `[6]`、absolute gripper openness `[1]`。
每个 `dataset_dirs` 条目必须提供 `mask_and_mapping_path` 和离线生成的
`normalization_stats_path`。前者同时定义 canonical 字段有效性和
`canonical key -> physical storage key` 映射；Dataset 不在训练时统计、猜测缺失字段或
兼容旧 `mask_path`。

物理目录拓扑、mask JSON、时间轴、插值、relative pose、normalization 和 TCP128D 规划见
[Canonical Data Contract](docs/data-contract.md)。

## 输出 ABI 简述

设 `N = len(frame_offsets)`、`K = action_steps_per_rgb_frame`：

```python
{
    "video":                 float32[N, 6, 3, 256, 256],
    "frame_offsets":         int64[N],
    "source_frame_indices":  int64[N],
    "frame_timestamps":      float64[N],
    "frame_valid":           bool[N],

    "state":                 float32[N, K, 128],  # absolute TCP
    "action":                float32[N, K, 128],  # fixed-anchor relative TCP
    "action_step_offsets":   int64[N, K],
    "action_timestamps":     float64[N, K],
    "action_valid":          bool[N, K],

    "state_feature_mask":    bool[128],
    "action_feature_mask":   bool[128],
    "camera_mask":           bool[N, 6],
    "image_pixel_mask":      bool[N, 6, 256, 256],

    "prompt":                str,
    "data_info":             dict,
}
```

State 和 Action 共享时间点、timestamps 和 validity。两者均使用左臂 `0:10`、右臂
`10:20`、保留/屏蔽 `20:128` 的 TCP128D 布局；State 保留 absolute pose，Action 的
XYZ/rotation 相对于 offset `0` anchor，gripper 保留 absolute openness。

完整 element/tactile mask 和 `data_info` 字段见
[Canonical Data Contract — 完整输出 ABI](docs/data-contract.md#6-完整输出-abi)。

## 详细文档

- [Input Pipeline Architecture](docs/architecture.md)：完整链路、仓库边界、内部数据流和文件职责。
- [Canonical Data Contract](docs/data-contract.md)：物理后端、输入字段、时间轴、插值、relative pose、normalization、TCP128D 和完整输出 ABI。
