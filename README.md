# ngad-canonical-dataloader

`ngad-canonical-dataloader` 是独立的 canonical Loading / Sample Construction 模块。
它从 canonical 数据集读取样本，以某个真实 RGB frame 为 anchor，构造统一的
RGB/State/Action 时间轴、absolute State TCP128、anchor-relative Action TCP128、
normalization、validity mask 和 metadata。

本仓库的边界到 `Dataset.__getitem__()` 为止：不包含 PyTorch `DataLoader`、GPU transfer、
Tokenizer、VAE、flow target/noise construction 或模型代码。后续模块只消费本仓库定义的
sample ABI，不需要知道底层使用 Lance+JPEG 还是 LeRobot v3 Parquet+H.264。

## 1. 输入模块

### 1.1 公共入口与存储后端

本仓库只暴露一个 public Dataset：`NGADCanonicalDataset`。它读取已经转换成 canonical
字段的数据，不包含 LIBERO、Hy-Embodied 或 UMI 的 raw-data adapter。

一个 `dataset_dirs[].path` 可以指向单个物理 root，也可以指向 canonical shard collection：

- 单 root：`<path>/meta/info.json`；
- table/fragment collection：`<path>/table_*/fragments/*/meta/info.json`；
- shard collection：`<path>/shard_*/meta/info.json`。

每个物理 root 由 `meta/info.json` 唯一识别为以下一种后端。

Lance + JPEG：

```text
<physical-root>/
├── meta/
│   ├── info.json
│   ├── tasks.parquet
│   └── episodes/
│       └── *.parquet
├── _versions/
└── data/
    └── *.lance
```

该后端要求 `info.json.canonical_schema == "ngad_hy_canonical_lance_v2"`，并且
`data/` 中恰好存在一个 Lance 数据集。图像以 JPEG payload 存储在 Lance 行中。

LeRobot v3 Parquet + H.264：

```text
<physical-root>/
├── meta/
│   ├── info.json
│   ├── tasks.parquet
│   └── episodes/
│       └── *.parquet
├── data/
│   └── .../*.parquet
└── videos/
    └── .../*.mp4
```

实际 Parquet 和 MP4 路径由 `info.json` 的 `data_path`、`video_path` 模板确定；视频使用
H.264 编码。两个后端都必须通过 episode metadata 提供 episode 长度、全局数据 offset 和
任务索引，Dataset 才能建立统一的全局 anchor/window 索引。

### 1.2 Canonical 字段合同

六路相机是固定逻辑槽位，顺序不可改变：

| 槽位 | Canonical key |
|---:|---|
| 0 | `observation.images.cam_head_left` |
| 1 | `observation.images.cam_head_right` |
| 2 | `observation.images.cam_left_wrist_left` |
| 3 | `observation.images.cam_left_wrist_right` |
| 4 | `observation.images.cam_right_wrist_left` |
| 5 | `observation.images.cam_right_wrist_right` |

Canonical logical schema 如下：

| 字段 | dtype / shape | 语义 |
|---|---|---|
| 六路 `observation.images.*` | `video[256,256,3]` | RGB 图像；物理缺失由 mask 声明 |
| `observation.state` | `float32[20]` | 双臂 absolute TCP，reshape 为 `[2,10]` |
| `action` | `float32[20]` | Canonical schema 字段；训练监督不读取它，而是由 state window 重算 |
| `observation.tactile.values` | `float32[4,3,25,6]` | 触觉值；可由 mask 声明缺失 |
| `observation.tactile.dt` | `float32[4,3]` | 触觉时间差；可由 mask 声明缺失 |
| `timestamp` | `float64[1]` | Episode 内物理时间 |
| `frame_index` | `int64[1]` | Episode 内帧索引 |
| `episode_index` | `int64[1]` | Episode 标识 |
| `index` | `int64[1]` | 数据集全局帧索引 |
| `task_index` | `int64[1]` | 当前 anchor 使用的任务索引 |

每条手臂的 TCP10 顺序固定为：absolute XYZ `[3]`、rotation matrix 前两行按 row-major
展开的 absolute Rot6D `[6]`、absolute gripper openness `[1]`。双臂顺序是左臂、右臂。

`observation.state` 和五个 identity 字段必须物理存在。相机、tactile 和落盘 `action`
是否物理存在，由每个数据集自己的 mask JSON 明确声明；Dataset 不根据数据集名称或存储
后端猜测字段可用性。

### 1.3 Mask 与 normalization 输入

每个 `dataset_dirs` 条目必须显式提供：

- `mask_path`：定义 `field_mask`、state/action 的 `element_mask[20]`，以及共享的
  `image_pixel_mask` NPZ 路径和 key；
- `normalization_stats_path`：该数据集已经离线统计完成的 normalization JSON。

`field_mask` 必须覆盖六路相机、state、action、两个 tactile 字段和五个 identity 字段。
不可用的相机输出黑图，同时 `camera_mask=False`、`image_pixel_mask=False`；不可用的 tactile
字段输出零 tensor，同时对应 `tactile_field_mask=False`。Dataset 不在训练期间计算统计量。

Normalization JSON 使用 `ngad_canonical_tcp_v1` schema，并提供双臂独立的：

```text
state_xyz_min:    [2,3]
state_xyz_max:    [2,3]
action_xyz_scale: [2,3]
```

多个数据集混合时，所有合法 anchor 按窗口数量自然组成同一个全局索引；每个 sample 使用其
所属数据集的 mask 和 normalization stats。

## 2. 时间轴对齐

### 2.1 一个 anchor-relative 时间轴

Dataset 不区分 long、history、current 或 future。YAML 中的 `frame_ranges` 直接描述相对
当前 anchor 的 RGB offset 闭区间，并展开为唯一、按时间递增的 `frame_offsets[N]`。

三个索引概念必须区分：

- `frame_offset`：相对 anchor 的语义时间坐标，物理时间差是
  `frame_offset / rgb_rate_hz`；
- `position`：该 offset 在输出 tensor 中的存储位置；
- `source_frame_index`：抽到具体 anchor 后，在当前 Episode 中实际读取的物理帧。

例如，在 10 Hz 配置中，offset `-1` 表示 anchor 之前 `0.1s`，不是 Python tensor 的
倒数第一个元素。固定布局在 Dataset 初始化时建立，具体 source index 和 timestamp 则在
`__getitem__()` 抽到 anchor 后才实例化。

RGB 只读取真实 source frame，不合成图像。因此每个 source FPS 必须不低于
`rgb_rate_hz`，且必须是 `rgb_rate_hz` 的整数倍。

### 2.2 Action 的帧内 K 维

Action rate 不单独配置，而是由整数倍数确定：

```python
K = action_steps_per_rgb_frame
action_rate_hz = rgb_rate_hz * K
```

Action 与 RGB 共用同一个 frame 轴，并额外保留帧内子步维度：

```text
video[i]  : [6,3,256,256]
action[i] : [K,128]
```

设 `frame_offsets[i] = f`，帧内 Action 位置为 `k in [0,K)`：

```python
frame_time = anchor_time + f / rgb_rate_hz
action_time = frame_time - (K - 1 - k) / (rgb_rate_hz * K)
```

因此每个 RGB frame 对应其前一个 RGB 周期
`(frame_time - 1 / rgb_rate_hz, frame_time]` 内的 K 个 Action。以 10 Hz RGB、`K=2`
为例：

| frame offset | `video[i]` | `action[i,0]` | `action[i,1]` |
|---:|---:|---:|---:|
| `-2` | `-0.20s` | `-0.25s` | `-0.20s` |
| `-1` | `-0.10s` | `-0.15s` | `-0.10s` |
| `0`（anchor） | `0.00s` | `-0.05s` | `0.00s` |
| `1` | `0.10s` | `0.05s` | `0.10s` |
| `2` | `0.20s` | `0.15s` | `0.20s` |

第二个 Action 子步与当前 RGB frame 同时刻，第一个位于前一 RGB frame 与当前 frame
之间。对于稀疏的 `frame_ranges`，Dataset 只返回每个选中 RGB frame 对应的局部 K 个
Action，不填充 range 之间未请求的时间空洞，也不把 Action 展平成 `[N*K,128]`。

### 2.3 State 插值、relative Action 与 TCP128

落盘 `action` 不作为监督。Dataset 在每个 Action timestamp 上重采样 absolute
`observation.state`：

- XYZ：一阶线性插值；
- Rot6D：先恢复 SO(3)，再做 shortest-path quaternion SLERP；
- gripper openness：一阶线性插值。

State 和 Action 复用同一份 absolute TCP 插值结果，避免各自重采样造成时间或数值偏差：

- State：直接对插值后的 absolute TCP 做 absolute normalization；
- Action：以当前 sample 的 offset `0` state 为固定 anchor，对插值后的 absolute TCP
  计算 relative pose，再做 relative normalization。

计算 Action 的公式为：

```python
relative_xyz = R_anchor.T @ (xyz_t - xyz_anchor)
relative_rotation = R_anchor.T @ R_t
gripper = absolute_openness_t
```

必须采用“先插值 absolute TCP，再计算 relative pose”的顺序。这样 XYZ、openness 只在
各自的线性空间插值，rotation 只在 SO(3) 上做一次 SLERP；不会在已经经过 Rot6D 投影、
relative pose 或 normalization 的表示上再次插值。

随后按 sample 所属数据集的外部 stats 做 normalization：absolute State XYZ 使用 min/max，
relative Action XYZ 使用以零为中心的 per-axis scale，Rot6D 不归一化，openness 保持
`[0,1]`。

State 与 Action 使用完全相同的 TCP20→TCP128 槽位规划：

| TCP128 index（Python slice） | 内容 | State 语义 | Action 语义 |
|---|---|---|---|
| `0:3` | 左臂 XYZ | absolute | anchor-relative |
| `3:9` | 左臂 Rot6D | absolute | anchor-relative |
| `9:10` | 左夹爪 openness | absolute | absolute |
| `10:13` | 右臂 XYZ | absolute | anchor-relative |
| `13:19` | 右臂 Rot6D | absolute | anchor-relative |
| `19:20` | 右夹爪 openness | absolute | absolute |
| `20:128` | 保留槽位 | 零且 masked | 零且 masked |

对应的 canonical `element_mask[20]` 原样写入 feature mask 的 `0:20`，`20:128` 始终为
false。State 与 Action 的 tensor shape、时间戳和 validity 完全对齐，都是 `[N,K,*]`。

所有查询都限制在当前 Episode 内。越过 Episode 边界的 RGB/Action 位置保留固定 tensor
槽位，但对应 `frame_valid`、`action_valid`、camera/pixel mask 为 false，不读取相邻
Episode，也不按 `task_index` 切分 Episode。sample 的 prompt 暂时使用 anchor 行的
`task_index`。

## 3. 输出模块

设：

- `N = len(frame_offsets)`；
- `K = action_steps_per_rgb_frame`；
- `V = 6` 个固定相机槽位。

`NGADCanonicalDataset.__getitem__()` 返回以下单样本 ABI：

```python
{
    "video":                         float32[N, 6, 3, 256, 256],
    "frame_offsets":                 int64[N],
    "source_frame_indices":          int64[N],
    "frame_timestamps":              float64[N],
    "frame_valid":                   bool[N],

    "state":                         float32[N, K, 128],
    "action":                        float32[N, K, 128],
    "action_step_offsets":           int64[N, K],
    "action_timestamps":             float64[N, K],
    "action_valid":                  bool[N, K],

    "state_feature_mask":            bool[128],
    "action_feature_mask":           bool[128],
    "observation_state_element_mask": bool[20],
    "action_element_mask":           bool[20],

    "observation.tactile.values":    float32[4, 3, 25, 6],
    "observation.tactile.dt":        float32[4, 3],
    "tactile_field_mask":            bool[2],

    "camera_mask":                   bool[N, 6],
    "image_pixel_mask":              bool[N, 6, 256, 256],
    "prompt":                        str,
    "data_info":                     dict,
}
```

`video` 已从 `uint8[0,255]` 转为 `float32[-1,1]`。`state[N,K,128]` 与
`action[N,K,128]` 共用 `action_step_offsets[N,K]`、`action_timestamps[N,K]` 和
`action_valid[N,K]`，不重复输出另一套 State 时间 metadata。Episode 边界外的 State 和
Action 都填零，并以同一个 `action_valid=False` 排除。

`data_info` 的字段合同为：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `img_hw` | `float32[2]` | `[height,width]` |
| `aspect_ratio` | scalar `float32` | `width / height` |
| `root_index` | `int` | 当前物理 root 在 Dataset 内的索引 |
| `episode_index` | `int` | 当前 Episode 标识 |
| `task_index` | `int` | anchor 行的任务索引 |
| `normalization_id` | `str` | 选择该 sample normalization stats 的数据集名称 |
| `source_fps` | `float` | 落盘源数据 FPS |
| `rgb_rate_hz` | `float` | 输出 RGB rate |
| `action_steps_per_rgb_frame` | `int` | 帧内 Action 子步数 K |
| `action_rate_hz` | `float` | `rgb_rate_hz * K` |
| `anchor_timestamp` | scalar `float64` | anchor 的物理时间 |
| `anchor_rgb_index` | `int` | anchor 在目标 RGB grid 中的 Episode 内索引 |

该 ABI 只描述单样本。PyTorch `DataLoader` 如何 shuffle、分布式分片、collate、prefetch
以及后续模块如何搬运或编码这些 tensor，均不属于本仓库。

## 4. 使用方式

### 4.1 YAML 配置

```yaml
schema_version: ngad_canonical_dataloader_v2

dataset:
  dataset_dirs:
    - name: libero
      path: /path/to/canonical/libero
      mask_path: /path/to/canonical/libero/mask.json
      normalization_stats_path: /path/to/stats/libero.json

    - name: hy_embodied
      path: /path/to/canonical/hy_embodied
      mask_path: /path/to/canonical/hy_embodied/mask.json
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

上述 `frame_ranges` 展开为 81 个具有真实时间意义的 RGB offset；对应输出
`video[81,6,3,256,256]`、`state[81,2,128]` 和 `action[81,2,128]`。

### 4.2 构造 Dataset

```python
from ngad_canonical_dataloader import build_dataset_from_yaml

dataset = build_dataset_from_yaml("configs/canonical.yaml")
sample = dataset[0]
```

### 4.3 读取指定 offset

例如读取 offset `-1` 对应的 RGB frame 及同一 frame 槽位下的 K 个 Action：

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

这里 `position` 是 tensor 存储位置，`-1` 才是带真实时间语义的 offset。Tokenizer、VAE
或模型适配层可以用同一接口读取任意已配置 offset，并依据 validity mask 决定哪些输入参与
后续计算：

```python
past = sample["frame_offsets"] < 0
anchor = sample["frame_offsets"] == 0
future = sample["frame_offsets"] > 0
```

如果特定模型需要 `[N*K,128]`，reshape 应由模型适配层完成；Loading ABI 始终保留
`state[N,K,128]` 和 `action[N,K,128]`，不创建 `action_history`、`recent_memory` 或
`long_memory` 等模型专用字段。
