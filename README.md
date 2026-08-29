# ngad-canonical-dataloader

`ngad-canonical-dataloader` 是独立的 canonical Loading / Sample Construction 模块。
它从 canonical 数据集读取样本，以某个真实 RGB frame 为 anchor，构造统一的
RGB/State/Action 时间轴、absolute State TCP128、anchor-relative Action TCP128、
normalization、validity mask 和 metadata。

本仓库的边界到 `Dataset.__getitem__()` 为止：不包含 PyTorch `DataLoader`、GPU transfer、
Tokenizer、VAE、flow target/noise construction 或模型代码。后续模块只消费本仓库定义的
sample ABI，不需要知道底层使用 Lance+JPEG 还是 LeRobot v3 Parquet+H.264。

## 1. 快速开始

### 1.1 编写 YAML 配置

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

### 1.2 构造 Dataset

```python
from ngad_canonical_dataloader import build_dataset_from_yaml

dataset = build_dataset_from_yaml("configs/canonical.yaml")
sample = dataset[0]
```

### 1.3 读取指定 offset

例如读取 offset `-1` 对应的 RGB frame，以及同一 frame 槽位下的 K 个 State/Action：

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

## 2. 总体架构与边界

### 2.1 完整 Input Pipeline

完整训练输入链路分为五层：

```text
Loading / Sample Construction
Dataset.__init__() + Dataset.__getitem__()
        ↓
Batching
DistributedSampler + PyTorch DataLoader
        ↓
Transfer
_extract_wam_batch()
        ↓
Model Input Preparation
VAE encode + text encode + flow noise/target construction
        ↓
Model
SanaWAM.forward()
```

本仓库只实现第一层 **Loading / Sample Construction**。其输出是 CPU 上单样本的
backend-neutral ABI；从 Batching 开始的所有环节均由调用方或模型主仓库负责：

| 层级 | 核心职责 | 是否属于本仓库 |
|---|---|---|
| Loading / Sample Construction | 读取 canonical 数据、建立 anchor/window、时间对齐、构造 State/Action/mask/metadata | 是 |
| Batching | shuffle、分布式分片、多进程读取、prefetch、collate、pinned memory | 否 |
| Transfer | batch ABI 检查、CPU→GPU、dtype/device 转换 | 否 |
| Model Input Preparation | 六视角 VAE encode、text encode、flow noise/target 构造 | 否 |
| Model | `SanaWAM.forward()` 及 loss | 否 |

因此本仓库不导入模型代码，也不要求模型了解 Lance、Parquet、JPEG 或 H.264。调用方只需
依据第 6 节的 sample ABI 进行 batching 和后续处理。

### 2.2 本仓库内部数据流

```text
YAML
  │
  ▼
config.py
严格校验配置并构造 NGADCanonicalDataset
  │
  ▼
Dataset.__init__()
发现 physical roots → 选择 backend → 读取 task/episode metadata
→ 加载 mask/stats → 建立统一全局 anchor 索引与 TimelineLayout
  │
  ▼
Dataset.__getitem__()
global index → root/episode/anchor → 解析 RGB/Action 时间点
→ table/image backend 按需读取 → absolute TCP 插值
→ absolute State + anchor-relative Action → normalization → TCP128
→ 组合 temporal/camera/pixel/feature mask 与 metadata
  │
  ▼
单样本 canonical ABI
video + state + action + masks + timestamps + prompt + data_info
```

`Dataset.__init__()` 只建立 metadata、Episode 和 anchor 的索引视图，不预加载全部视频。
`Dataset.__getitem__()` 抽到具体 anchor 后才把语义 offset 实例化为当前 Episode 的 timestamp
与 source index，并只读取该 sample 需要的数据行和图像。

### 2.3 文件职责

| 文件 | 职责 |
|---|---|
| `pyproject.toml` | 定义 Python 包、运行依赖、canonical/test 可选依赖和 pytest 发现路径 |
| `ngad_canonical_dataloader/__init__.py` | 暴露 `NGADCanonicalDataset`、配置加载和 YAML 构造入口 |
| `ngad_canonical_dataloader/config.py` | 严格校验版本化 YAML，并把配置转换为 Dataset 构造参数 |
| `ngad_canonical_dataloader/datasets/__init__.py` | 只导出唯一 public Dataset，隔离具体实现文件 |
| `ngad_canonical_dataloader/datasets/canonical.py` | 实现 `NGADCanonicalDataset.__init__()`、`__getitem__()`、全局 Episode/anchor 索引和最终 sample 组装 |
| `ngad_canonical_dataloader/backends/__init__.py` | 根据物理 root 特征选择唯一的 table/image backend 组合 |
| `ngad_canonical_dataloader/backends/table.py` | 读取 Lance 或 LeRobot v3 Parquet 的 task、episode metadata 与 canonical 行 |
| `ngad_canonical_dataloader/backends/image.py` | 将 Lance JPEG payload 或 LeRobot v3 H.264 视频解码为统一的 `uint8[T,3,256,256]` RGB tensor |
| `ngad_canonical_dataloader/windows.py` | 展开语义 `frame_ranges`，建立 offset→position 映射和 Episode 内 RGB/Action validity |
| `ngad_canonical_dataloader/action.py` | absolute TCP 插值、Rot6D/SO(3) 转换、relative pose、normalization 和 TCP20→TCP128 packing |
| `configs/canonical.yaml` | 可直接复制修改的统一时间轴与多数据集配置模板 |
| `tests/` | 覆盖配置、时间轴、backend 选择、TCP128、mask 和 sample ABI 的定向测试 |

### 2.4 明确边界

本仓库负责：

- 读取已经满足 canonical schema 的物理数据；
- 建立按窗口数量自然混合的全局 anchor 索引；
- 构造统一 RGB/State/Action 时间轴及 Episode validity；
- 从 absolute state 插值并生成 absolute State、anchor-relative Action；
- 应用外部 normalization stats、mask 和 TCP128 packing；
- 输出稳定、与物理 backend 解耦的单样本 ABI。

本仓库不负责：

- 把 LIBERO、Hy-Embodied、UMI 等 raw 数据转换成 canonical 数据；
- 生成或训练时统计 normalization stats、field/element/pixel mask；
- `DistributedSampler`、PyTorch `DataLoader`、collate、prefetch 或 pinned memory；
- `_extract_wam_batch()`、GPU transfer 或 dtype/device 策略；
- Tokenizer、VAE、camera/token attention、flow target/noise、loss 或模型 forward；
- 将 `[N,K,128]` reshape 成某个特定模型的私有 ABI。

## 3. 输入模块

### 3.1 公共入口与存储后端

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

### 3.2 Canonical 字段合同

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

### 3.3 Mask 与 normalization 输入

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

## 4. 时间轴对齐

### 4.1 一个 anchor-relative 时间轴

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

### 4.2 Action 的帧内 K 维

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

所有查询都限制在当前 Episode 内。越过 Episode 边界的 RGB/State/Action 位置保留固定
tensor 槽位，但对应 `frame_valid`、`action_valid`、camera/pixel mask 为 false，不读取
相邻 Episode，也不按 `task_index` 切分 Episode。sample 的 prompt 暂时使用 anchor 行的
`task_index`。

## 5. State / Action 构造与表示

完整构造顺序如下：

```text
落盘 absolute observation.state
        ↓  在 Action timestamp 上插值一次
absolute_state_grid [N,K,2,10]
        ├─────────────────────────────────────┐
        ↓                                     ↓
absolute normalization                 相对 offset 0 anchor
        ↓                                     ↓
State TCP20                            relative Action TCP20
        ↓                                     ↓
TCP128 packing                         relative normalization
                                              ↓
                                       TCP128 packing
```

State 和 Action 必须复用同一份 `absolute_state_grid`。Dataset 不读取落盘 `action` 作为训练
监督，也不会分别插值 State 和 Action。

### 5.1 Absolute TCP 插值

落盘 `action` 不作为监督。Dataset 在每个 Action timestamp 上重采样 absolute
`observation.state`：

- **XYZ**：对 lower/upper source state 做一阶线性插值；
- **Rot6D**：先把 row-major Rot6D 投影为合法旋转矩阵，再转为 quaternion，执行
  shortest-path SLERP，最后转回旋转矩阵前两行的 row-major Rot6D；
- **gripper openness**：在 absolute `[0,1]` 开度空间做一阶线性插值。

每个目标时间点先由 source FPS 与 `action_rate_hz` 计算 lower source index、upper source
index 和插值比例。插值发生在 absolute TCP 空间，输出双臂
`absolute_state_grid[N,K,2,10]`。Episode 外的 target index 会被 clamp 以保证读取安全，但
最终值通过 `action_valid[N,K]` 置零并排除。

采用“先插值 absolute TCP，再计算 relative pose”的原因是：XYZ、openness 只在线性空间
插值，rotation 只在 SO(3) 上做一次 SLERP；不会在已经经过 Rot6D 投影、relative pose 或
normalization 的表示上再次插值，从而避免两条输出链路产生额外数值偏差。

### 5.2 Fixed-anchor Relative Action

State 直接保留 `absolute_state_grid` 的 absolute pose。Action 则对左右手分别使用当前
sample 的 offset `0` absolute state 作为同一个固定 anchor。对任意目标时刻 `t`：

```python
p_relative = R_anchor.T @ (p_t - p_anchor)
R_relative = R_anchor.T @ R_t
gripper_action = absolute_openness_t
```

含义是：

- relative XYZ 在该手臂 anchor TCP 的局部坐标系中表达；
- relative rotation 表达从 anchor orientation 到目标 orientation 的旋转；
- 所有过去和未来 Action 都相对于同一个 offset `0` anchor，不是相邻时刻 delta；
- gripper 不计算 relative difference，继续保留目标时刻的 absolute openness。

relative rotation 最终重新编码为 row-major Rot6D。左右臂分别使用各自的 anchor，不共享
TCP frame。

### 5.3 Per-dataset Normalization

Normalization stats 由 `dataset_dirs[].normalization_stats_path` 外部提供。Dataset 不在训练
期间扫描或更新统计量；混合训练时，每个 sample 根据自己的 `normalization_id` 使用对应
数据集 stats。

每个数据集提供：

```text
state_xyz_min:    [2,3]
state_xyz_max:    [2,3]
action_xyz_scale: [2,3]
```

State 和 Action 分别处理：

| 特征 | State | Action |
|---|---|---|
| XYZ | 每只手臂、每个轴按 min/max 映射到 `[-1,1]`；越界值允许超出该区间，最终 clamp 到 `[-5,5]` | 每只手臂、每个轴除以对称 `action_xyz_scale`；保持零中心，不做 `[-1,1]` clamp |
| Rot6D | 不做统计归一化 | 不做统计归一化 |
| gripper openness | clamp 到 `[0,1]` | 保留目标 absolute openness，并 clamp 到 `[0,1]` |

因此 State 与 Action 虽然 shape 和时间点完全对齐，但 XYZ 的物理语义与 normalization
不同。部署侧可以调用 `dataset.denormalize_action(action, normalization_id)`，把 Action
TCP128 的 active relative XYZ 乘回对应 scale；Rot6D 和 openness 保持不变，保留槽位保持
为零。

### 5.4 TCP20 → TCP128D Packing 与 Mask

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
false：

```text
observation_state_element_mask[20] → state_feature_mask[128]
action_element_mask[20]            → action_feature_mask[128]
```

完成 TCP128 packing 后，active `0:20` 再乘以相应 feature mask，使无效 canonical
element 变为零；保留维 `20:128` 从创建时即固定为零且 feature mask 为 false。最后再使用
共同的 `action_valid[N,K]` 清零 Episode 边界外的整个 State/Action time slot。

最终生产 ABI 为：

```text
state  : float32[N,K,128]  # absolute TCP
action : float32[N,K,128]  # fixed-anchor relative pose + absolute openness
```

两者共用 `action_step_offsets[N,K]`、`action_timestamps[N,K]` 和
`action_valid[N,K]`，但分别保留自己的 element/feature mask。

## 6. 输出模块

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
