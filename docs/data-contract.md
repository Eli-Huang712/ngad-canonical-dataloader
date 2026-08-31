# Canonical Data Contract and Transform Design

本文给出 `NGADCanonicalDataset` 的详细输入合同、统一时间轴、State/Action 构造、
normalization、TCP128D packing 和完整输出 ABI。仓库在完整 Input Pipeline 中的位置见
[architecture.md](architecture.md)。

## 1. 物理存储后端

`dataset_dirs[].path` 可以指向 dataset root，也可以直接指向 single-table root：

- dataset root：只枚举直接子目录中名称严格匹配 `table_\d{3}` 的 table，并按数字排序；
- single-table root：当 path 自身是 `table_\d{3}` 时，只读取 path 自身，不继续向内搜索。

两者是确定性正式拓扑，不是递归 fallback，也不存在 `table_name` 配置字段。Loader 不接受
root manifest、fragment 或 shard 拓扑。每个 table 的 `meta/info.json` 必须声明
`total_episodes` 和 `total_frames`，并使用相同逻辑核对 Episode metadata 和 frame count。

Episode metadata 的 `dataset_from_index`、`dataset_to_index` 和物理数据行的 `index` 都是
table-local，并在每张 table 中独立从 0 开始。Dataset 在这些物理索引之上建立统一的
全局窗口索引；这个对外 sample index 不会被写回或冒充落盘 row index。

每张 table 必须且只能发布下列一种 payload；backend 由这一唯一物理结构确定。

### 1.1 Lance + JPEG

```text
<dataset-root>/
└── table_000/
    ├── meta/
    │   ├── info.json
    │   ├── tasks.parquet
    │   └── episodes/
    │       └── *.parquet
    └── table_000.lance/
        ├── _transactions/
        ├── _versions/
        └── data/
            └── *.lance
```

该后端要求 `canonical_schema == "ngad_hy_canonical_lance_v2"`，并且
`<table-name>/<table-name>.lance/data/` 至少包含一个 `*.lance`。图像以 JPEG payload
存储在 Lance 行中。

### 1.2 LeRobot v3 Parquet + H.264

```text
<dataset-root>/
└── table_001/
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

该后端要求 `data/` 与 `videos/` 同时存在。实际 Parquet 和 MP4 路径由 `info.json` 的
`data_path`、`video_path` 模板确定；视频使用 H.264 编码。两个后端都必须通过 episode
metadata 提供 Episode 长度、table-local 数据 offset 和任务索引。

## 2. Canonical 输入字段

### 2.1 六路固定相机槽位

| 槽位 | Canonical key |
|---:|---|
| 0 | `observation.images.cam_head_left` |
| 1 | `observation.images.cam_head_right` |
| 2 | `observation.images.cam_left_wrist_left` |
| 3 | `observation.images.cam_left_wrist_right` |
| 4 | `observation.images.cam_right_wrist_left` |
| 5 | `observation.images.cam_right_wrist_right` |

顺序不可改变。Lance inline JPEG 的可用图像必须声明为 RGB `image[256,256,3]`，
Parquet/H.264 的可用图像必须声明为 RGB `video[256,256,3]`；物理缺失由 mask contract
声明，不允许按数据集名称猜测。

### 2.2 字段合同

| 字段 | dtype / shape | 语义 |
|---|---|---|
| 六路 `observation.images.*` | `image[256,256,3]` 或 `video[256,256,3]` | backend 对应的 RGB 图像；物理缺失由 mask 声明 |
| `observation.state` | `float32[20]` | 双臂 absolute TCP，reshape 为 `[2,10]` |
| `action` | `float32[20]` | Canonical schema 字段；训练监督不读取它，而是由 state window 重算 |
| `observation.tactile.values` | `float32[4,3,25,6]` | 触觉值；可由 mask 声明缺失 |
| `observation.tactile.dt` | `float32[4,3]` | 触觉时间差；可由 mask 声明缺失 |
| `timestamp` | `float64[1]` | Episode 内物理时间 |
| `frame_index` | `int64[1]` | Episode 内帧索引 |
| `episode_index` | `int64[1]` | Episode 标识 |
| `index` | `int64[1]` | 数据集全局帧索引 |
| `task_index` | `int64[1]` | 当前 anchor 使用的任务索引 |

每条手臂的 TCP10 顺序固定为 absolute XYZ `[3]`、rotation matrix 前两行按 row-major
展开的 absolute Rot6D `[6]`、absolute gripper openness `[1]`。双臂顺序是左臂、右臂。

`observation.state` 和五个 identity 字段必须物理存在。相机、tactile 和落盘 `action`
是否物理存在，由每个数据集自己的 mask JSON 明确声明。

## 3. Field Mapping、Mask 与 Normalization 输入

每个 `dataset_dirs` 条目必须严格只包含 `name`、`path` 和
`mask_and_mapping_path`。后者同时定义 `field_mapping`、`field_mask`、state/action 的
`element_mask[20]`，以及共享的 `image_pixel_mask` NPZ 路径和 key。

`dataset.normalization_stats_path` 是必填字段，并且是 sample mode 的唯一选择器：

- 非空字符串：canonical mode。LIBERO、Hy、UMI 和其他 canonical table 共用这一份
  global normalization；Dataset 只加载一次并只构造一个 `CanonicalTCPTransform`；
- `null`：video-only mode。不读取 normalization 文件，不创建 TCP transform，不插值或
  normalization state/action，也不生成 dummy stats 或伪造 TCP128。

各 table 的原始 `meta/stats.json` 不是正式时间轴上重构的 anchor-relative Action 统计，
禁止直接作为 canonical global normalization 输入。

`mask_and_mapping_path` 指向的 JSON 顶层必须严格为：

```json
{
  "dataset": "hy_embodied",
  "field_mapping": {
    "observation.images.cam_head_left": "observation.images.cam_head",
    "observation.images.cam_left_wrist_left": "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist_left": "observation.images.cam_right_wrist"
  },
  "field_mask": {},
  "element_mask": {},
  "image_pixel_mask": {}
}
```

`field_mapping` 的方向固定为 `canonical key -> physical storage key`。未出现在
`field_mapping` 中的可用字段采用 canonical 与 physical 同名语义；只要二者不同，就必须
显式映射。`field_mask` 和 `element_mask` 始终使用 canonical key。`field_mask=false`
的字段禁止出现在 `field_mapping` 中；映射目标未出现在 table 的
`meta/info.json.features` 时，Dataset 初始化立即报错。

Lance backend 在实际取列时把映射后的 dotted physical key 转成 underscore column，例如
`observation.images.cam_head` → `observation_images_cam_head`；LeRobot backend 直接用映射后的
相机 key 解析 episode video metadata 和 `video_path`。

`field_mask` 必须覆盖六路相机、state、action、两个 tactile 字段和五个 identity 字段。
不可用的相机输出黑图，同时 `camera_mask=False`、`image_pixel_mask=False`；不可用的 tactile
字段输出零 tensor，同时对应 `tactile_field_mask=False`。

Normalization JSON 使用 `ngad_canonical_tcp_v1` schema，并提供双臂独立的：

```text
state_xyz_min:    [2,3]
state_xyz_max:    [2,3]
action_xyz_scale: [2,3]
```

Dataset 不在训练期间计算或更新统计量。多个数据集混合时，所有合法 anchor 按窗口数量自然
组成同一个全局索引；每个 sample 使用其所属数据集的 mask 和 normalization stats。

## 4. 统一时间轴

### 4.1 Semantic offset

Dataset 不区分 long、history、current 或 future。YAML 中的 `frame_ranges` 直接描述相对
当前 anchor 的 RGB offset 闭区间，并展开为唯一、按时间递增的 `frame_offsets[N]`。

- `frame_offset`：相对 anchor 的语义时间坐标，物理时间差是
  `frame_offset / rgb_rate_hz`；
- `position`：该 offset 在输出 tensor 中的存储位置；
- `source_frame_index`：抽到具体 anchor 后，在当前 Episode 中实际读取的物理帧。

在 10 Hz 配置中，offset `-1` 表示 anchor 之前 `0.1s`，不是 Python tensor 的倒数第一个
元素。固定布局在 Dataset 初始化时建立，具体 source index 和 timestamp 则在
`__getitem__()` 抽到 anchor 后实例化。

RGB 只读取真实 source frame，不合成图像。每个 source FPS 必须不低于 `rgb_rate_hz`，且
必须是 `rgb_rate_hz` 的整数倍。

### 4.2 帧内 K 个 State/Action 子步

Action rate 由整数倍数确定：

```python
K = action_steps_per_rgb_frame
action_rate_hz = rgb_rate_hz * K
```

State/Action 与 RGB 共用 frame 轴，并额外保留帧内 K 维：

```text
video[i]  : [6,3,256,256]
state[i]  : [K,128]
action[i] : [K,128]
```

设 `frame_offsets[i] = f`，帧内位置为 `k in [0,K)`：

```python
frame_time = anchor_time + f / rgb_rate_hz
action_time = frame_time - (K - 1 - k) / (rgb_rate_hz * K)
```

因此每个 RGB frame 对应其前一个 RGB 周期
`(frame_time - 1 / rgb_rate_hz, frame_time]` 内的 K 个 State/Action。以 10 Hz RGB、
`K=2` 为例：

| frame offset | `video[i]` | substep 0 | substep 1 |
|---:|---:|---:|---:|
| `-2` | `-0.20s` | `-0.25s` | `-0.20s` |
| `-1` | `-0.10s` | `-0.15s` | `-0.10s` |
| `0`（anchor） | `0.00s` | `-0.05s` | `0.00s` |
| `1` | `0.10s` | `0.05s` | `0.10s` |
| `2` | `0.20s` | `0.15s` | `0.20s` |

第二个子步与当前 RGB frame 同时刻，第一个位于前一 RGB frame 与当前 frame 之间。
对于稀疏 `frame_ranges`，Dataset 不填充 range 之间未请求的时间空洞，也不把 State/Action
展平成 `[N*K,128]`。

所有查询都限制在当前 Episode 内。越过 Episode 边界的位置保留固定 tensor 槽位，但
对应 `frame_valid`、`action_valid`、camera/pixel mask 为 false，不读取相邻 Episode，也
不按 `task_index` 切分 Episode。sample 的 prompt 暂时使用 anchor 行的 `task_index`。

## 5. State / Action 构造

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

- **XYZ**：对 lower/upper source state 做一阶线性插值；
- **Rot6D**：先投影为合法旋转矩阵，再转为 quaternion，执行 shortest-path SLERP，最后
  转回旋转矩阵前两行的 row-major Rot6D；
- **gripper openness**：在 absolute `[0,1]` 开度空间做一阶线性插值。

每个目标时间点由 source FPS 与 `action_rate_hz` 计算 lower source index、upper source
index 和插值比例。Episode 外的 target index 会被 clamp 以保证读取安全，但最终值通过
`action_valid[N,K]` 置零并排除。

必须先插值 absolute TCP，再计算 relative pose。这样 XYZ、openness 只在线性空间插值，
rotation 只在 SO(3) 上做一次 SLERP，不会在 Rot6D、relative pose 或 normalization 结果上
再次插值。

### 5.2 Fixed-anchor Relative Action

State 直接保留 `absolute_state_grid` 的 absolute pose。Action 对左右手分别使用当前 sample
的 offset `0` absolute state 作为固定 anchor：

```python
p_relative = R_anchor.T @ (p_t - p_anchor)
R_relative = R_anchor.T @ R_t
gripper_action = absolute_openness_t
```

- relative XYZ 在该手臂 anchor TCP 的局部坐标系中表达；
- relative rotation 表达从 anchor orientation 到目标 orientation 的旋转；
- 所有过去和未来 Action 都相对于同一个 offset `0` anchor，不是相邻时刻 delta；
- gripper 不计算 relative difference，继续保留目标时刻的 absolute openness；
- 左右臂分别使用各自的 anchor，不共享 TCP frame。

### 5.3 Global Normalization

本节仅适用于 canonical mode；video-only mode 完全跳过本节和 TCP128 构造。

| 特征 | State | Action |
|---|---|---|
| XYZ | 每只手臂、每个轴按 min/max 映射到 `[-1,1]`；越界值最终 clamp 到 `[-5,5]` | 每只手臂、每个轴除以对称 `action_xyz_scale`；保持零中心，不做 `[-1,1]` clamp |
| Rot6D | 不做统计归一化 | 不做统计归一化 |
| gripper openness | clamp 到 `[0,1]` | 保留目标 absolute openness，并 clamp 到 `[0,1]` |

部署侧可以调用 `dataset.denormalize_action(action)`，使用唯一 global scale 把 Action
TCP128 的 active relative XYZ 乘回物理尺度；Rot6D 和 openness 保持不变，保留槽位保持为零。

### 5.4 TCP20 → TCP128D

| TCP128 index（Python slice） | 内容 | State 语义 | Action 语义 |
|---|---|---|---|
| `0:3` | 左臂 XYZ | absolute | anchor-relative |
| `3:9` | 左臂 Rot6D | absolute | anchor-relative |
| `9:10` | 左夹爪 openness | absolute | absolute |
| `10:13` | 右臂 XYZ | absolute | anchor-relative |
| `13:19` | 右臂 Rot6D | absolute | anchor-relative |
| `19:20` | 右夹爪 openness | absolute | absolute |
| `20:128` | 保留槽位 | 零且 masked | 零且 masked |

```text
observation_state_element_mask[20] → state_feature_mask[128]
action_element_mask[20]            → action_feature_mask[128]
```

完成 TCP128 packing 后，active `0:20` 乘以相应 feature mask；保留维 `20:128` 固定为零且
feature mask 为 false。最后使用共同的 `action_valid[N,K]` 清零 Episode 边界外的整个
State/Action time slot。

## 6. 完整输出 ABI

设 `N = len(frame_offsets)`、`K = action_steps_per_rgb_frame`：

两种模式共同返回：

```python
{
    "video":                         float32[N, 6, 3, 256, 256],
    "frame_offsets":                 int64[N],
    "source_frame_indices":          int64[N],
    "frame_timestamps":              float64[N],
    "frame_valid":                   bool[N],

    "camera_mask":                   bool[N, 6],
    "image_pixel_mask":              bool[N, 6, 256, 256],
    "prompt":                        str,
    "data_info":                     dict,
}
```

Canonical mode 另外返回：

```python
{
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
}
```

Video-only mode 不返回上面的任何字段，尤其不返回 state/action、Action 时间轴或
feature/element mask。

`video` 已从 `uint8[0,255]` 转为 `float32[-1,1]`。State 与 Action 共用
`action_step_offsets[N,K]`、`action_timestamps[N,K]` 和 `action_valid[N,K]`，不重复输出
另一套 State 时间 metadata。

`data_info` 字段：

| 字段 | 类型 / shape | 含义 |
|---|---|---|
| `sample_mode` | `str` | `"canonical"` 或 `"video_only"` |
| `img_hw` | `float32[2]` | `[height,width]` |
| `aspect_ratio` | scalar `float32` | `width / height` |
| `root_index` | `int` | 当前 physical table 在 Dataset 内的索引 |
| `episode_index` | `int` | 当前 Episode 标识 |
| `task_index` | `int` | anchor 行的任务索引 |
| `source_fps` | `float` | 落盘源数据 FPS |
| `rgb_rate_hz` | `float` | 输出 RGB rate |
| `action_steps_per_rgb_frame` | `int` | 帧内 State/Action 子步数 K |
| `action_rate_hz` | `float` | `rgb_rate_hz * K` |
| `anchor_timestamp` | scalar `float64` | anchor 的物理时间 |
| `anchor_rgb_index` | `int` | anchor 在目标 RGB grid 中的 Episode 内索引 |
