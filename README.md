# ngad-canonical-dataloader

从 `NGADv1pp` 的 `eli/ngad-canonical-tcp@34eef9b` 机械式抽取的独立 loading package。

当前版本保留原实现中的：

- canonical Lance + JPEG 与 LeRobot v3 Parquet + MP4 读取；
- 以 anchor 为基准的 relative TCP action；
- absolute/relative TCP normalization；
- TCP20 到 TCP128 packing 与 feature mask；
- RGB/action 双频窗口和 episode-bounded Memory 索引。

当前版本不包含：

- `diffusion` Dataset registry；
- PyTorch `DataLoader`、`DistributedSampler` 和训练入口；
- `_extract_wam_batch()`；
- Tokenizer、VAE、flow noise/target 或 `SanaWAM.forward()`。

## 公共入口

```python
from ngad_canonical_dataloader import build_dataset_from_yaml

dataset = build_dataset_from_yaml("configs/canonical.yaml")
sample = dataset[0]
```

YAML 根节点严格包含 `schema_version` 和 `dataset`，完整模板见
[`configs/canonical.yaml`](configs/canonical.yaml)。当前只有一个 public Dataset，不再通过
`type` 选择 LIBERO、Hy-Embodied 或 UMI 专用实现。每个 `dataset_dirs` 条目必须提供
`name`、`path`、`mask_path` 和 `normalization_stats_path`。

## Canonical 输入字段

- 六路固定顺序的 `observation.images.cam_*` RGB video，单帧 `[256,256,3]`；
- `observation.state`: `float32[20]` absolute dual-arm TCP；
- `action`: `float32[20]`，当前仅验证存在，训练 action 仍从 state 重建；
- `observation.tactile.values`: `float32[4,3,25,6]`；
- `observation.tactile.dt`: `float32[4,3]`；
- `timestamp`、`frame_index`、`episode_index`、`index`、`task_index`。

`mask_path` 指向每个数据集自己的 canonical mask JSON。该 sidecar 的 `field_mask` 决定
相机和 tactile 字段是否真实存在，`element_mask` 提供 state/action 的逐元素 20D validity，
`image_pixel_mask` 指向由全部有效相机共享的 `bool[256,256]` NPZ mask。字段不可用时，
Dataset 根据 sidecar 确定性补黑图或零 tensor，不按 backend 猜测，也不提供 fallback。
sidecar 的 `dataset` 必须与 root 的 `name` 相同；其中 pixel-mask path 相对 mask JSON
所在目录解析。

Dataset sample 保留 anchor 时刻的两个 canonical tactile tensor，并输出：

- `observation_state_element_mask: bool[20]`；
- `action_element_mask: bool[20]`；
- `proprio_feature_mask/action_feature_mask/action_history_feature_mask: bool[128]`；
- `camera_view_mask: bool[6]`；
- `tactile_field_mask: bool[2]`，顺序为 tactile values、tactile dt；
- 主视频和 Memory 对应的 pixel/temporal mask。

旧 `arm_mask[2]` ABI 已删除。TCP128 的 `0:20` validity 直接保留 canonical element mask，
`20:128` 始终为 false。

本次抽取只建立独立修改基线，不声称已经形成最终解耦 ABI。后续重写应从
`NGADCanonicalDataset.__init__()` 和 `NGADCanonicalDataset.__getitem__()` 开始。

## Loading / Sample Construction 设计对照

对照基线：飞书《[Dataloader(Data Input Pipeline)](https://axvb1jzxlgb.feishu.cn/wiki/Aabmwi8ZniGSzakeU0LcdJXanuf?from=from_copylink)》的
“2. Loading / Sample Construction”章节；代码基线为本仓库
`main@3eb4cf03caba28aa65c4ae39b7203544b922fd92`。

状态含义：

- **已实现**：小仓库中存在与文档语义一致的生产代码路径；
- **部分实现**：主体路径存在，但字段合同、有效性来源或严格校验与文档不完全一致；
- **明确排除**：根据当前小仓库职责主动不提供，不作为待修缺陷。

以下结论是逐项静态代码审查结果，不等同于真实 Lance/LeRobot 数据运行证明。当前
`tests/` 覆盖 public import、YAML 解析、TCP128 packing、element mask 和基础窗口 shape，
尚未覆盖两个物理 backend 的 `Dataset.__init__()`/`Dataset.__getitem__()` 端到端读取。

### 2.1 数据入口与物理存储后端

| 飞书验收项 | 状态 | 小仓库实现与差异 |
|---|---|---|
| 正式入口为 `NGADCanonicalDataset`；`dataset_dirs` 可包含多个具名 root | **已实现** | 每个 root 必须有唯一 `name`，以及 `path`、`mask_path`、`normalization_stats_path`。 |
| 自动识别 Lance table/fragment + JPEG 与 LeRobot v3 Parquet + MP4 | **已实现** | `canonical.py::_detect_backend()`、`_take_lance_rows()`、`_take_lerobot_v3_rows()`、`_decode_lance_camera()`、`_decode_video_camera()`。 |
| 固定六路双目相机和固定顺序 | **已实现** | 当前 canonical key 以代码中的 `observation.images.cam_*` 命名为准，不采用飞书旧字段拼写。 |
| `observation.state=[20]`，按 `[2,10]` 解释；每臂为 absolute XYZ + Rot6D + absolute openness | **已实现** | `_validate_features()` 校验 `[20]`，`_reshape_state_window()` reshape 为 `[T,2,10]`。 |
| 读取并校验 `timestamp`、`frame_index`、`episode_index`、`index`、`task_index` | **已实现** | 两个 `_take_*_rows()` 读取这些字段，并校验行身份；`_validate_sample_timestamps()` 校验时间轴。 |
| 读取静态 `bool[256,256]` image pixel mask | **已实现** | `_load_mask_contract()` 读取 `mask_path`；`_load_pixel_mask()` 按 JSON 指定的 NPZ path/key 读取单一 mask，并仅应用到 `field_mask=true` 的相机。 |
| 落盘 `action=[20]` 只校验存在，不作为训练监督读取 | **已实现** | `_validate_features()` 校验 action shape；两个 row reader 均不读取 action，监督由 state 重建。 |
| LIBERO、Hy-Embodied、UMI raw reader 作为过渡 adapter | **明确排除** | 小仓库只提供 canonical public reader；source-specific adapters 保持删除，原始数据必须先在仓库外完成 canonical 转换。 |
| tactile 字段遵循真实 Dataset 合同 | **已实现** | tactile shape 以 canonical Dataset 为准；`field_mask=true` 时读取，false 或物理缺失时返回固定 shape 零 tensor，并通过 `tactile_field_mask[2]` 标记。 |

### 2.2 `Dataset.__init__()`：建立统一全局索引

| 飞书验收项 | 状态 | 小仓库实现与差异 |
|---|---|---|
| 拒绝旧 `target_fps`、stride、`history_chunks` 和全局 stats 参数 | **已实现** | 列出的旧字段会被拒绝，YAML 拒绝未知 key；直接调用 Dataset 时允许其他 `**extra`，这是当前明确保留的构造行为。 |
| root 可直接指向物理 root，也可指向 `table_*/fragments/*` 或 `shard_*` 集合目录 | **已实现** | `__init__()` 展开 `meta/info.json`、`table_*/fragments/*/meta/info.json` 和 `shard_*/meta/info.json`。 |
| 读取 info、task/episode metadata、数据行范围和每路视频时间范围 | **已实现** | `_read_metadata()` 分别读取 Lance 与 LeRobot v3 metadata，并校验 episode offset 和视频时间范围。 |
| 按物理文件特征选择 backend | **已实现** | `_detect_backend()` 返回唯一的 `lance_jpeg` 或 `lerobot_v3`，无法识别时直接报错。 |
| 以 episode 为唯一时序边界做 train/validation split；task 不切 episode | **已实现** | `split_episode_indices()` 只接收 episode id；后续窗口索引不以 task 为边界。 |
| 每个 named dataset 读取独立外部 normalization stats，并建立独立 transform | **已实现** | 每个 `name` 对应一个 `CanonicalTCPTransform`；stats 不在训练期间计算。 |
| 加载 pixel、camera 和 TCP element validity | **已实现** | 所有 validity 均来自必填 `mask_path`：camera 读取 `field_mask`，TCP 读取 state/action `element_mask[20]`；`_backend_masks()` 已删除且没有 fallback。 |
| 将 episode 映射到目标 RGB/action 网格并计算 anchor 数量 | **已实现** | `_target_episode_length()` 分别建立两套目标网格；每个目标 RGB tick 都可作为 anchor，尾部不足由 padding mask 表达。 |
| 用 `_episode_window_ends` 建前缀和，以 `_locate_window()` 反查 root/episode/anchor | **已实现** | 全局整数 index 可映射回对应 episode 和 episode 内 anchor。 |
| 混合数据不加权，按合法窗口数量自然采样 | **已实现** | 所有 root/episode 进入同一个全局前缀和；没有 dataset weight 或 source sampler。 |

### 2.3 双频率时间合同

| 飞书验收项 | 状态 | 小仓库实现与差异 |
|---|---|---|
| 分别配置 `target_rgb_fps` 与 `target_action_fps`，且 action/RGB 比率为正整数 | **已实现** | YAML 与 Dataset 构造函数均使用两个变量；`__init__()` 和窗口函数均校验整数比率。 |
| source FPS 不低于目标 RGB FPS，且是其整数倍；RGB anchor 必须是真实帧 | **已实现** | `__init__()` 拒绝低 FPS/非整数倍；`_source_indices()` 只映射真实 source frame，不插值 RGB。 |
| absolute TCP state 可重采样到 action grid | **已实现** | `_state_interpolation_indices()` 为 current/future/history target tick 生成 lower/upper/fraction。 |
| XYZ 与 openness 线性插值；Rot6D 经 SO(3) shortest-path SLERP | **已实现** | `interpolate_canonical_tcp()` 调用 quaternion SLERP，之后转回 row-major Rot6D。 |
| 默认 10 Hz RGB、20 Hz action、17 帧视频、32 个 future target；slot 0 是 `t0+0.05s` | **已实现** | `wam_window_indices()` 的 action 从 `anchor_action_index + 1` 开始，RGB 从 anchor 开始。 |

### 2.4 `Dataset.__getitem__()`：构造单个 sample

| 飞书验收项 | 状态 | 小仓库实现与差异 |
|---|---|---|
| global index 定位 episode 与 RGB anchor | **已实现** | `_locate_window()`。 |
| 构造 17 个 RGB、32 个 future action target 及尾部 padding mask | **已实现** | `wam_window_indices()`。 |
| 构造 strict-past recent/long RGB 与 reached-state history，且不跨 episode | **已实现** | `wam_memory_indices()`。 |
| 将目标 RGB grid 映射到真实 source frame | **已实现** | `_source_indices()`。 |
| 为 current/future/history state 生成插值 bracket | **已实现** | `_state_interpolation_indices()`。 |
| 按 backend 读取结构化行并解码六路相机 | **已实现** | 两组 `_take_*`/`_decode_*` 方法。 |
| 校验实际 timestamp 与 metadata FPS 一致 | **已实现** | `_validate_sample_timestamps()`，容差固定为 `1e-4` 秒。 |
| 在 action grid 上重建 absolute TCP state | **已实现** | `interpolate_canonical_tcp()`。 |
| 由 absolute state 生成 proprio、future action 和 action history | **已实现** | `CanonicalTCPTransform.encode_proprio()` 与 `encode_action_targets()`。 |
| 输出六视角 tensor，并组合 camera/pixel/temporal mask | **已实现** | `camera_view_mask[6]` 直接来自 sidecar；不可用相机无需物理落盘，Dataset 补 `-1.0` 黑图且对应 pixel mask 全 false。 |
| 使用 anchor 行 `task_index` 构造 prompt，并返回 `data_info` | **已实现** | 正式 canonical 两个 backend 的 `episode_tasks` 均为空，因此实际使用 anchor `task_index`。 |
| episode 尾部 pad、开头 Memory 补黑/补零并置 validity=false，绝不借相邻 episode | **已实现** | future index clamp 到末帧；Memory index clamp 到首帧后再用 mask 和 `-1.0` 黑图/零 TCP 排除。 |
| task 不是时序边界；同一 episode 的窗口和 Memory 可以跨 task | **已实现** | 索引逻辑只使用 episode 长度；prompt 只取当前 anchor 的 task。飞书提到的 UMI raw adapter 差异在本仓库不适用。 |

### 2.5 Relative TCP、normalization 与 TCP128

| 飞书验收项 | 状态 | 小仓库实现与差异 |
|---|---|---|
| 不读取落盘 action；future target 相对 current-frame anchor 计算 | **已实现** | `tcp_target_relative_to_anchor()` 实现 `R_anchor^T (p_target-p_anchor)` 与 `R_anchor^T R_target`；gripper 保留 target absolute openness。 |
| action history 是过去 reached state 相对当前 anchor，不是历史 controller command | **已实现** | 过去 absolute state 与 future target 共用 `encode_action_targets()`，anchor 始终是 current state。 |
| normalization 只读每个数据集的外部 JSON；训练时不扫描、不更新、无 stats fallback | **已实现** | `_normalization_transform()` 校验外部 JSON；构造 root 时缺失/非法即报错。 |
| proprio XYZ 做 per-arm min/max；relative XYZ 除对称 scale；Rot6D 不变；openness clamp `[0,1]` | **已实现** | `action.py` 中 absolute/relative normalization 与飞书一致，absolute XYZ 输出最多保留到 `[-5,5]`。 |
| TCP20 写入 TCP128 的 `0:20`，保留维 `20:128` 为零 | **已实现** | `pack_dual_arm_tcp()`。 |
| canonical element mask 映射到 TCP128 feature mask | **已实现** | 旧 `arm_mask[2]` 已删除；`element_mask_to_feature_mask()` 原样保留 state/action 的 20D validity 到 TCP128 `0:20`，`20:128` 为 false。 |

### 2.6 Memory 时间合同

| 飞书验收项 | 状态 | 小仓库实现与差异 |
|---|---|---|
| recent RGB 为严格过去 24 帧 | **已实现** | `wam_memory_indices()` 生成 `anchor-24 ... anchor-1`。 |
| long RGB 为 5 个 slot × 8 帧，slot anchor 间隔 50 个 RGB frame，且位于 recent 之前 | **已实现** | long anchor 先限制在 `anchor - recent_memory_frames - 1` 之前，再按 50 帧网格向过去取 5 个 slot。 |
| action history 为严格过去 10 个 action-grid reached-state tick | **已实现** | 生成 `anchor_action-10 ... anchor_action-1`，随后从 absolute state 插值并转换为 current-anchor relative TCP128。 |
| episode 开头不足时保持固定 shape，以 validity mask 表达缺失 | **已实现** | recent/long/history 同时返回 clamp 后 index 与独立 validity；`__getitem__()` 用黑图/零值和 mask 构造固定 shape。 |

### 汇总结论

按上表拆分的 47 个验收项中，当前职责范围内为：**46 项已实现、1 项明确排除、0 项部分实现**。
核心的双 backend 读取、统一窗口索引、10/20 Hz 时间合同、absolute TCP 插值、fixed-anchor
relative TCP128、外部 normalization、canonical tensor mask 和 episode-bounded Memory 均已存在。
当前相机 key 与 tactile shape 以真实 canonical Dataset 为准；raw adapters 保持删除；直接构造
Dataset 时继续允许未知 `**extra`。真实 Lance/LeRobot root 的端到端测试按当前决定暂不执行，
因此“已实现”仍表示静态代码路径存在，不代表真实数据运行已经通过。

## 来源映射

| 独立仓库 | NGADv1pp 来源 |
|---|---|
| `ngad_canonical_dataloader/datasets/canonical.py` | `ngad/datasets/canonical.py` |
| `ngad_canonical_dataloader/action.py` | `ngad/utils/rotation.py` + `ngad/utils/tcp.py` |
| `ngad_canonical_dataloader/windows.py` | data-only functions from `ngad/utils/wam.py` |
| `ngad_canonical_dataloader/memory.py` | `ngad/utils/wam_memory.py` |

后续修改已删除 source-specific Dataset，并增加正式 canonical schema 与 YAML 配置入口。
