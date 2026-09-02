# Input Pipeline Architecture

本文说明 `ngad-canonical-dataloader` 在完整训练 Input Pipeline 中的位置、内部执行链路、
代码文件职责和明确边界。Canonical 字段、时间轴和 State/Action 数值合同见
[data-contract.md](data-contract.md)。

## 1. 完整 Input Pipeline

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
backend-neutral ABI；从 Batching 开始的所有环节均由调用方或模型主仓库负责。

| 层级 | 核心职责 | 是否属于本仓库 |
|---|---|---|
| Loading / Sample Construction | 读取 canonical 数据、建立 anchor/window、时间对齐、构造 State/Action/mask/metadata | 是 |
| Batching | shuffle、分布式分片、多进程读取、prefetch、collate、pinned memory | 否 |
| Transfer | batch ABI 检查、CPU→GPU、dtype/device 转换 | 否 |
| Model Input Preparation | 六视角 VAE encode、text encode、flow noise/target 构造 | 否 |
| Model | `SanaWAM.forward()` 及 loss | 否 |

因此本仓库不导入模型代码，也不要求模型了解 Lance、Parquet、JPEG 或 H.264。调用方只需
消费 README 定义的单样本 ABI，再自行完成 batching、transfer 和模型输入准备。

## 2. 仓库内部数据流

```text
YAML
  │
  ▼
config.py
严格校验配置并构造 NGADCanonicalDataset
  │
  ▼
Dataset.__init__()
解析 dataset/table/flat root → 按唯一物理 payload 选择 backend → 读取 task/episode metadata
→ 加载 per-root mask/mapping 和 global stats → 建立统一全局 anchor 索引与 TimelineLayout
  │
  ▼
Dataset.__getitem__()
global index → table/episode/anchor → 解析 RGB/State/Action 时间点
→ table/image backend 按需读取 → 组合 video/camera/pixel/frame metadata
→ canonical mode 才执行 absolute TCP 插值、relative Action、normalization 与 TCP128
  │
  ▼
单样本 ABI
video-only: video + camera/pixel/frame mask + prompt + data_info
canonical: 上述字段 + state + action + action/feature/element mask
```

`Dataset.__init__()` 只建立 metadata、Episode 和 anchor 的索引视图，不预加载全部视频。
它完成：

- 解析具名 `dataset_dirs`；
- 从 dataset root 的直接子目录枚举并排序 `table_NNN`、直接读取一张 single-table root，
  或把 `meta/info.json + data/ + videos/` 的 flat LeRobot v3 root 作为唯一 physical table；
  三种路径都不递归且没有 `table_name` 配置；
- 根据 table 中唯一的 Lance 或 Parquet/H.264 payload 选择物理 backend；
- 读取 task/episode metadata、source FPS、episode offsets 和视频时间范围；
- 加载每个数据集自己的 mask-and-mapping contract，并为所有混合数据加载唯一一份
  global normalization stats；
- 按 Episode 完成 train/validation split；
- 根据每个 Episode 的目标 RGB 长度建立全局 anchor 前缀和；
- 建立固定 `TimelineLayout` 和 offset→position 映射。

`Dataset.__getitem__()` 抽到具体 anchor 后才把语义 offset 实例化为当前 Episode 的 timestamp
与 source index，并只读取当前 sample 需要的数据行和图像。它完成：

- global index → physical table / Episode / anchor；
- 生成 RGB target index 和 Episode validity；
- 将 RGB target 映射到真实 source frame；
- 调用 table/image backend 读取并解码数据；
- 组合 frame、camera、pixel mask、prompt 和 metadata；
- video-only mode 不读取 sidecar 中无效的 State/tactile 字段；
- 仅在 canonical mode 为高频 state 生成插值 bracket，并构造 absolute State、
  fixed-anchor relative Action、normalization、TCP128 和对应 mask；
- 返回稳定的单样本 ABI。

## 3. 文件职责

| 文件 | 职责 |
|---|---|
| `pyproject.toml` | 定义 Python 包、运行依赖、canonical/test 可选依赖和 pytest 发现路径 |
| `ngad_canonical_dataloader/__init__.py` | 暴露 `NGADCanonicalDataset`、配置加载和 YAML 构造入口 |
| `ngad_canonical_dataloader/config.py` | 严格校验版本化 YAML，并转换为 Dataset 构造参数 |
| `ngad_canonical_dataloader/datasets/__init__.py` | 只导出唯一 public Dataset，隔离具体实现文件 |
| `ngad_canonical_dataloader/datasets/canonical.py` | 实现 `NGADCanonicalDataset.__init__()`、`__getitem__()`、全局 Episode/anchor 索引和最终 sample 组装 |
| `ngad_canonical_dataloader/backends/__init__.py` | 根据 table 的唯一物理 payload 选择 table/image backend 组合 |
| `ngad_canonical_dataloader/backends/table.py` | 按 canonical→physical mapping 读取 Lance 或 LeRobot v3 Parquet 的 task、episode metadata 与数据行，并恢复 canonical key |
| `ngad_canonical_dataloader/backends/image.py` | 按映射后的物理相机 key 将 Lance JPEG payload 或 LeRobot v3 H.264 视频解码为统一的 `uint8[T,3,256,256]` RGB tensor |
| `ngad_canonical_dataloader/windows.py` | 展开语义 `frame_ranges`，建立 offset→position 映射和 Episode 内 RGB/Action validity |
| `ngad_canonical_dataloader/action.py` | absolute TCP 插值、Rot6D/SO(3) 转换、relative pose、normalization 和 TCP20→TCP128 packing |
| `configs/canonical.yaml` | 可直接复制修改的统一时间轴与多数据集配置模板 |
| `tests/` | 覆盖配置、时间轴、backend 选择、TCP128、mask 和 sample ABI 的定向测试 |

## 4. 明确边界

本仓库负责：

- 读取已经满足 canonical schema 的物理数据；
- 建立按窗口数量自然混合的全局 anchor 索引；
- 构造统一 RGB/State/Action 时间轴及 Episode validity；
- 从 absolute state 插值并生成 absolute State、anchor-relative Action；
- 应用外部 normalization stats、mask 和 TCP128 packing；
- 输出稳定、与物理 backend 解耦的单样本 ABI。

本仓库不负责：

- 把 LIBERO、Hy-Embodied、UMI 等 raw 数据转换成 canonical 数据；
- 生成或训练时统计 normalization stats、field mapping、field/element/pixel mask；
- `DistributedSampler`、PyTorch `DataLoader`、collate、prefetch 或 pinned memory；
- `_extract_wam_batch()`、GPU transfer 或 dtype/device 策略；
- Tokenizer、VAE、camera/token attention、flow target/noise、loss 或模型 forward；
- 将 `[N,K,128]` reshape 成某个特定模型的私有 ABI。

## 5. Public API

本仓库只暴露一个 public Dataset：`NGADCanonicalDataset`。推荐通过 YAML 入口构造：

```python
from ngad_canonical_dataloader import build_dataset_from_yaml

dataset = build_dataset_from_yaml("configs/canonical.yaml")
sample = dataset[0]
```

底层 backend class、插值函数和 window helper 是 Loading 实现细节，不构成模型侧 ABI。
