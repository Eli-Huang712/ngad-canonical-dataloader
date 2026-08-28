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
`type` 选择 LIBERO、Hy-Embodied 或 UMI 专用实现。

## Canonical 输入字段

- 六路 `observation.images.cam_*` RGB video，单帧 `[256,256,3]`；
- `observation.state`: `float32[20]` absolute dual-arm TCP；
- `action`: `float32[20]`，当前仅验证存在，训练 action 仍从 state 重建；
- `observation.tactile.values`: `float32[4,3,25,6]`；
- `observation.tactile.dt`: `float32[4,3]`；
- `timestamp`、`frame_index`、`episode_index`、`index`、`task_index`。

Dataset sample 保留原模型侧字段，同时新增 anchor 时刻的两个 canonical tactile key。

本次抽取只建立独立修改基线，不声称已经形成最终解耦 ABI。后续重写应从
`NGADCanonicalDataset.__init__()` 和 `NGADCanonicalDataset.__getitem__()` 开始。

## 来源映射

| 独立仓库 | NGADv1pp 来源 |
|---|---|
| `ngad_canonical_dataloader/datasets/canonical.py` | `ngad/datasets/canonical.py` |
| `ngad_canonical_dataloader/rotation.py` | `ngad/utils/rotation.py` |
| `ngad_canonical_dataloader/tcp.py` | `ngad/utils/tcp.py` |
| `ngad_canonical_dataloader/windows.py` | data-only functions from `ngad/utils/wam.py` |
| `ngad_canonical_dataloader/memory.py` | `ngad/utils/wam_memory.py` |

后续修改已删除 source-specific Dataset，并增加正式 canonical schema 与 YAML 配置入口。
