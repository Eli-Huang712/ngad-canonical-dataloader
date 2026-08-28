# ngad-canonical-dataloader

从 `NGADv1pp` 的 `eli/ngad-canonical-tcp@34eef9b` 机械式抽取的独立 loading package。

当前版本保留原实现中的：

- canonical Lance + JPEG 与 LeRobot v3 Parquet + MP4 读取；
- LIBERO、Hy-Embodied、UMI 数据适配器；
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
from ngad_canonical_dataloader import NGADCanonicalDataset

dataset = NGADCanonicalDataset(...)
sample = dataset[0]
```

本次抽取只建立独立修改基线，不声称已经形成最终解耦 ABI。后续重写应从
`NGADCanonicalDataset.__init__()` 和 `NGADCanonicalDataset.__getitem__()` 开始。

## 来源映射

| 独立仓库 | NGADv1pp 来源 |
|---|---|
| `ngad_canonical_dataloader/datasets/canonical.py` | `ngad/datasets/canonical.py` |
| `ngad_canonical_dataloader/datasets/libero*.py` | `ngad/datasets/libero*.py` |
| `ngad_canonical_dataloader/datasets/hy_embodied*.py` | `ngad/datasets/hy_embodied*.py` |
| `ngad_canonical_dataloader/datasets/umi*.py` | `ngad/datasets/umi*.py` |
| `ngad_canonical_dataloader/rotation.py` | `ngad/utils/rotation.py` |
| `ngad_canonical_dataloader/tcp.py` | `ngad/utils/tcp.py` |
| `ngad_canonical_dataloader/windows.py` | data-only functions from `ngad/utils/wam.py` |
| `ngad_canonical_dataloader/memory.py` | `ngad/utils/wam_memory.py` |

