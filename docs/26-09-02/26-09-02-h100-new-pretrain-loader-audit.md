# H100 新增 PRETRAIN_DATA Loader 检查

## 目的

检查 H100 `PRETRAIN_DATA` 新增数据集，增加明确的 flat LeRobot v3 root 拓扑，并用当前
DataLoader 读取真实 UMI/Hy batch。实现不引入递归搜索或旧配置 fallback。

## 版本与位置

- 本地分支：`main`
- 基线 commit：`d51377ac450b0066bc0c8eb13939bcfae47275ff`
- 本轮 commits：
  - `86380de refactor: 删除 DataLoader YAML schema version`
  - `120097f feat: 支持 flat LeRobot v3 数据根`
  - video-only 无效字段读取修改待本轮最终提交
- H100 Loader 测试树：
  `/gpfs/jiuquyun/projects/jhhuang/ngad-canonical-dataloader-schema-removal-test`
- H100 数据根：`/gpfs/jiuquyun/datasets/PRETRAIN_DATA`
- 临时测试配置：
  `/gpfs/jiuquyun/projects/jhhuang/ngad-canonical-dataloader-runs/schema-removal-260831/umi-flat-video-only.yaml`

## 只读数据结论

- 原 Hy 数据仍为 19 张 `table_NNN`，并有 19 份配置；没有新增 table。
- 新增数据集为 `UMI-Collectsite-KS3-canonical-v3`：89,169 episodes、20,170,029
  frames、30 Hz。
- UMI 使用 flat LeRobot v3 拓扑：根目录直接包含 `meta/`、`data/`、`videos/`、
  `tactile/` 和 `image_pixel_mask_umi.npz`，没有 `table_NNN`。
- 六路相机均声明为 H.264 `[256,256,3]`；Episode video range、Parquet shard metadata
  与当前 Parquet/H.264 backend 所需字段一致。
- `meta/modality_mask.json` 只有 `dataset`、`field_mask`、`image_pixel_mask`，没有严格
  sidecar 所需的 `field_mapping` 和 `element_mask`。
- modality mask 将 `observation.state` 标记为 false；虽然物理 Parquet 中有 `[20]`
  state，当前数据合同不允许把它静默当作有效 canonical State。

## 实际测试结果

使用无 `schema_version` 的 video-only YAML 调用 `build_dataset_from_yaml()`，初始化在 table
发现阶段按设计失败：

```text
ValueError: .../UMI-Collectsite-KS3-canonical-v3 is neither a table_NNN root nor a
dataset root with direct table_NNN children.
```

独立执行严格 sidecar 校验，得到第二个明确失败：

```text
ValueError: modality_mask.json must contain exactly
['dataset', 'element_mask', 'field_mapping', 'field_mask', 'image_pixel_mask'].
```

因此当前测试没有进入 Parquet row read 或 H.264 decode，不能报告真实 sample loading 已通过。

## 最终实现

- 保留 dataset root 和 single `table_NNN` root，并新增 flat LeRobot v3 root：path 自身必须
  同时存在 `meta/info.json`、`data/` 和 `videos/`。
- Flat payload 和直接 `table_NNN` 子目录不能同时存在；出现时明确报错，不做自动选择。
- Video-only 不再要求 State 有效，也不会从 row backend 读取 State 或 tactile；canonical
  mode 仍严格要求有效 State，并从 State 重建 Action。
- UMI 严格 sidecar 与 YAML 暂存于 jhhuang 测试目录；没有修改只读的 UMI 数据发布根。

## 测试结果

- H100 package tests：flat-root commit 后 `22 passed in 6.65s`；video-only 字段读取修改后
  `22 passed in 1.86s`。
- UMI 真实 DataLoader smoke：rc=0。
  - Dataset：6,730,098 windows、89,169 episodes、1 physical table；
  - `video=[1,81,6,3,256,256]`；六路 anchor camera mask 全为 true；
  - `image_pixel_mask=[1,81,6,256,256]`；
  - batch 不含 State/Action，`sample_mode=video_only`；
  - prompt 为“关闭现代开放式格柜”。
- UMI 日志：
  `/gpfs/jiuquyun/projects/jhhuang/ngad-canonical-dataloader-runs/umi-flat-260902/real-dataloader-smoke.log`，
  rc 文件同目录 `real-dataloader-smoke.rc`。
- Hy 19 份发布 YAML 已删除 `schema_version`，新解析器验证 19/19 通过、schema line 数为 0。
- Hy `table_000` 无版本 YAML 真实 smoke：rc=0，输出仍为
  `video=[1,81,6,3,256,256]`，anchor camera mask `[1,0,1,0,1,0]`。
- Hy 配置修改前备份：
  `/gpfs/jiuquyun/projects/jhhuang/ngad-canonical-dataloader-runs/schema-removal-260902/hy-configs-before.tar.gz`，
  SHA-256 为 `5e50f8068fd801870de6dfdef1f26aaff8dd2b01c515dd1d81933f4ccf4996fd`。

## 当前限制

- UMI 数据根对 `ai-users` 只有读权限，因此正式 sidecar/YAML 尚未发布到数据根；当前文件
  位于 `/gpfs/jiuquyun/projects/jhhuang/ngad-canonical-dataloader-runs/umi-flat-260902/`。
- UMI `observation.state` 在发布 modality mask 中为 false，本轮仅验证 video-only；不能据此
  声明 canonical State/Action training 已可用。
