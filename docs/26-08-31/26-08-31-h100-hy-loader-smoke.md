# H100 Hy-Embodied Canonical Loader 检查

## 目的

以 H100 已转换数据的实际发布拓扑为准，修正 canonical table 发现、物理 backend 选择和
table-local row offset 读取，并完成真实数据 DataLoader smoke test。

## 版本与位置

- 分支：`eli/single-dataset-topology`
- 修改前 commit：`01f8b14`
- H100 数据：`/gpfs/jiuquyun/datasets/PRETRAIN_DATA/Hy-Embodied-0.5-VLA-Data`
- H100 代码：`/gpfs/jiuquyun/projects/jhhuang/ngad-canonical-dataloader`
- 状态：完成（single-table、全局 normalization 与 video-only 真实 loading 均通过）

## 只读检查结论

- 根目录没有 `tables.parquet`，直接包含 19 个 `table_NNN/` 和 `reports/`。
- 每张 table 均为 `meta/ + reports/ + table_NNN.lance/`；`info.json` 不含
  `storage_backend`。
- 19 张 table 的 `fps` 均为 30；`info.total_episodes`、`info.total_frames` 与 Episode
  metadata 相符。
- 每张 table 的 `dataset_from_index` 都从 0 开始，因此物理 Episode/row offset 是
  table-local；Dataset 对外仍建立统一的全局 window index。
- Lance 相机在 `info.features` 中声明为 `dtype=image`，不是 `video`。
- `meta/stats.json` 中两臂 gripper 维范围均为 `0–90`，与 Loader 约定的 `[0,1]`
  absolute openness 不一致；转换报告未进行数值语义 quality check。
- Lance payload 文件原权限包含 `0600`，普通 `jhhuang` 账号可读 metadata，但无法读取物理行。

## 当前改动

- 删除根级 `tables.parquet` 依赖，按 `table_NNN` 直接子目录发现并排序 table。
- Episode/row offset 按 table-local 语义校验和读取。
- 根据唯一物理 payload 选择 Lance/JPEG 或 Parquet/H.264 backend。
- Lance/JPEG 相机按 `image[256,256,3]` 校验，Parquet/H.264 按
  `video[256,256,3]` 校验。

## 验证结果

- 本地 `compileall`：通过；本机系统 Python 未安装 pytest，因此 pytest 在 H100 执行。
- H100 package tests：Slurm `h100` partition、1 GPU，`14 passed in 7.38s`。
- 普通用户 Dataset 初始化：19 physical tables、215,577 episodes，0.70 秒。
- 普通用户首个 sample：在 `Lance.take()` 处因 payload `0600` 返回 permission denied；这是
  文件权限，不是 Loader 逻辑错误。
- 经明确授权后，使用 `h100-admin + sudo` 在 Slurm 中只读执行真实 DataLoader：
  `batch_size=2`、`num_workers=2` 成功。
- 输出：`video=[2,1,6,3,256,256]`、`state=[2,1,2,128]`、
  `action=[2,1,2,128]`；camera mask 为 `[1,0,1,0,1,0]`。
- 运行目录：
  `/gpfs/jiuquyun/projects/jhhuang/ngad-canonical-dataloader-runs/hy-smoke-260831/`。

## Single-table、全局 normalization 与 video-only 验证

### 改动与版本

- `b40eb27 feat: 兼容 single-table canonical 拓扑`
- `53261de refactor: 统一全局 normalization 配置`
- `feat: 支持无统计量的视频读取模式`（本次提交）。
- H100 独立测试 worktree：
  `/gpfs/jiuquyun/projects/jhhuang/ngad-canonical-dataloader-video-only-53261de`。
- H100 运行与审计目录：
  `/gpfs/jiuquyun/projects/jhhuang/ngad-canonical-dataloader-runs/video-only-260831/`。

### 19 份发布配置

- 更新目录：
  `/gpfs/jiuquyun/datasets/PRETRAIN_DATA/Hy-Embodied-0.5-VLA-Data/dataset_configs/configs/`。
- 19 份 YAML 与实际发布的 19 张 table 一一对应，每份 `dataset_dirs` 只指向一个
  `table_NNN` 直接根目录。
- `dataset.normalization_stats_path: null`，因此使用 `video_only` sample mode；未保留
  per-root normalization、`table_name` 或旧配置 fallback。
- 更新前配置备份：
  `/gpfs/jiuquyun/projects/jhhuang/ngad-canonical-dataloader-runs/video-only-260831/configs-before.tar.gz`，
  SHA-256 为 `fbc9411f744854c1198dfb466cd2986b02baeb80ef8b3bf203af7dddc4ffd2b4`。
- 配置校验：19 个配置、19 个唯一实际 table、全部 `normalization_stats_path=null`、
  全部解析为 `sample_mode=video_only`；日志 `config-update.log`，rc=0。

### ai-users 发布权限

- 修改范围严格限制为数据发布根目录，未跟随符号链接，未修改 youzi 原始配置。
- 修改前审计：12,248 个普通文件、190 个目录、64 个符号链接；所有实体文件/目录的
  group 均为 `ai-users`，目录均可组遍历，7,967 个文件缺少组读权限。
- sudo 操作只对缺少组读权限的普通文件增加 `g+r`；未执行 `chgrp`，未增加 world 权限。
- 修改后审计：`missing_group_read=0`、`missing_group_traverse=0`、`wrong_gid=0`。
- 修改前后精确权限清单分别为 `permissions-before.jsonl` 与
  `permissions-after.jsonl`，可用于逐文件回退。

### 测试结果

- H100 package tests：`19 passed in 2.79s`，日志 `package-pytest.log`，rc=0。
- 普通 `h100` 账号经 Slurm 最小 1-GPU allocation 读取
  `dataset_configs/configs/hy_table_000.yaml` 成功；GPU 只用于满足队列 GRES 约束，
  smoke 本身不调用模型。
- 真实 Dataset：3,599,931 个窗口、11,604 个 episode、1 张 physical table。
- batch keys 仅包含 `video/frame_offsets/source_frame_indices/frame_timestamps/frame_valid/`
  `camera_mask/image_pixel_mask/prompt/data_info`；没有 `state`、`action` 或 TCP 字段。
- 输出形状：`video=[1,81,6,3,256,256]`、`camera_mask=[1,81,6]`、
  `image_pixel_mask=[1,81,6,256,256]`、`frame_valid=[1,81]`。
- anchor camera mask 为 `[1,0,1,0,1,0]`，`sample_mode=["video_only"]`。
- 结果日志 `video-only-ai-users.log`，rc 文件 `video-only-ai-users.rc` 为 0。

## 剩余数据合同问题

- 有效 action/state gripper 在 Loader 输出中变为 1；落盘原值范围为 `0–90`，而当前
  canonical ABI 要求 `[0,1]`。本次没有在 Loader 中静默加入 `/90` 转换。
- 首个 task prompt 为 `__UNKNOWN__`，第二个样本能正常读到中文任务文本。这属于落盘
  task metadata 内容，需要数据侧判断 `__UNKNOWN__` 是否允许。
- 数据 root 没有生产用 `mask_and_mapping`、pixel mask 和兼容当前 TCP transform 的 mixed
  normalization stats。本次 smoke 使用 jhhuang 运行目录中的 diagnostic-only sidecar；
  它们不能用于训练。
