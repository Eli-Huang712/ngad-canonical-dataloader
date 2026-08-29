# Extraction baseline

- Source repository: `NGADv1pp`
- Source branch: `eli/ngad-canonical-tcp`
- Source commit: `34eef9bb8c54ffb4b1bcbb46f232471519c5b54e`
- Extraction date: 2026-08-28
- Scope: canonical Dataset loading/sample construction, relative action, normalization and TCP128
- Excluded: training entry, transfer, tokenizer, VAE, flow construction and model

The initial extraction intentionally preserves the source behavior. It is a baseline for the
next decoupling pass, not an assertion that the current sample ABI is final or correct.

## Local validation

- `python3 -m compileall -q ngad_canonical_dataloader tests`: passed.
- Runtime import and pytest were not executed locally because the system Python does not provide
  `torch` or `pytest`. Real backend/data correctness is intentionally outside this extraction step.

## Canonical-only follow-up

- Removed the LIBERO, Hy-Embodied and UMI-specific Dataset implementations.
- Kept one `NGADCanonicalDataset` public reader.
- Aligned six camera names and tactile fields with the published canonical document.
- Added a strict versioned YAML configuration and `build_dataset_from_yaml()`.
- Added a required per-root `mask_path` sidecar. Camera/tactile field validity and state/action
  element validity now come from canonical masks rather than physical-backend inference.
- Replaced `arm_mask[2]` with state/action element masks `[20]` and their direct TCP128 tensor masks.
- Merged the extracted rotation and TCP helpers into one `action.py` module. It owns Rot6D
  interpolation, anchor-relative pose construction, normalization, TCP128 packing and masks; the
  previous `rotation.py` and `tcp.py` import paths no longer exist.
- Fields declared unavailable may be physically absent; the Dataset emits deterministic black/zero
  tensors and explicit tensor masks without a backend fallback.
- Moved physical storage access into internal table/image backends. Lance and Parquet readers now
  normalize rows to canonical keys, JPEG and H.264 readers return uint8 RGB tensors, and
  `NGADCanonicalDataset.__getitem__()` no longer branches on a physical backend.
- Removed source-adapter-only axis-angle, single-arm packing, chunk-wrapper and single-rate window
  helpers. Deployment inverse transforms and Dataset normalization export methods remain intact.
- Removed unused root/episode metadata. Camera order and 256x256 resolution are fixed ABI constants,
  and decoded images are validated rather than silently resized.

## Unified timeline follow-up (2026-08-29)

- Replaced the separate current/future, recent, long and action-history index paths with one list of
  anchor-relative inclusive `frame_ranges`.
- Removed independent `target_action_fps` and action horizons. The action grid is now derived from
  `rgb_rate_hz * action_steps_per_rgb_frame`.
- Added a fixed `TimelineLayout` containing semantic frame offsets, per-frame action-step offsets and
  the lightweight `offset_to_position` lookup used by downstream callers.
- Changed the sample ABI to time-major `video[N,6,3,256,256]` and frame-aligned
  `action[N,K,128]`, with explicit offset, timestamp, source-index and validity tensors.
- Deleted the old `memory.py` production path and all legacy output fields; there is no old-ABI
  compatibility or fallback.
- Local `compileall`, config/timeline assertions and a synthetic backend `Dataset.__getitem__()` ABI
  test passed with the local conda Python.
- H200-1 CPU test: branch/worktree state was transferred without GPU use to
  `/data/home/jhhuang/projects/ngad-canonical-dataloader-test-260829-1443`; archive SHA256 was
  `fa2505f571fc499b90e977479ef6b596d7717ed93d7fd171dec350134a5e8cf2`. Command:
  `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /data/cache/conda/envs/maxliu/sana/bin/python -m pytest -q`.
  Result after the final unified `action[N,K,128]` rename: `11 passed in 1.49s`.
- Real Lance/LeRobot root tests remain a separate runtime gate; the completed H200 run used the
  synthetic backend test and did not read production datasets.
