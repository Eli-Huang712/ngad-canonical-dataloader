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
- Real Lance/LeRobot root tests remain intentionally deferred; local syntax compilation is the only
  validation performed for this follow-up because the system Python has no `torch` installation.
