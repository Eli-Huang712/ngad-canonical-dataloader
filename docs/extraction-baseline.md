# Extraction baseline

- Source repository: `NGADv1pp`
- Source branch: `eli/ngad-canonical-tcp`
- Source commit: `34eef9bb8c54ffb4b1bcbb46f232471519c5b54e`
- Extraction date: 2026-08-28
- Scope: Dataset loading/sample construction, relative action, normalization, TCP128 and source adapters
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
