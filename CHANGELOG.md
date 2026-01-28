# Changelog

## Unreleased
- Removed `trax.data.batcher`; use `trax.data.make_inputs` for data pipelines and YAML configs.
- `trainer.train` now defaults to `trax.data.make_inputs`; pass a configured make_inputs partial or Inputs.
