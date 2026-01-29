# Changelog

## Unreleased
- Removed `trax.data.batcher`; use `trax.data.make_streams` for data pipelines and YAML configs.
- `trainer.train` now defaults to `trax.data.make_streams`; pass a configured make_streams partial or StreamBundle.
- Removed `trax.data.preprocessing.inputs.Inputs`; configure pipelines via `trax.data.make_streams`.
