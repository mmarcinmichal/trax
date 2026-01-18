# Dataset Loading vs. Preprocessing

This repo now separates dataset *loading* from *preprocessing* to keep the
interface stable and make it easier to migrate to Hydra composition.

## Loader (raw TFDS streams)

- Use `trax.data.data_streams` or `trax.data.loader.interface.DatasetLoader`.
- Loader returns raw `tf.data.Dataset` objects only (no transforms).

Example (gin):

```gin
data_streams.dataset_name = 'c4/en:2.3.0'
data_streams.data_dir = None
data_streams.eval_holdout_size = 0.1
data_streams.download = True
```

Example (Hydra):

```yaml
dataset_loader:
  _target_: "trax.data.loader.interface.DatasetLoader"
  dataset_name: "c4/en:2.3.0"
  data_dir: null
  eval_holdout_size: 0.1
  download: true
  train_split: null
  eval_split: null
```

## Preprocessing (explicit pipeline layer)

- Use `trax.data.tf_dataset_streams` to apply TF preprocessing and convert to
  NumPy streams.
- Place all `preprocess_fn` and `bare_preprocess_fn` here.

Example (gin):

```gin
batcher.data_streams = @data.tf_dataset_streams
tf_dataset_streams.datasets = @data.data_streams
tf_dataset_streams.input_name = 'inputs'
tf_dataset_streams.target_name = 'targets'
tf_dataset_streams.bare_preprocess_fn = @data.c4_bare_preprocess_fn
tf_dataset_streams.preprocess_fn = @data.c4_preprocess
```

## Migration mapping

- `data_streams.preprocess_fn` -> `tf_dataset_streams.preprocess_fn`
- `data_streams.bare_preprocess_fn` -> `tf_dataset_streams.bare_preprocess_fn`
- `data_streams.input_name` / `target_name` -> `tf_dataset_streams.input_name` / `target_name`
- `data_streams.shuffle_buffer_size` -> `tf_dataset_streams.shuffle_buffer_size`
- `data_streams.download` -> `DatasetLoader.download` (or `data_streams.download` in gin)
