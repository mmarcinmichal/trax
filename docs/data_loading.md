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

## Preprocessing (Serial pipelines)

- Use `trax.data.Serial` with stream inputs from `trax.data.TFDS`.
- Build train/eval pipelines with `trax.data.make_inputs`.
- BERT helpers are exposed via `trax.data.BertNextSentencePredictionInputs` and
  `trax.data.CreateBertInputs` in module pipelines.

Example (gin):

```gin
make_inputs.train_stream = [
  @training/data.TFDS(),
  @data.NextSentencePrediction(),
  @data.SentencePieceTokenize(),
  @data.MLM(),
  @data.FilterEmptyExamples(),
  @data.AppendValue(),
  @data.TruncateToLength(),
  @data.PadToLength(),
  @data.AddLossWeights(),
  @data.Shuffle(),
  @data.Batch(),
]

make_inputs.eval_stream = [
  @validation/data.TFDS(),
  @data.NextSentencePrediction(),
  @data.SentencePieceTokenize(),
  @data.MLM(),
  @data.FilterEmptyExamples(),
  @data.AppendValue(),
  @data.TruncateToLength(),
  @data.PadToLength(),
  @data.AddLossWeights(),
  @validation/data.Batch(),
]

data.TFDS.dataset_name = 'c4/en:2.3.0'
data.TFDS.keys = ('text',)
training/data.TFDS.train = True
validation/data.TFDS.train = False

train.inputs = @trax.data.make_inputs
```

## Migration mapping

- `tf_dataset_streams` pipelines -> `make_inputs` + `Serial` steps
- `preprocess_fn`/`bare_preprocess_fn` -> `Serial` steps like `Tokenize`, `MLM`, `FilterByLength`
