# coding=utf-8
# Copyright 2022 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TensorFlow data sources (loading only) and stream conversion utilities."""

import gin
import numpy as np

from trax import fastmath
from trax.data.loader.tf import interface


def dataset_to_stream(dataset, input_name):
    """Takes a tf.Dataset and creates a numpy stream of ready batches."""
    for example in fastmath.dataset_as_numpy(dataset):
        features = example[0]

        chosen_input = input_name if input_name in features else "inputs"
        if chosen_input not in features:
            raise KeyError(
                f"Expected input feature '{input_name}' (or 'inputs') not found in features: {list(features.keys())}"
            )

        if not isinstance(features[chosen_input], np.ndarray):
            input = np.array(features[chosen_input]).reshape(1, -1)
        else:
            input = features[chosen_input]

        if not isinstance(example[1], np.ndarray):
            output = np.array(example[1]).reshape(1, -1)
        else:
            output = example[1]

        inp, out = input, output
        mask = features["mask"] if "mask" in features else None
        if isinstance(inp, np.uint8):
            inp = inp.astype(np.int32)
        if isinstance(out, np.uint8):
            out = out.astype(np.int32)
        yield (inp, out) if mask is None else (inp, out, mask)


@gin.configurable(module="trax.data")
def data_streams(
    dataset_name,
    data_dir=None,
    eval_holdout_size=0,
    use_alt_eval=False,
    train_split=None,
    eval_split=None,
    download=True,
    shuffle_train_files=True,
    shuffle_eval_files=False,
    host_id=None,
    n_hosts=None,
    require_train_split=True,
):
    """Loads raw train/eval tf.data.Datasets (no preprocessing)."""
    loader = interface.DatasetLoader(
        dataset_name=dataset_name,
        data_dir=data_dir,
        eval_holdout_size=eval_holdout_size,
        use_alt_eval=use_alt_eval,
        download=download,
        train_split=train_split,
        eval_split=eval_split,
        shuffle_train_files=shuffle_train_files,
        shuffle_eval_files=shuffle_eval_files,
        host_id=host_id,
        n_hosts=n_hosts,
        require_train_split=require_train_split,
    )
    return loader.datasets()


@gin.configurable(module="trax.data")
def TFDS(  # pylint: disable=invalid-name
    dataset_name,
    data_dir=None,
    keys=None,
    train=True,
    use_alt_eval=False,
    shuffle_train=True,
    host_id=None,
    n_hosts=None,
    eval_holdout_size=0,
):
    """Creates a data source from a TFDS dataset (raw only)."""
    loader = interface.DatasetLoader(
        dataset_name=dataset_name,
        data_dir=data_dir,
        eval_holdout_size=eval_holdout_size,
        use_alt_eval=use_alt_eval,
        shuffle_train_files=shuffle_train,
        shuffle_eval_files=False,
        host_id=host_id,
        n_hosts=n_hosts,
        require_train_split=train,
    )
    streams = loader.datasets()
    dataset = streams.train if train else streams.eval
    if train and dataset is None:
        raise ValueError(
            f"Dataset {dataset_name} does not provide a train split for training."
        )
    if keys:
        def select_from(example):
            return tuple(example[k] for k in keys)

        dataset = dataset.map(select_from)
    dataset = dataset.repeat()

    def gen(generator=None):
        del generator
        for example in fastmath.dataset_as_numpy(dataset):
            yield example

    return gen
