# coding=utf-8
# Copyright 2024 The Trax Authors.
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

"""Tests for Hydra data stream configs."""

import sys

from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import tensorflow as tf

from absl import flags
from absl.testing import absltest
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from trax import data as trax_data
from trax import fastmath
from trax.data.loader.tf.interface import DatasetStreams
from trax.utils.learning.supervised.trainer import hydra as hydra_utils

FLAGS = flags.FLAGS


def _ensure_flags_parsed():
    try:
        flags.FLAGS(sys.argv)
    except flags.Error:
        flags.FLAGS([sys.argv[0]], known_only=True)


def _fake_tfds_stream(texts, tuple_len=1):
    def _tfds(*args, **kwargs):
        del args, kwargs

        def _gen(_=None):
            for text in texts:
                if tuple_len == 1:
                    yield (text,)
                else:
                    yield tuple(text for _ in range(tuple_len))

        return _gen

    return _tfds


def _repo_config_dir():
    repo_root = Path(__file__).resolve().parents[5]
    return str(
        repo_root / "resources" / "learning" / "supervised" / "configs" / "yaml"
    )


def _apply_cfg_updates(cfg, updates, prefix=""):
    with open_dict(cfg):
        for key, value in updates.items():
            full_key = f"{prefix}{key}"
            if "." in key:
                parent_key, leaf = full_key.rsplit(".", 1)
                parent = OmegaConf.select(cfg, parent_key)
                if isinstance(parent, DictConfig):
                    parent[leaf] = value
                    continue
            OmegaConf.update(cfg, full_key, value, merge=not isinstance(value, dict))


def _apply_cfg_updates_if_present(cfg, updates, prefix=""):
    with open_dict(cfg):
        for key, value in updates.items():
            full_key = f"{prefix}{key}"
            if OmegaConf.select(cfg, full_key) is None:
                continue
            if "." in key:
                parent_key, leaf = full_key.rsplit(".", 1)
                parent = OmegaConf.select(cfg, parent_key)
                if isinstance(parent, DictConfig):
                    parent[leaf] = value
                    continue
            OmegaConf.update(cfg, full_key, value, merge=not isinstance(value, dict))


def _normalize_data_cfg(cfg):
    data_node = OmegaConf.select(cfg, "data.bert.data") or OmegaConf.select(cfg, "data")
    if data_node is None:
        return cfg
    # Flatten nested base config (data.data.*) into data.* for interpolation.
    def _flatten_data(container):
        while isinstance(container, dict) and "data" in container:
            base = container.get("data") or {}
            overlay = {k: v for k, v in container.items() if k != "data"}
            container = {**base, **overlay}
        return container

    container = OmegaConf.to_container(data_node, resolve=False)
    while isinstance(container, dict) and len(container) == 1:
        only_value = next(iter(container.values()))
        if isinstance(only_value, dict):
            container = only_value
            continue
        break
    if isinstance(container, dict) and "data" in container:
        container = _flatten_data(container)
    elif isinstance(container, dict):
        for key, value in container.items():
            if isinstance(value, dict) and "data" in value:
                base = _flatten_data(value)
                rest = {k: v for k, v in container.items() if k != key}
                container = {**base, **rest}
                break
    return OmegaConf.create({"data": container})


def _instantiate_stream(cfg, stream):
    if isinstance(stream, ListConfig):
        stream = list(stream)
    resolved = []
    for item in stream:
        if isinstance(item, str) and item.startswith("${") and item.endswith("}"):
            target = OmegaConf.select(cfg, item[2:-1])
            resolved.append(_instantiate_with_len_map_fix(target))
        else:
            resolved.append(_instantiate_with_len_map_fix(item))
    return resolved


def _instantiate_with_len_map_fix(item):
    if isinstance(item, DictConfig):
        target = item.get("_target_")
    elif isinstance(item, dict):
        target = item.get("_target_")
    else:
        target = None

    if target == "trax.data.TruncateToLength":
        container = OmegaConf.to_container(item, resolve=False)
        len_map = container.get("len_map", {})
        len_map = {
            key: tuple(value) for key, value in len_map.items()
        }
        kwargs = {key: value for key, value in container.items() if key != "_target_"}
        kwargs["len_map"] = len_map
        return trax_data.TruncateToLength(**kwargs)
    if target in ("trax.data.TFDS", "trax.data.loader.tf.base.TFDS"):
        container = OmegaConf.to_container(item, resolve=True)
        dataset_name = container.get("dataset_name")
        if dataset_name is None or isinstance(trax_data.loader.tf.base.TFDS, mock.Mock):
            return instantiate(item)
        train = container.get("train", True)
        keys = container.get("keys")

        def _stream(_=None):
            streams = trax_data.loader.tf.base.data_streams(
                dataset_name=dataset_name,
                data_dir=container.get("data_dir"),
                eval_holdout_size=container.get("eval_holdout_size", 0),
                use_alt_eval=container.get("use_alt_eval", False),
                shuffle_train_files=container.get("shuffle_train", True),
                shuffle_eval_files=container.get("shuffle_eval", False),
                host_id=container.get("host_id"),
                n_hosts=container.get("n_hosts"),
                require_train_split=train,
            )
            dataset = streams.train if train else streams.eval
            if keys:
                def select_from(example):
                    return tuple(example[k] for k in keys)

                dataset = dataset.map(select_from)
            for example in fastmath.dataset_as_numpy(dataset.repeat()):
                yield example

        return _stream
    return instantiate(item)


def _dict_to_tuple(input_key, target_key):
    def _map(stream):
        for example in stream:
            if isinstance(example, dict):
                yield (example[input_key], example[target_key])
            else:
                yield example

    return _map


def _dict_to_tuple_cfg(input_key, target_key):
    return {
        "_target_": "trax.data.DictToTuple",
        "keys": [input_key, target_key],
    }


def _dataset_streams_from_dicts(dataset_name, train_dict, eval_dict, supervised_keys):
    train_ds = tf.data.Dataset.from_tensor_slices(train_dict)
    eval_ds = tf.data.Dataset.from_tensor_slices(eval_dict)
    return DatasetStreams(
        train=train_ds,
        eval=eval_ds,
        supervised_keys=supervised_keys,
        dataset_name=dataset_name,
        data_dir=None,
    )


def _text_preprocess_overrides(tokenize_keys, tuple_keys=("inputs", "targets")):
    return {
        "GenericTextPreprocess": {
            "_target_": "trax.data.Serial",
            "_args_": [
                "${data.Rekey}",
                "${data.TestConvertToUnicode}",
                "${data.TestTokenize}",
                "${data.DictToTuple}",
            ],
        },
        "TestConvertToUnicode._target_": "trax.data.ConvertToUnicode",
        "TestConvertToUnicode.keys": list(tokenize_keys),
        "TestTokenize._target_": "trax.data.encoder.encoder.Tokenize",
        "TestTokenize.vocab_type": "char",
        "TestTokenize.keys": list(tokenize_keys),
        "DictToTuple._target_": "trax.data.DictToTuple",
        "DictToTuple.keys": list(tuple_keys),
    }


def _tuple_text_stream_overrides(tokenize_indices=(0, 1)):
    return {
        "make_streams.train_stream": [
            "${data.train.TFDS}",
            "${data.TestConvertToUnicode}",
            "${data.TestTokenize}",
            "${data.train.BucketByLength}",
            "${data.AddLossWeights}",
        ],
        "make_streams.eval_stream": [
            "${data.eval.TFDS}",
            "${data.TestConvertToUnicode}",
            "${data.TestTokenize}",
            "${data.eval.BucketByLength}",
            "${data.AddLossWeights}",
        ],
        "TestConvertToUnicode._target_": "trax.data.ConvertToUnicode",
        "TestConvertToUnicode.keys": list(tokenize_indices),
        "TestTokenize._target_": "trax.data.encoder.encoder.Tokenize",
        "TestTokenize.vocab_type": "char",
        "TestTokenize.keys": list(tokenize_indices),
    }


@contextmanager
def _data_cfg(config_name, updates):
    config_name_flag = FLAGS["hydra_config_name"]
    overrides_flag = FLAGS["hydra_overrides"]
    config_dir_flag = FLAGS["hydra_config_dir"]
    original_name = config_name_flag.value
    original_overrides = overrides_flag.value
    original_dir = config_dir_flag.value
    try:
        config_name_flag.value = config_name
        overrides_flag.value = []
        config_dir_flag.value = _repo_config_dir()
        cfg = _normalize_data_cfg(hydra_utils.compose_config())
        _apply_cfg_updates(cfg, updates, prefix="data.")
        yield cfg
    finally:
        config_name_flag.value = original_name
        overrides_flag.value = original_overrides
        config_dir_flag.value = original_dir


@contextmanager
def _compose_data_cfg(config_name):
    config_name_flag = FLAGS["hydra_config_name"]
    overrides_flag = FLAGS["hydra_overrides"]
    config_dir_flag = FLAGS["hydra_config_dir"]
    original_name = config_name_flag.value
    original_overrides = overrides_flag.value
    original_dir = config_dir_flag.value
    try:
        config_name_flag.value = config_name
        overrides_flag.value = []
        config_dir_flag.value = _repo_config_dir()
        cfg = _normalize_data_cfg(hydra_utils.compose_config())
        yield cfg
    finally:
        config_name_flag.value = original_name
        overrides_flag.value = original_overrides
        config_dir_flag.value = original_dir


def _build_inputs_from_cfg(cfg, tfds_patch, texts, tuple_len=1):
    with mock.patch(
        tfds_patch,
        side_effect=_fake_tfds_stream(texts, tuple_len=tuple_len),
    ) as tfds_mock:
        make_inputs_cfg = OmegaConf.select(cfg, "data.make_streams")
        train_stream = _instantiate_stream(cfg, make_inputs_cfg.train_stream)
        eval_stream = _instantiate_stream(cfg, make_inputs_cfg.eval_stream)
        inputs = trax_data.make_streams(
            train_stream=train_stream, eval_stream=eval_stream
        )
        batch = next(inputs.train_stream)
    return batch, tfds_mock


def _make_inputs_from_cfg(cfg):
    make_inputs_cfg = OmegaConf.select(cfg, "data.make_streams")
    train_stream = _instantiate_stream(cfg, make_inputs_cfg.train_stream)
    eval_stream = _instantiate_stream(cfg, make_inputs_cfg.eval_stream)
    return trax_data.make_streams(train_stream=train_stream, eval_stream=eval_stream)


def _assert_make_inputs_batch(cfg, expected_len=None, expected_batch_size=None):
    inputs = _make_inputs_from_cfg(cfg)
    batch = next(inputs.train_stream)
    if expected_len is not None:
        assert len(batch) == expected_len
    if expected_batch_size is not None:
        assert batch[0].shape[0] == expected_batch_size


class HydraDataTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        _ensure_flags_parsed()

    def _run_make_inputs_case(
        self,
        config_name,
        updates,
        streams,
        expected_len=None,
        expected_batch_size=None,
    ):
        with _data_cfg(config_name, updates) as cfg:
            with mock.patch(
                "trax.data.loader.tf.base.data_streams",
                return_value=streams,
            ):
                _assert_make_inputs_batch(
                    cfg,
                    expected_len=expected_len,
                    expected_batch_size=expected_batch_size,
                )

    def test_bert_builds_streams(self):
        # Explicitly select the data YAML config.
        updates = {
            "Tokenize._target_": "trax.data.encoder.encoder.Tokenize",
            "Tokenize.vocab_type": "char",
            "Tokenize.keys": [0, 1],
            "PadToLength.len_map": {0: 16, 1: 16},
            "PadToLength.pad_value": {0: 0, 1: 0},
            "Batch.batch_size": 2,
            "Shuffle.queue_size": 2,
            "DummyStream._target_": "trax.data.loader.tf.base.TFDS",
            "make_streams.train_stream": [
                "${data.DummyStream}",
                "${data.Tokenize}",
                "${data.PadToLength}",
                "${data.Batch}",
            ],
            "make_streams.eval_stream": [
                "${data.DummyStream}",
                "${data.Tokenize}",
                "${data.PadToLength}",
                "${data.Batch}",
            ],
        }
        texts = [
            "Alpha beta gamma. Delta epsilon zeta.",
            "Eta theta iota. Kappa lambda mu.",
            "Nu xi omicron. Pi rho sigma.",
            "Tau upsilon phi. Chi psi omega.",
        ]
        with _data_cfg("data/bert/bert", updates) as cfg:
            batch, tfds_mock = _build_inputs_from_cfg(
                cfg,
                "trax.data.loader.tf.base.TFDS",
                texts,
                tuple_len=2,
            )

        self.assertTrue(tfds_mock.called)
        self.assertLen(batch, 2)
        self.assertEqual(batch[0].shape[0], 2)

    def test_bert_pretraining_loads_data(self):
        # Explicitly select the data YAML config.
        updates = {
            "train.BertNextSentencePredictionInputs.dataset_name": "dummy",
            "eval.BertNextSentencePredictionInputs.dataset_name": "dummy",
            "Tokenize._target_": "trax.data.encoder.encoder.Tokenize",
            "Tokenize.vocab_type": "char",
            "Tokenize.keys": [0, 1],
            "PadToLength.len_map": {0: 16, 1: 16, 2: 16},
            "PadToLength.pad_value": {0: 0, 1: 0, 2: 0},
            "TruncateToLength.len_map": {0: (16,), 1: (16,), 2: (16,)},
            "Batch.batch_size": 2,
            "Shuffle.queue_size": 2,
            "mask_random_tokens.masking_prob": 1.0,
        }
        texts = [
            "Alpha beta gamma. Delta epsilon zeta.",
            "Eta theta iota. Kappa lambda mu.",
            "Nu xi omicron. Pi rho sigma.",
            "Tau upsilon phi. Chi psi omega.",
        ]
        with _data_cfg("data/bert/bert_pretraining", updates) as cfg:
            batch, tfds_mock = _build_inputs_from_cfg(
                cfg,
                "trax.data.preprocessing.modules.bert.TFDS",
                texts,
            )

        self.assertTrue(tfds_mock.called)
        self.assertLen(batch, 7)
        self.assertEqual(batch[0].shape[0], 2)

    def test_bert_pretraining_onlynsp_loads_data(self):
        updates = {
            "train.BertNextSentencePredictionInputs.dataset_name": "dummy",
            "eval.BertNextSentencePredictionInputs.dataset_name": "dummy",
            "Tokenize._target_": "trax.data.encoder.encoder.Tokenize",
            "Tokenize.vocab_type": "char",
            "Tokenize.keys": [0, 1],
            "PadToLength.len_map": {0: 16, 1: 16, 2: 16},
            "PadToLength.pad_value": {0: 0, 1: 0, 2: 0},
            "TruncateToLength.len_map": {0: (16,), 1: (16,), 2: (16,)},
            "Batch.batch_size": 2,
            "Shuffle.queue_size": 2,
        }
        texts = [
            "Alpha beta gamma. Delta epsilon zeta.",
            "Eta theta iota. Kappa lambda mu.",
            "Nu xi omicron. Pi rho sigma.",
            "Tau upsilon phi. Chi psi omega.",
        ]

        with _data_cfg("data/bert/bert_pretraining_onlynsp", updates) as cfg:
            batch, tfds_mock = _build_inputs_from_cfg(
                cfg,
                "trax.data.preprocessing.modules.bert.TFDS",
                texts,
            )

        self.assertTrue(tfds_mock.called)
        self.assertLen(batch, 5)
        self.assertEqual(batch[0].shape[0], 2)

    def test_bert_pretraining_onlymlm_loads_data(self):
        updates = {
            "train.TFDS._target_": "trax.data.loader.tf.base.TFDS",
            "eval.TFDS._target_": "trax.data.loader.tf.base.TFDS",
            "train.TFDS.dataset_name": "dummy",
            "eval.TFDS.dataset_name": "dummy",
            "Tokenize._target_": "trax.data.encoder.encoder.Tokenize",
            "Tokenize.vocab_type": "char",
            "Tokenize.keys": [0],
            "PadToLength.len_map": {0: 16, 1: 16, 2: 16},
            "PadToLength.pad_value": {0: 0, 1: 0, 2: 0},
            "TruncateToLength.len_map": {0: (16,), 1: (16,), 2: (16,)},
            "Batch.batch_size": 2,
            "Shuffle.queue_size": 2,
            "mask_random_tokens.masking_prob": 1.0,
        }
        texts = [
            "Alpha beta gamma. Delta epsilon zeta.",
            "Eta theta iota. Kappa lambda mu.",
            "Nu xi omicron. Pi rho sigma.",
            "Tau upsilon phi. Chi psi omega.",
        ]

        with _data_cfg("data/bert/bert_pretraining_onlymlm", updates) as cfg:
            batch, tfds_mock = _build_inputs_from_cfg(
                cfg,
                "trax.data.loader.tf.base.TFDS",
                texts,
            )

        self.assertTrue(tfds_mock.called)
        self.assertLen(batch, 5)
        self.assertEqual(batch[0].shape[0], 2)

    def test_mlp_mnist_make_inputs_smoke(self):
        updates = {
            "Batch.batch_size": 2,
            "train.Stream.shuffle": False,
            "train.Stream.shuffle_buffer_size": 0,
            "train.Stream.seed": 0,
            "eval.Stream.shuffle": False,
            "eval.Stream.shuffle_buffer_size": 0,
            "eval.Stream.seed": 0,
        }
        images = np.arange(16, dtype=np.uint8).reshape(4, 4)
        labels = np.array([0, 1, 2, 3], dtype=np.uint8)
        train_ds = tf.data.Dataset.from_tensor_slices(
            {"image": images, "label": labels}
        )
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {"image": images, "label": labels}
        )
        streams = DatasetStreams(
            train=train_ds,
            eval=eval_ds,
            supervised_keys=(["image"], ["label"]),
            dataset_name="mnist",
            data_dir=None,
        )

        with _data_cfg("data/mlp/mlp_mnist", updates) as cfg:
            with mock.patch(
                "trax.data.loader.tf.base.data_streams",
                return_value=streams,
            ):
                inputs = _make_inputs_from_cfg(cfg)
                stream = inputs.train_stream
                batch = next(stream(1)) if callable(stream) else next(stream)

        self.assertLen(batch, 3)
        self.assertEqual(batch[0].shape[0], 2)

    def test_hourglass_imagenet32_make_inputs_smoke(self):
        updates = {
            "BatchTrain.batch_size": 2,
            "BatchEval.batch_size": 2,
            "train.Stream.shuffle": False,
            "train.Stream.shuffle_buffer_size": 0,
            "train.Stream.seed": 0,
            "eval.Stream.shuffle": False,
            "eval.Stream.shuffle_buffer_size": 0,
            "eval.Stream.seed": 0,
        }
        images = np.arange(4 * 32 * 32 * 3, dtype=np.uint8).reshape(4, 32, 32, 3)
        labels = np.array([0, 1, 2, 3], dtype=np.uint8)
        train_ds = tf.data.Dataset.from_tensor_slices(
            {"image": images, "label": labels}
        )
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {"image": images, "label": labels}
        )
        streams = DatasetStreams(
            train=train_ds,
            eval=eval_ds,
            supervised_keys=(["image"], ["label"]),
            dataset_name="downsampled_imagenet/32x32",
            data_dir=None,
        )

        with _data_cfg("data/hourglass/hourglass_imagenet32", updates) as cfg:
            with mock.patch(
                "trax.data.loader.tf.base.data_streams",
                return_value=streams,
            ):
                _assert_make_inputs_batch(cfg, expected_len=3, expected_batch_size=2)

    def test_hourglass_enwik8_make_inputs_smoke(self):
        updates = {
            "train.Stream.shuffle": False,
            "train.Stream.shuffle_buffer_size": 0,
            "train.Stream.seed": 0,
            "eval.Stream.shuffle": False,
            "eval.Stream.shuffle_buffer_size": 0,
            "eval.Stream.seed": 0,
        }
        targets = np.arange(32, dtype=np.uint8).reshape(4, 8)
        train_ds = tf.data.Dataset.from_tensor_slices(
            {"inputs": targets, "targets": targets}
        )
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {"inputs": targets, "targets": targets}
        )
        streams = DatasetStreams(
            train=train_ds,
            eval=eval_ds,
            supervised_keys=(["targets"], ["targets"]),
            dataset_name="t2t_enwik8_l2k",
            data_dir=None,
        )

        with _data_cfg("data/hourglass/hourglass_enwik8", updates) as cfg:
            with mock.patch(
                "trax.data.loader.tf.base.data_streams",
                return_value=streams,
            ):
                _assert_make_inputs_batch(cfg, expected_len=3)

    def test_hourglass_imagenet64_make_inputs_smoke(self):
        updates = {
            "BatchTrain.batch_size": 2,
            "BatchEval.batch_size": 2,
            "train.Stream.shuffle": False,
            "train.Stream.shuffle_buffer_size": 0,
            "train.Stream.seed": 0,
            "eval.Stream.shuffle": False,
            "eval.Stream.shuffle_buffer_size": 0,
            "eval.Stream.seed": 0,
        }
        images = np.arange(4 * 32 * 32 * 3, dtype=np.uint8).reshape(4, 32, 32, 3)
        labels = np.array([0, 1, 2, 3], dtype=np.uint8)
        train_ds = tf.data.Dataset.from_tensor_slices(
            {"image": images, "label": labels}
        )
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {"image": images, "label": labels}
        )
        streams = DatasetStreams(
            train=train_ds,
            eval=eval_ds,
            supervised_keys=(["image"], ["label"]),
            dataset_name="downsampled_imagenet/64x64",
            data_dir=None,
        )

        with _data_cfg("data/hourglass/hourglass_imagenet64", updates) as cfg:
            with mock.patch(
                "trax.data.loader.tf.base.data_streams",
                return_value=streams,
            ):
                _assert_make_inputs_batch(cfg, expected_len=3, expected_batch_size=2)

    def test_reformer_imagenet64_make_inputs_smoke(self):
        updates = {
            "train.Stream.shuffle": False,
            "train.Stream.shuffle_buffer_size": 0,
            "train.Stream.seed": 0,
            "eval.Stream.shuffle": False,
            "eval.Stream.shuffle_buffer_size": 0,
            "eval.Stream.seed": 0,
        }
        targets = np.arange(32, dtype=np.uint8).reshape(4, 8)
        train_ds = tf.data.Dataset.from_tensor_slices(
            {"inputs": targets, "targets": targets}
        )
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {"inputs": targets, "targets": targets}
        )
        streams = DatasetStreams(
            train=train_ds,
            eval=eval_ds,
            supervised_keys=(["targets"], ["targets"]),
            dataset_name="t2t_image_imagenet64_gen_flat_rev",
            data_dir=None,
        )

        with _data_cfg("data/reformer/reformer_imagenet64", updates) as cfg:
            with mock.patch(
                "trax.data.loader.tf.base.data_streams",
                return_value=streams,
            ):
                _assert_make_inputs_batch(cfg, expected_len=3)

    def test_reformer_imagenet64_testing_make_inputs_smoke(self):
        updates = {
            "train.BucketByLength.boundaries": [8],
            "train.BucketByLength.batch_sizes": [2, 2],
            "eval.BucketByLength.boundaries": [8],
            "eval.BucketByLength.batch_sizes": [2, 2],
            "train.Stream.shuffle": False,
            "train.Stream.shuffle_buffer_size": 0,
            "train.Stream.seed": 0,
            "eval.Stream.shuffle": False,
            "eval.Stream.shuffle_buffer_size": 0,
            "eval.Stream.seed": 0,
        }
        targets = np.arange(16, dtype=np.uint8).reshape(4, 4)
        train_ds = tf.data.Dataset.from_tensor_slices(
            {"inputs": targets, "targets": targets}
        )
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {"inputs": targets, "targets": targets}
        )
        streams = DatasetStreams(
            train=train_ds,
            eval=eval_ds,
            supervised_keys=(["targets"], ["targets"]),
            dataset_name="t2t_image_imagenet64_gen_flat_rev",
            data_dir=None,
        )

        with _data_cfg("data/reformer/reformer_imagenet64_testing", updates) as cfg:
            with mock.patch(
                "trax.data.loader.tf.base.data_streams",
                return_value=streams,
            ):
                _assert_make_inputs_batch(cfg, expected_len=3)

    def test_reformer_enwik8_make_inputs_smoke(self):
        updates = {
            "train.Stream.shuffle": False,
            "train.Stream.shuffle_buffer_size": 0,
            "train.Stream.seed": 0,
            "eval.Stream.shuffle": False,
            "eval.Stream.shuffle_buffer_size": 0,
            "eval.Stream.seed": 0,
        }
        targets = np.arange(32, dtype=np.uint8).reshape(4, 8)
        train_ds = tf.data.Dataset.from_tensor_slices(
            {"inputs": targets, "targets": targets}
        )
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {"inputs": targets, "targets": targets}
        )
        streams = DatasetStreams(
            train=train_ds,
            eval=eval_ds,
            supervised_keys=(["targets"], ["targets"]),
            dataset_name="t2t_enwik8_l65k",
            data_dir=None,
        )

        with _data_cfg("data/reformer/reformer_enwik8", updates) as cfg:
            with mock.patch(
                "trax.data.loader.tf.base.data_streams",
                return_value=streams,
            ):
                _assert_make_inputs_batch(cfg, expected_len=3)

    def test_reformer_bair_robot_pushing_make_inputs_smoke(self):
        updates = {
            "train.Stream.shuffle": False,
            "train.Stream.shuffle_buffer_size": 0,
            "train.Stream.seed": 0,
            "eval.Stream.shuffle": False,
            "eval.Stream.shuffle_buffer_size": 0,
            "eval.Stream.seed": 0,
            "train.Stream.preprocess_fn": None,
            "eval.Stream.preprocess_fn": None,
        }
        targets = np.arange(32, dtype=np.uint8).reshape(4, 8)
        train_ds = tf.data.Dataset.from_tensor_slices(
            {"inputs": targets, "targets": targets}
        )
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {"inputs": targets, "targets": targets}
        )
        streams = DatasetStreams(
            train=train_ds,
            eval=eval_ds,
            supervised_keys=(["targets"], ["targets"]),
            dataset_name="t2t_video_bair_robot_pushing",
            data_dir=None,
        )

        with _data_cfg("data/reformer/reformer_bair_robot_pushing", updates) as cfg:
            with mock.patch(
                "trax.data.loader.tf.base.data_streams",
                return_value=streams,
            ):
                _assert_make_inputs_batch(cfg, expected_len=3)

    def test_resnet50_frn_imagenet_make_inputs_smoke(self):
        updates = {
            "BatchTrain.batch_size": 2,
            "BatchEval.batch_size": 2,
            "train.Stream.shuffle": False,
            "train.Stream.shuffle_buffer_size": 0,
            "train.Stream.seed": 0,
            "eval.Stream.shuffle": False,
            "eval.Stream.shuffle_buffer_size": 0,
            "eval.Stream.seed": 0,
        }
        images = np.arange(4 * 32 * 32 * 3, dtype=np.uint8).reshape(4, 32, 32, 3)
        labels = np.array([0, 1, 2, 3], dtype=np.uint8)
        train_ds = tf.data.Dataset.from_tensor_slices(
            {"image": images, "label": labels}
        )
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {"image": images, "label": labels}
        )
        streams = DatasetStreams(
            train=train_ds,
            eval=eval_ds,
            supervised_keys=(["image"], ["label"]),
            dataset_name="t2t_image_imagenet224",
            data_dir=None,
        )

        with _data_cfg("data/resnet/resnet50_frn_imagenet_8gb", updates) as cfg:
            with mock.patch(
                "trax.data.loader.tf.base.data_streams",
                return_value=streams,
            ):
                _assert_make_inputs_batch(cfg, expected_len=3, expected_batch_size=2)

    def test_scientific_papers_terraformer_pretrained_make_inputs_smoke(self):
        updates = {
            "train.BucketByLength.boundaries": [8],
            "train.BucketByLength.batch_sizes": [2, 1],
            "eval.BucketByLength.boundaries": [8],
            "eval.BucketByLength.batch_sizes": [2, 1],
            "train.Stream.shuffle": False,
            "train.Stream.shuffle_buffer_size": 0,
            "train.Stream.seed": 0,
            "eval.Stream.shuffle": False,
            "eval.Stream.shuffle_buffer_size": 0,
            "eval.Stream.seed": 0,
        }
        updates.update(_tuple_text_stream_overrides())
        article = np.array(
            ["article a", "article b", "article c", "article d"], dtype=object
        )
        abstract = np.array(
            ["abstract a", "abstract b", "abstract c", "abstract d"], dtype=object
        )
        train_ds = tf.data.Dataset.from_tensor_slices(
            {"article": article, "abstract": abstract}
        )
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {"article": article, "abstract": abstract}
        )
        streams = DatasetStreams(
            train=train_ds,
            eval=eval_ds,
            supervised_keys=(["article"], ["abstract"]),
            dataset_name="scientific_papers/arxiv:1.1.1",
            data_dir=None,
        )

        with _data_cfg(
            "data/scientific_papers/scientific_papers_terraformer_pretrained",
            updates,
        ) as cfg:
            with mock.patch(
                "trax.data.loader.tf.base.data_streams",
                return_value=streams,
            ):
                _assert_make_inputs_batch(cfg, expected_len=3)

    def test_scientific_papers_terraformer_favor_make_inputs_smoke(self):
        updates = {
            "train.BucketByLength.boundaries": [8],
            "train.BucketByLength.batch_sizes": [2, 1],
            "eval.BucketByLength.boundaries": [8],
            "eval.BucketByLength.batch_sizes": [2, 1],
            "train.Stream.shuffle": False,
            "train.Stream.shuffle_buffer_size": 0,
            "train.Stream.seed": 0,
            "eval.Stream.shuffle": False,
            "eval.Stream.shuffle_buffer_size": 0,
            "eval.Stream.seed": 0,
        }
        updates.update(_tuple_text_stream_overrides())
        article = np.array(
            ["article a", "article b", "article c", "article d"], dtype=object
        )
        abstract = np.array(
            ["abstract a", "abstract b", "abstract c", "abstract d"], dtype=object
        )
        train_ds = tf.data.Dataset.from_tensor_slices(
            {"article": article, "abstract": abstract}
        )
        eval_ds = tf.data.Dataset.from_tensor_slices(
            {"article": article, "abstract": abstract}
        )
        streams = DatasetStreams(
            train=train_ds,
            eval=eval_ds,
            supervised_keys=(["article"], ["abstract"]),
            dataset_name="scientific_papers/arxiv:1.1.1",
            data_dir=None,
        )

        with _data_cfg(
            "data/scientific_papers/scientific_papers_terraformer_favor",
            updates,
        ) as cfg:
            with mock.patch(
                "trax.data.loader.tf.base.data_streams",
                return_value=streams,
            ):
                _assert_make_inputs_batch(cfg, expected_len=3)

    def test_make_inputs_smoke_batch_configs(self):
        images = np.arange(4 * 32 * 32 * 3, dtype=np.uint8).reshape(4, 32, 32, 3)
        labels = np.array([0, 1, 2, 3], dtype=np.int32).reshape(4, 1)
        image_mapper = _dict_to_tuple("image", "label")
        image_mapper_cfg = _dict_to_tuple_cfg("image", "label")
        imdb_inputs = np.array(
            ["alpha", "beta", "gamma", "delta"], dtype=object
        )
        imdb_labels = np.array([0, 1, 0, 1], dtype=np.int32)
        text_mapper = _dict_to_tuple("text", "label")
        text_mapper_cfg = _dict_to_tuple_cfg("text", "label")

        cases = [
            {
                "config_name": "data/hourglass/hourglass_cifar10",
                "streams": _dataset_streams_from_dicts(
                    "cifar10",
                    {"image": images, "label": labels},
                    {"image": images, "label": labels},
                    (["image"], ["label"]),
                ),
                "updates": {
                    "BatchTrain.batch_size": 2,
                    "BatchEval.batch_size": 2,
                },
                "mapper": image_mapper,
                "mapper_cfg": image_mapper_cfg,
            },
            {
                "config_name": "data/resnet/wide_resnet_cifar10_8gb",
                "streams": _dataset_streams_from_dicts(
                    "cifar10",
                    {"image": images, "label": labels},
                    {"image": images, "label": labels},
                    (["image"], ["label"]),
                ),
                "updates": {
                    "BatchTrain.batch_size": 2,
                    "BatchEval.batch_size": 2,
                },
                "mapper": image_mapper,
                "mapper_cfg": image_mapper_cfg,
            },
            {
                "config_name": "data/resnet/resnet50_imagenet_8gb_testing",
                "streams": _dataset_streams_from_dicts(
                    "t2t_image_imagenet224",
                    {"image": images, "label": labels},
                    {"image": images, "label": labels},
                    (["image"], ["label"]),
                ),
                "updates": {
                    "BatchTrain.batch_size": 2,
                    "BatchEval.batch_size": 2,
                },
                "mapper": image_mapper,
                "mapper_cfg": image_mapper_cfg,
            },
            {
                "config_name": "data/transformer/transformer_imdb_tfds",
                "streams": _dataset_streams_from_dicts(
                    "imdb_reviews",
                    {"text": imdb_inputs, "label": imdb_labels},
                    {"text": imdb_inputs, "label": imdb_labels},
                    (["text"], ["label"]),
                ),
                "updates": {
                    "BatchTrain.batch_size": 2,
                    "BatchEval.batch_size": 2,
                },
                "text_overrides": _text_preprocess_overrides(["inputs"]),
                "mapper": text_mapper,
                "mapper_cfg": text_mapper_cfg,
            },
        ]

        for case in cases:
            with self.subTest(config=case["config_name"]):
                updates = {
                    "train.Stream.shuffle": False,
                    "train.Stream.shuffle_buffer_size": 0,
                    "train.Stream.seed": 0,
                    "eval.Stream.shuffle": False,
                    "eval.Stream.shuffle_buffer_size": 0,
                    "eval.Stream.seed": 0,
                    "train.Stream.preprocess_fn": case["mapper_cfg"],
                    "eval.Stream.preprocess_fn": case["mapper_cfg"],
                }
                updates.update(case["updates"])
                text_overrides = case.get("text_overrides")
                if text_overrides:
                    updates.update(text_overrides)
                self._run_make_inputs_case(
                    case["config_name"],
                    updates,
                    case["streams"],
                    expected_len=3,
                    expected_batch_size=2,
                )

    def test_make_inputs_smoke_lm_buckets(self):
        targets = np.arange(32, dtype=np.int32).reshape(4, 8)
        targets_mapper = _dict_to_tuple("targets", "targets")
        targets_mapper_cfg = _dict_to_tuple_cfg("targets", "targets")
        datasets = [
            ("data/transformer/transformer_big_lm1b_8gb", "t2t_languagemodel_lm1b32k"),
            ("data/transformer/transformer_lm1b_8gb_testing", "t2t_languagemodel_lm1b32k"),
            ("data/transformer/transformer_ptb_16gb", "t2t_languagemodel_ptb10k"),
            ("data/transformer/layerdrop_transformer_lm1b", "t2t_languagemodel_lm1b32k"),
            ("data/transformer/cond_skipping_transformer_lm1b", "t2t_languagemodel_lm1b32k"),
            ("data/transformer/layerdrop_every_transformer_lm1b", "t2t_languagemodel_lm1b32k"),
            ("data/transformer/layerdrop_ushape_transformer_lm1b", "t2t_languagemodel_lm1b32k"),
            ("data/transformer/skipping_transformer_lm1b", "t2t_languagemodel_lm1b32k"),
            ("data/sparse/sparse_lm1b_pretrain_16gb", "t2t_languagemodel_lm1b32k"),
            ("data/lstm/lstm_lm1b", "t2t_languagemodel_lm1b32k"),
        ]

        for config_name, dataset_name in datasets:
            streams = _dataset_streams_from_dicts(
                dataset_name,
                {"inputs": targets, "targets": targets},
                {"inputs": targets, "targets": targets},
                (["inputs"], ["targets"]),
            )
            with self.subTest(config=config_name):
                updates = {
                    "train.Stream.shuffle": False,
                    "train.Stream.shuffle_buffer_size": 0,
                    "train.Stream.seed": 0,
                    "eval.Stream.shuffle": False,
                    "eval.Stream.shuffle_buffer_size": 0,
                    "eval.Stream.seed": 0,
                    "train.Stream.preprocess_fn": targets_mapper_cfg,
                    "eval.Stream.preprocess_fn": targets_mapper_cfg,
                    "train.BucketByLength.bucket_length": 4,
                    "train.BucketByLength.batch_size_per_device": 2,
                    "train.BucketByLength.eval_batch_size": 2,
                    "train.BucketByLength.max_eval_length": 8,
                    "eval.BucketByLength.bucket_length": 4,
                    "eval.BucketByLength.batch_size_per_device": 2,
                    "eval.BucketByLength.eval_batch_size": 2,
                    "eval.BucketByLength.max_eval_length": 8,
                }
                self._run_make_inputs_case(
                    config_name,
                    updates,
                    streams,
                    expected_len=3,
                )

    def test_make_inputs_smoke_bucketed_pairs(self):
        images = np.arange(4 * 32 * 32 * 3, dtype=np.uint8).reshape(4, 32, 32, 3)
        labels = np.array([0, 1, 2, 3], dtype=np.int32)
        inputs = np.array(["input a", "input b", "input c", "input d"], dtype=object)
        targets = np.array(["target a", "target b", "target c", "target d"], dtype=object)
        article = np.array(
            ["article a", "article b", "article c", "article d"], dtype=object
        )
        abstract = np.array(
            ["abstract a", "abstract b", "abstract c", "abstract d"], dtype=object
        )
        image_mapper = _dict_to_tuple("image", "image")
        inputs_mapper = _dict_to_tuple("inputs", "targets")
        article_mapper = _dict_to_tuple("article", "abstract")
        image_mapper_cfg = _dict_to_tuple_cfg("image", "image")
        inputs_mapper_cfg = _dict_to_tuple_cfg("inputs", "targets")
        article_mapper_cfg = _dict_to_tuple_cfg("article", "abstract")

        cases = [
            {
                "config_name": "data/reformer/reformer_cifar10",
                "streams": _dataset_streams_from_dicts(
                    "cifar10",
                    {"image": images, "label": labels},
                    {"image": images, "label": labels},
                    (["image"], ["label"]),
                ),
                "mapper": image_mapper,
                "mapper_cfg": image_mapper_cfg,
                "bucket_updates": {
                    "train.BucketByLength.bucket_length": 4,
                    "train.BucketByLength.batch_size_per_device": 2,
                    "train.BucketByLength.eval_batch_size": 2,
                    "train.BucketByLength.max_eval_length": 8,
                    "eval.BucketByLength.bucket_length": 4,
                    "eval.BucketByLength.batch_size_per_device": 2,
                    "eval.BucketByLength.eval_batch_size": 2,
                    "eval.BucketByLength.max_eval_length": 8,
                },
            },
            {
                "config_name": "data/reformer/reformer_pc_enpl",
                "streams": _dataset_streams_from_dicts(
                    "para_crawl/enpl_plain_text",
                    {"en": inputs, "pl": targets},
                    {"en": inputs, "pl": targets},
                    (["en"], ["pl"]),
                ),
                "mapper": inputs_mapper,
                "mapper_cfg": inputs_mapper_cfg,
                "text_overrides": _text_preprocess_overrides(["inputs", "targets"]),
                "bucket_updates": {
                    "train.BucketByLength.bucket_length": 4,
                    "train.BucketByLength.batch_size_per_device": 2,
                    "train.BucketByLength.eval_batch_size": 2,
                    "train.BucketByLength.max_eval_length": 8,
                    "train.BucketByLength.buckets_include_inputs_in_length": True,
                    "eval.BucketByLength.bucket_length": 4,
                    "eval.BucketByLength.batch_size_per_device": 2,
                    "eval.BucketByLength.eval_batch_size": 2,
                    "eval.BucketByLength.max_eval_length": 8,
                    "eval.BucketByLength.buckets_include_inputs_in_length": True,
                },
            },
            {
                "config_name": "data/scientific_papers/scientific_papers_terraformer",
                "streams": _dataset_streams_from_dicts(
                    "scientific_papers/arxiv:1.1.1",
                    {"article": article, "abstract": abstract},
                    {"article": article, "abstract": abstract},
                    (["article"], ["abstract"]),
                ),
                "mapper": article_mapper,
                "mapper_cfg": article_mapper_cfg,
                "text_overrides": _text_preprocess_overrides(["inputs", "targets"]),
                "bucket_updates": {
                    "train.BucketByLength.buckets": [[4], [2, 1]],
                    "eval.BucketByLength.buckets": [[4], [2, 1]],
                },
            },
            {
                "config_name": "data/transformer/transformer_finetune_squad_16gb",
                "streams": _dataset_streams_from_dicts(
                    "squad/plain_text:1.0.0",
                    {"inputs": inputs, "targets": targets},
                    {"inputs": inputs, "targets": targets},
                    (["inputs"], ["targets"]),
                ),
                "mapper": inputs_mapper,
                "mapper_cfg": inputs_mapper_cfg,
                "bucket_updates": {
                    "train.BucketByLength.buckets": [[4], [2, 2]],
                    "eval.BucketByLength.buckets": [[4], [2, 2]],
                },
            },
        ]

        for case in cases:
            with self.subTest(config=case["config_name"]):
                updates = {
                    "train.Stream.shuffle": False,
                    "train.Stream.shuffle_buffer_size": 0,
                    "train.Stream.seed": 0,
                    "eval.Stream.shuffle": False,
                    "eval.Stream.shuffle_buffer_size": 0,
                    "eval.Stream.seed": 0,
                    "train.Stream.preprocess_fn": case["mapper_cfg"],
                    "eval.Stream.preprocess_fn": case["mapper_cfg"],
                }
                text_overrides = case.get("text_overrides")
                if text_overrides:
                    updates.update(text_overrides)
                updates.update(case["bucket_updates"])
                self._run_make_inputs_case(
                    case["config_name"],
                    updates,
                    case["streams"],
                    expected_len=3,
                )

    def test_make_inputs_smoke_for_remaining_configs(self):
        repo_root = Path(__file__).resolve().parents[5]
        data_root = (
            repo_root
            / "resources"
            / "learning"
            / "supervised"
            / "configs"
            / "yaml"
            / "data"
        )
        skip_paths = {
            data_root / "base.yaml",
            data_root / "loader.yaml",
            data_root / "scientific_papers" / "scientific_papers_base.yaml",
            data_root / "data" / "c4.yaml",
            data_root / "data" / "c4_trax_data.yaml",
        }
        inputs = np.arange(16, dtype=np.int32).reshape(4, 4)
        targets = np.arange(20, dtype=np.int32).reshape(4, 5)

        for path in sorted(data_root.rglob("*.yaml")):
            if path in skip_paths:
                continue
            rel = path.relative_to(data_root)
            config_name = f"data/{rel.with_suffix('').as_posix()}"
            with self.subTest(config=config_name):
                try:
                    ctx = _compose_data_cfg(config_name)
                    cfg = ctx.__enter__()
                except Exception as exc:
                    from hydra.errors import MissingConfigException

                    if isinstance(exc, MissingConfigException):
                        continue
                    raise
                try:
                    if OmegaConf.select(cfg, "data.make_streams") is None:
                        continue
                    if OmegaConf.select(cfg, "data.train.Stream") is None:
                        continue

                    input_key = OmegaConf.select(cfg, "data.train.Stream.input_name") or "inputs"
                    target_key = OmegaConf.select(cfg, "data.train.Stream.target_name") or "targets"
                    input_mapper_cfg = _dict_to_tuple_cfg(input_key, target_key)
                    if input_key == target_key:
                        train_dict = {input_key: inputs}
                        eval_dict = {input_key: inputs}
                        supervised_keys = ([input_key], [target_key])
                    else:
                        train_dict = {input_key: inputs, target_key: targets}
                        eval_dict = {input_key: inputs, target_key: targets}
                        supervised_keys = ([input_key], [target_key])
                    streams = _dataset_streams_from_dicts(
                        "dummy_dataset",
                        train_dict,
                        eval_dict,
                        supervised_keys,
                    )

                    updates = {
                        "train.Stream.shuffle": False,
                        "train.Stream.shuffle_buffer_size": 0,
                        "train.Stream.seed": 0,
                        "eval.Stream.shuffle": False,
                        "eval.Stream.shuffle_buffer_size": 0,
                        "eval.Stream.seed": 0,
                        "train.Stream.preprocess_fn": input_mapper_cfg,
                        "eval.Stream.preprocess_fn": input_mapper_cfg,
                        "Batch.batch_size": 2,
                        "BatchTrain.batch_size": 2,
                        "BatchEval.batch_size": 2,
                        "train.BucketByLength.bucket_length": 4,
                        "train.BucketByLength.batch_size_per_device": 2,
                        "train.BucketByLength.eval_batch_size": 2,
                        "train.BucketByLength.max_eval_length": 8,
                        "eval.BucketByLength.bucket_length": 4,
                        "eval.BucketByLength.batch_size_per_device": 2,
                        "eval.BucketByLength.eval_batch_size": 2,
                        "eval.BucketByLength.max_eval_length": 8,
                    }
                    _apply_cfg_updates_if_present(cfg, updates, prefix="data.")

                    with mock.patch(
                        "trax.data.loader.tf.base.data_streams",
                        return_value=streams,
                    ):
                        inputs_obj = _make_inputs_from_cfg(cfg)
                        stream = inputs_obj.train_stream
                        batch = next(stream(1)) if callable(stream) else next(stream)
                        self.assertTrue(len(batch) >= 2)
                finally:
                    ctx.__exit__(None, None, None)


if __name__ == "__main__":
    absltest.main()
