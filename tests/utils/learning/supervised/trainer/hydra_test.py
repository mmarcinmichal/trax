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

"""End-to-end Hydra trainer tests."""

import sys

from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from absl import flags
from absl.testing import absltest
from hydra.utils import instantiate
from omegaconf import OmegaConf

import numpy as np

from trax import data as trax_data
from trax import fastmath

from trax.utils.learning.supervised.trainer import hydra as hydra_utils

FLAGS = flags.FLAGS

E2E_EXPERIMENTS = {
    "experiment/mlp/mlp_mnist": [
        "+train.train.steps=2",
        "+train.train.eval_frequency=1",
        "+train.train.eval_steps=1",
        "+data.Batch.batch_size=4",
    ],
    "experiment/lstm/lstm_lm1b": [
        "+train.train.steps=2",
        "+train.train.eval_frequency=1",
        "+train.train.eval_steps=1",
        "+data.train.LM1BFilterByLength.max_target_length=16",
        "+data.eval.LM1BFilterByLength.max_target_length=16",
        "+data.train.BucketByLength.batch_size_per_device=2",
        "+data.eval.BucketByLength.batch_size_per_device=2",
        "+data.train.BucketByLength.eval_batch_size=2",
        "+data.eval.BucketByLength.eval_batch_size=2",
        "+data.train.BucketByLength.max_eval_length=16",
        "+data.eval.BucketByLength.max_eval_length=16",
    ],
}


def _ensure_flags_parsed():
    try:
        flags.FLAGS(sys.argv)
    except flags.Error:
        flags.FLAGS([sys.argv[0]], known_only=True)


def _repo_config_dir():
    repo_root = Path(__file__).resolve().parents[5]
    return str(
        repo_root / "resources" / "learning" / "supervised" / "configs" / "yaml"
    )


@contextmanager
def _compose_experiment_cfg_with_overrides(config_name, overrides):
    config_name_flag = FLAGS["hydra_config_name"]
    overrides_flag = FLAGS["hydra_overrides"]
    config_dir_flag = FLAGS["hydra_config_dir"]
    original_name = config_name_flag.value
    original_overrides = overrides_flag.value
    original_dir = config_dir_flag.value
    try:
        config_name_flag.value = config_name
        overrides_flag.value = overrides
        config_dir_flag.value = _repo_config_dir()
        cfg = hydra_utils.compose_config()
        yield cfg
    finally:
        config_name_flag.value = original_name
        overrides_flag.value = original_overrides
        config_dir_flag.value = original_dir


def _select_experiment_root(cfg):
    experiment_node = OmegaConf.select(cfg, "experiment")
    if experiment_node is None:
        return cfg
    if OmegaConf.is_config(experiment_node):
        keys = list(experiment_node.keys())
        if len(keys) == 1:
            inner = experiment_node.get(keys[0])
            if inner is not None:
                return inner
    return experiment_node


def _flatten_nested_group(root, key):
    node = OmegaConf.select(root, key)
    if node is None or not OmegaConf.is_config(node):
        return
    OmegaConf.set_struct(node, False)
    while OmegaConf.is_config(node) and key in node:
        child = node.get(key)
        if not OmegaConf.is_config(child):
            break
        OmegaConf.set_struct(child, False)
        child_container = OmegaConf.to_container(child, resolve=False)
        if not isinstance(child_container, dict):
            break
        for child_key, child_value in child_container.items():
            if child_key not in node:
                node[child_key] = child_value
        node = child


def _hoist_nested_data(cfg):
    data_node = OmegaConf.select(cfg, "data")
    if data_node is None or not OmegaConf.is_config(data_node):
        return
    OmegaConf.set_struct(data_node, False)
    while True:
        nested = data_node.get("data")
        if not OmegaConf.is_config(nested):
            break
        OmegaConf.set_struct(nested, False)
        nested_container = OmegaConf.to_container(nested, resolve=False)
        if not isinstance(nested_container, dict):
            break
        for key, value in nested_container.items():
            if key not in data_node:
                data_node[key] = value
        data_node = nested


def _hoist_nested_group(cfg, group_key, nested_key):
    group_node = OmegaConf.select(cfg, group_key)
    if group_node is None or not OmegaConf.is_config(group_node):
        return
    OmegaConf.set_struct(group_node, False)
    while True:
        nested = group_node.get(nested_key)
        if not OmegaConf.is_config(nested):
            break
        OmegaConf.set_struct(nested, False)
        nested_container = OmegaConf.to_container(nested, resolve=False)
        if not isinstance(nested_container, dict):
            break
        for key, value in nested_container.items():
            if key not in group_node:
                group_node[key] = value
        group_node = nested


def _ensure_model_and_optimizer(cfg):
    model_node = OmegaConf.select(cfg, "model.model_fn")
    if model_node is None:
        for candidate in ("model.MLP", "model.RNNLM", "model.BERT"):
            candidate_node = OmegaConf.select(cfg, candidate)
            if candidate_node is not None:
                OmegaConf.update(cfg, "model.model_fn", candidate_node, merge=False)
                break

    optimizer_node = OmegaConf.select(cfg, "optim.optimizer")
    if optimizer_node is None or (
        OmegaConf.is_config(optimizer_node) and "_target_" not in optimizer_node
    ):
        for candidate in (
            "optim.optim.optim.Adam",
            "optim.optim.optim.Adafactor",
            "optim.optim.Adam",
            "optim.optim.Adafactor",
            "optim.Adam",
            "optim.Adafactor",
        ):
            candidate_node = OmegaConf.select(cfg, candidate)
            if candidate_node is not None:
                OmegaConf.update(cfg, "optim.optimizer", candidate_node, merge=False)
                break


def _promote_optimizer_targets(cfg):
    for name in ("Adam", "Adafactor"):
        target_node = OmegaConf.select(cfg, f"optim.optim.optim.{name}")
        if target_node is None or "_target_" not in target_node:
            continue
        existing = OmegaConf.select(cfg, f"optim.{name}")
        if existing is None:
            OmegaConf.update(cfg, f"optim.{name}", target_node, merge=False)
            continue
        if OmegaConf.is_config(existing) and "_target_" not in existing:
            OmegaConf.update(cfg, f"optim.{name}._target_", target_node["_target_"], merge=False)


def _flatten_train_node(cfg):
    train_node = OmegaConf.select(cfg, "train")
    if train_node is None:
        return
    while (
        OmegaConf.is_config(train_node)
        and "train" in train_node
        and len(train_node) == 1
    ):
        train_node = train_node.get("train")
    if OmegaConf.is_config(train_node) and "train" in train_node and len(train_node) > 1:
        train_container = OmegaConf.to_container(train_node, resolve=False)
        if isinstance(train_container, dict):
            train_container.pop("train", None)
            train_node = OmegaConf.create(
                train_container, flags={"allow_objects": True}
            )
    if train_node is not None:
        OmegaConf.update(cfg, "train", train_node, merge=False)


def _build_stream_bundle(exp_cfg):
    streams = {"train_stream": None, "eval_stream": None, "train_eval_stream": None}
    for stream_key in streams:
        stream_node = OmegaConf.select(exp_cfg, f"data.make_streams.{stream_key}")
        if stream_node is None:
            continue
        if OmegaConf.is_config(stream_node):
            stream_node = OmegaConf.to_container(stream_node, resolve=False)
        if stream_node is None:
            continue
        stream_list = []
        for item in stream_node:
            if isinstance(item, str) and item.startswith("${") and item.endswith("}"):
                ref = item[2:-1]
                resolved = OmegaConf.select(exp_cfg, ref)
                if resolved is not None:
                    item = resolved
            if OmegaConf.is_config(item):
                item = OmegaConf.to_container(item, resolve=False)
            if isinstance(item, dict) and "_target_" in item:
                item = instantiate(item)
            stream_list.append(item)
        streams[stream_key] = stream_list
    if streams["train_stream"] is None:
        return None
    return trax_data.make_streams(
        train_stream=streams["train_stream"],
        eval_stream=streams["eval_stream"],
        train_eval_stream=streams["train_eval_stream"],
    )


def _fake_tfds(*_args, **kwargs):
    dataset_name = kwargs.get("dataset_name", "")
    keys = kwargs.get("keys")

    if "mnist" in dataset_name or keys:
        image = np.zeros((28, 28, 1), dtype=np.float32)
        label = np.int32(0)

        def _gen(_=None):
            while True:
                yield (image, label)

        return _gen

    if "lm1b" in dataset_name or "languagemodel" in dataset_name:
        tokens = np.arange(8, dtype=np.int32)

        def _gen(_=None):
            while True:
                yield {"inputs": tokens, "targets": tokens}

        return _gen

    def _gen(_=None):
        while True:
            yield "dummy"

    return _gen


class HydraEndToEndTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        _ensure_flags_parsed()

    def test_end_to_end_training(self):
        fastmath.disable_jit()
        with mock.patch(
            "trax.data.loader.tf.base.TFDS", side_effect=_fake_tfds
        ), mock.patch(
            "trax.data.TFDS", side_effect=_fake_tfds
        ), mock.patch(
            "trax.data.preprocessing.modules.bert.TFDS", side_effect=_fake_tfds
        ):
            for config_name, overrides in E2E_EXPERIMENTS.items():
                with self.subTest(config=config_name):
                    with _compose_experiment_cfg_with_overrides(
                        config_name, overrides
                    ) as cfg:
                        exp_root = _select_experiment_root(cfg)
                        for key in (
                            "data",
                            "model",
                            "optim",
                            "schedule",
                            "train",
                            "ckpt",
                        ):
                            _flatten_nested_group(exp_root, key)
                        exp_cfg = OmegaConf.create(
                            OmegaConf.to_container(exp_root, resolve=False),
                            flags={"allow_objects": True},
                        )
                        _hoist_nested_data(exp_cfg)
                        OmegaConf.update(
                            exp_cfg, "data.make_streams._partial_", False, merge=False
                        )
                        stream_bundle = _build_stream_bundle(exp_cfg)
                        if stream_bundle is not None:
                            OmegaConf.update(
                                exp_cfg, "train.inputs", stream_bundle, merge=False
                            )
                            OmegaConf.update(
                                exp_cfg, "data.make_streams", None, merge=False
                            )
                        _hoist_nested_group(exp_cfg, "optim", "optim")
                        _hoist_nested_group(exp_cfg, "model", "model")
                        _hoist_nested_group(exp_cfg, "schedule", "schedule")
                        _promote_optimizer_targets(exp_cfg)
                        _ensure_model_and_optimizer(exp_cfg)
                        _flatten_train_node(exp_cfg)
                        OmegaConf.update(exp_cfg, "train.steps", 1, merge=False)
                        OmegaConf.update(exp_cfg, "train.eval_frequency", 1, merge=False)
                        OmegaConf.update(exp_cfg, "train.eval_steps", 1, merge=False)
                        OmegaConf.update(exp_cfg, "ckpt", {}, merge=False)

                        output_dir = self.create_tempdir().full_path
                        hydra_utils.train_with_hydra(exp_cfg, output_dir)

                        output_path = Path(output_dir)
                        self.assertTrue(output_path.exists())
                        self.assertTrue(
                            any(output_path.iterdir()),
                            msg="Expected training artifacts in output_dir.",
                        )
