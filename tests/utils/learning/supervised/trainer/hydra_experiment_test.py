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

"""Tests for full Hydra experiment configs."""

import sys

from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from absl import flags
from absl.testing import absltest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from trax.utils.learning.supervised.trainer import hydra as hydra_utils

FLAGS = flags.FLAGS

# Add more experiment configs here to exercise full object instantiation.
FULL_INIT_EXPERIMENTS = (
    "experiment/bert/bert",
    "experiment/mlp/mlp_mnist",
    "experiment/lstm/lstm_lm1b",
)

FULL_INIT_OVERRIDES = {
    "experiment/bert/bert": [
        "+model.BERT.init_checkpoint=null",
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
def _compose_experiment_cfg(config_name):
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
        cfg = hydra_utils.compose_config()
        yield cfg
    finally:
        config_name_flag.value = original_name
        overrides_flag.value = original_overrides
        config_dir_flag.value = original_dir


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


def _fake_tfds(*_args, **kwargs):
    keys = kwargs.get("keys")

    def _gen(_=None):
        if keys:
            yield tuple("dummy" for _ in keys)
        else:
            yield "dummy"

    return _gen


def _get_train_node(cfg):
    train_node = OmegaConf.select(cfg, "train")
    if train_node is None:
        train_node = OmegaConf.select(cfg, "experiment.train")
    if train_node is None:
        train_node = OmegaConf.select(_select_experiment_root(cfg), "train")
    if train_node is None:
        return None
    if isinstance(train_node, dict) and "train" in train_node:
        return train_node["train"]
    return train_node


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


def _find_make_streams_node(cfg):
    root = _select_experiment_root(cfg)
    for path in (
        "data.make_streams",
        "data.data.make_streams",
        "data.bert.data.make_streams",
        "data.bert.data.data.make_streams",
    ):
        node = OmegaConf.select(root, path)
        if node is not None:
            return node
    data_node = OmegaConf.select(root, "data")
    if not isinstance(data_node, dict) and not OmegaConf.is_config(data_node):
        return None
    container = OmegaConf.to_container(data_node, resolve=False)
    stack = [container] if isinstance(container, dict) else []
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        if "make_streams" in current:
            return current.get("make_streams")
        for value in current.values():
            if isinstance(value, dict):
                stack.append(value)
    return None


def _select_cfg(cfg, path):
    node = OmegaConf.select(cfg, path)
    if node is None:
        node = OmegaConf.select(cfg, f"experiment.{path}")
    if node is None:
        node = OmegaConf.select(_select_experiment_root(cfg), path)
    return node


def _extract_interpolation(value):
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return value[2:-1]
    return None


def _find_first_target_node(cfg, base_key):
    base_node = OmegaConf.select(cfg, base_key)
    if base_node is None:
        return None
    container = OmegaConf.to_container(base_node, resolve=False)
    if not isinstance(container, dict):
        return None
    stack = [container]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        if "_target_" in current:
            return current
        for value in current.values():
            if isinstance(value, dict):
                stack.append(value)
    return None


def _has_target(node):
    if node is None:
        return False
    if OmegaConf.is_config(node):
        return "_target_" in node
    if isinstance(node, dict):
        return "_target_" in node
    return False


def _instantiate_if_target(node):
    if _has_target(node):
        return instantiate(node)
    return None


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


def _resolve_optimizer_node(cfg, optimizer_ref):
    if optimizer_ref:
        node = _select_cfg(cfg, optimizer_ref)
        if _has_target(node):
            return node
        if optimizer_ref.startswith("optim."):
            name = optimizer_ref.split(".", 1)[1]
            node = _select_cfg(cfg, f"optim.optim.{name}")
            if _has_target(node):
                return node
    return None


class HydraExperimentTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        _ensure_flags_parsed()

    def test_builds_experiment_pipelines(self):
        repo_root = Path(__file__).resolve().parents[5]
        experiment_root = (
            repo_root
            / "resources"
            / "learning"
            / "supervised"
            / "configs"
            / "yaml"
            / "experiment"
        )

        with mock.patch(
            "trax.data.loader.tf.base.TFDS", side_effect=_fake_tfds
        ), mock.patch(
            "trax.data.TFDS", side_effect=_fake_tfds
        ), mock.patch(
            "trax.data.preprocessing.modules.bert.TFDS", side_effect=_fake_tfds
        ):
            for path in sorted(experiment_root.rglob("*.yaml")):
                rel = path.relative_to(experiment_root)
                config_name = f"experiment/{rel.with_suffix('').as_posix()}"
                with self.subTest(config=config_name):
                    ctx = _compose_experiment_cfg(config_name)
                    cfg = ctx.__enter__()
                    try:
                        exp_root = _select_experiment_root(cfg)
                        for key in ("data", "model", "optim", "schedule", "train", "ckpt"):
                            _flatten_nested_group(exp_root, key)
                        exp_cfg = OmegaConf.create(
                            OmegaConf.to_container(exp_root, resolve=False)
                        )
                        _hoist_nested_data(exp_cfg)

                        train_node = _get_train_node(exp_cfg)
                        if train_node is not None:
                            train_cfg = OmegaConf.to_container(
                                train_node, resolve=False
                            )
                            while (
                                isinstance(train_cfg, dict)
                                and "train" in train_cfg
                                and len(train_cfg) == 1
                            ):
                                train_cfg = train_cfg.get("train", {})
                        else:
                            train_cfg = {}
                        if not isinstance(train_cfg, dict):
                            train_cfg = {}

                        inputs = None
                        inputs_ref = _extract_interpolation(train_cfg.get("inputs"))
                        if inputs_ref:
                            inputs_node = _select_cfg(exp_cfg, inputs_ref)
                            if _has_target(inputs_node):
                                inputs = inputs_node
                        if inputs is None:
                            make_streams_node = _find_make_streams_node(exp_cfg)
                            if _has_target(make_streams_node):
                                inputs = make_streams_node

                        model = None
                        model_ref = _extract_interpolation(train_cfg.get("model"))
                        if model_ref:
                            model_node = _select_cfg(exp_cfg, model_ref)
                            if _has_target(model_node):
                                model = model_node
                        if model is None:
                            model_node = _select_cfg(exp_cfg, "model.model_fn")
                            if model_node is None:
                                model_node = _find_first_target_node(exp_cfg, "model")
                            if _has_target(model_node):
                                model = model_node

                        optimizer = None
                        optimizer_ref = _extract_interpolation(train_cfg.get("optimizer"))
                        if optimizer_ref:
                            optim_node = _select_cfg(exp_cfg, optimizer_ref)
                            if _has_target(optim_node):
                                optimizer = optim_node
                        if optimizer is None:
                            optim_node = _select_cfg(exp_cfg, "optim.optimizer")
                            if optim_node is None:
                                optim_node = _find_first_target_node(exp_cfg, "optim")
                            if _has_target(optim_node):
                                optimizer = optim_node

                        lr_schedule_fn = None
                        schedule_ref = _extract_interpolation(
                            train_cfg.get("lr_schedule_fn")
                        )
                        if schedule_ref:
                            schedule_node = _select_cfg(exp_cfg, schedule_ref)
                            if _has_target(schedule_node):
                                lr_schedule_fn = schedule_node
                        if lr_schedule_fn is None:
                            schedule_node = _select_cfg(exp_cfg, "schedule.lr_schedule_fn")
                            if _has_target(schedule_node):
                                lr_schedule_fn = schedule_node
                            else:
                                schedule_node = _select_cfg(exp_cfg, "schedule")
                                if isinstance(schedule_node, dict):
                                    if "multifactor" in schedule_node:
                                        if _has_target(schedule_node["multifactor"]):
                                            lr_schedule_fn = schedule_node["multifactor"]
                                    elif len(schedule_node) == 1:
                                        schedule_value = next(iter(schedule_node.values()))
                                        if _has_target(schedule_value):
                                            lr_schedule_fn = schedule_value

                        ckpt_node = _select_cfg(exp_cfg, "ckpt")
                        ckpt_cfg = (
                            OmegaConf.to_container(ckpt_node, resolve=True)
                            if ckpt_node is not None
                            else {}
                        )
                        if isinstance(ckpt_cfg, dict) and "ckpt" in ckpt_cfg:
                            ckpt_cfg = ckpt_cfg.get("ckpt", {})

                        inputs_expected = _find_make_streams_node(exp_cfg) or _find_first_target_node(exp_cfg, "data")
                        if inputs is None and (inputs_ref or _has_target(inputs_expected)):
                            self.assertIsNotNone(inputs)

                        model_expected = _find_first_target_node(exp_cfg, "model")
                        if model is None and (model_ref or model_expected is not None):
                            self.assertIsNotNone(model)
                        optim_expected = _find_first_target_node(exp_cfg, "optim")
                        if optimizer is None and (optim_expected is not None or optimizer_ref):
                            self.assertIsNotNone(optimizer)
                        if ckpt_cfg is not None:
                            self.assertIsInstance(ckpt_cfg, dict)
                        if lr_schedule_fn is not None:
                            self.assertTrue(_has_target(lr_schedule_fn))
                    finally:
                        ctx.__exit__(None, None, None)

    def test_instantiates_selected_experiments(self):
        with mock.patch(
            "trax.data.loader.tf.base.TFDS", side_effect=_fake_tfds
        ), mock.patch(
            "trax.data.TFDS", side_effect=_fake_tfds
        ), mock.patch(
            "trax.data.preprocessing.modules.bert.TFDS", side_effect=_fake_tfds
        ), mock.patch(
            "trax.models.BERT", side_effect=lambda *args, **kwargs: object()
        ):
            for config_name in FULL_INIT_EXPERIMENTS:
                with self.subTest(config=config_name):
                    overrides = FULL_INIT_OVERRIDES.get(config_name, [])
                    ctx = _compose_experiment_cfg_with_overrides(
                        config_name, overrides
                    )
                    cfg = ctx.__enter__()
                    try:
                        exp_root = _select_experiment_root(cfg)
                        for key in ("data", "model", "optim", "schedule", "train", "ckpt"):
                            _flatten_nested_group(exp_root, key)
                        exp_cfg = OmegaConf.create(
                            OmegaConf.to_container(exp_root, resolve=False)
                        )
                        _hoist_nested_data(exp_cfg)

                        train_node = _get_train_node(exp_cfg)
                        if train_node is not None:
                            train_cfg = OmegaConf.to_container(
                                train_node, resolve=False
                            )
                            while (
                                isinstance(train_cfg, dict)
                                and "train" in train_cfg
                                and len(train_cfg) == 1
                            ):
                                train_cfg = train_cfg.get("train", {})
                        else:
                            train_cfg = {}
                        if not isinstance(train_cfg, dict):
                            train_cfg = {}

                        inputs_node = None
                        inputs_ref = _extract_interpolation(train_cfg.get("inputs"))
                        if inputs_ref:
                            inputs_node = _select_cfg(exp_cfg, inputs_ref)
                        if inputs_node is None:
                            inputs_node = _find_make_streams_node(exp_cfg)
                        inputs = _instantiate_if_target(inputs_node)

                        model_node = None
                        model_ref = _extract_interpolation(train_cfg.get("model"))
                        if model_ref:
                            model_node = _select_cfg(exp_cfg, model_ref)
                        if model_node is None:
                            model_node = _select_cfg(exp_cfg, "model.model_fn")
                        if model_node is None:
                            model_node = _find_first_target_node(exp_cfg, "model")
                        model = _instantiate_if_target(model_node)

                        optim_node = None
                        optimizer_ref = _extract_interpolation(train_cfg.get("optimizer"))
                        if optimizer_ref:
                            optim_node = _resolve_optimizer_node(exp_cfg, optimizer_ref)
                        if optim_node is None:
                            optim_node = _select_cfg(exp_cfg, "optim.optimizer")
                        if optim_node is None:
                            optim_node = _find_first_target_node(exp_cfg, "optim")
                        optimizer = _instantiate_if_target(optim_node)

                        schedule_node = None
                        schedule_ref = _extract_interpolation(
                            train_cfg.get("lr_schedule_fn")
                        )
                        if schedule_ref:
                            schedule_node = _select_cfg(exp_cfg, schedule_ref)
                        if schedule_node is None:
                            schedule_node = _select_cfg(exp_cfg, "schedule.lr_schedule_fn")
                        if schedule_node is None:
                            schedule_node = _select_cfg(exp_cfg, "schedule")
                            if isinstance(schedule_node, dict) and len(schedule_node) == 1:
                                schedule_node = next(iter(schedule_node.values()))
                        lr_schedule_fn = _instantiate_if_target(schedule_node)

                        self.assertIsNotNone(inputs)
                        self.assertIsNotNone(model)
                        self.assertIsNotNone(optimizer)
                        if lr_schedule_fn is not None:
                            self.assertTrue(callable(lr_schedule_fn))
                    finally:
                        ctx.__exit__(None, None, None)
