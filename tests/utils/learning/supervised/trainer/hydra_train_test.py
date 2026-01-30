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

"""Tests for Hydra train configs."""

import sys

from contextlib import contextmanager
from pathlib import Path

from absl import flags
from absl.testing import absltest
from omegaconf import OmegaConf

from trax.utils.learning.supervised.trainer import hydra as hydra_utils

FLAGS = flags.FLAGS


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
def _compose_train_cfg(config_name):
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


def _get_train_node(cfg):
    train_node = OmegaConf.select(cfg, "train")
    if train_node is None:
        return None
    if isinstance(train_node, dict) and "train" in train_node:
        return train_node["train"]
    return train_node


class HydraTrainTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        _ensure_flags_parsed()

    def test_builds_train_configs(self):
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

        for path in sorted(experiment_root.rglob("*.yaml")):
            rel = path.relative_to(experiment_root)
            config_name = f"experiment/{rel.with_suffix('').as_posix()}"
            with self.subTest(config=config_name):
                ctx = _compose_train_cfg(config_name)
                cfg = ctx.__enter__()
                try:
                    train_node = _get_train_node(cfg)
                    if train_node is None:
                        continue
                    train_cfg = OmegaConf.to_container(train_node, resolve=True)
                    self.assertIsInstance(train_cfg, dict)
                finally:
                    ctx.__exit__(None, None, None)
