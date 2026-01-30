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

"""Tests for Hydra checkpoint configs."""

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
def _compose_ckpt_cfg(config_name):
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


def _normalize_ckpt_cfg(cfg):
    ckpt_node = OmegaConf.select(cfg, "ckpt")
    if ckpt_node is None:
        return None
    container = OmegaConf.to_container(ckpt_node, resolve=True)
    if isinstance(container, dict) and "ckpt" in container:
        return container.get("ckpt", {})
    return container


class HydraCkptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        _ensure_flags_parsed()

    def test_builds_ckpt_configs(self):
        repo_root = Path(__file__).resolve().parents[5]
        ckpt_root = (
            repo_root
            / "resources"
            / "learning"
            / "supervised"
            / "configs"
            / "yaml"
            / "ckpt"
        )
        skip_paths = {
            ckpt_root / "base.yaml",
        }

        for path in sorted(ckpt_root.rglob("*.yaml")):
            if path in skip_paths:
                continue
            rel = path.relative_to(ckpt_root)
            config_name = f"ckpt/{rel.with_suffix('').as_posix()}"
            with self.subTest(config=config_name):
                ctx = _compose_ckpt_cfg(config_name)
                cfg = ctx.__enter__()
                try:
                    ckpt_cfg = _normalize_ckpt_cfg(cfg)
                    self.assertIsInstance(ckpt_cfg, dict)
                finally:
                    ctx.__exit__(None, None, None)
