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

"""Tests for Hydra optimizer configs."""

import inspect
import sys

from contextlib import contextmanager
from pathlib import Path

from absl import flags
from absl.testing import absltest
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

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
def _compose_optim_cfg(config_name):
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


def _maybe_build(optim_obj):
    if callable(optim_obj):
        try:
            sig = inspect.signature(optim_obj)
            required = [
                param
                for param in sig.parameters.values()
                if param.default is param.empty
                and param.kind
                in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
            ]
            if not required:
                return optim_obj()
        except (TypeError, ValueError):
            return optim_obj
    return optim_obj


class HydraOptimTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        _ensure_flags_parsed()

    def test_builds_optimizers_from_yaml(self):
        repo_root = Path(__file__).resolve().parents[5]
        optim_root = (
            repo_root
            / "resources"
            / "learning"
            / "supervised"
            / "configs"
            / "yaml"
            / "optim"
        )
        skip_paths = {
            optim_root / "base.yaml",
        }

        for path in sorted(optim_root.rglob("*.yaml")):
            if path in skip_paths:
                continue
            rel = path.relative_to(optim_root)
            config_name = f"optim/{rel.with_suffix('').as_posix()}"
            with self.subTest(config=config_name):
                ctx = _compose_optim_cfg(config_name)
                cfg = ctx.__enter__()
                try:
                    optim_cfg = OmegaConf.select(cfg, "optim")
                    if optim_cfg is None:
                        continue
                    for key, value in optim_cfg.items():
                        if not isinstance(value, DictConfig):
                            continue
                        target = value.get("_target_")
                        if not target:
                            continue
                        optim_obj = instantiate(value)
                        built = _maybe_build(optim_obj)
                        self.assertIsNotNone(
                            built,
                            msg=f"Optimizer '{key}' did not build for {config_name}",
                        )
                finally:
                    ctx.__exit__(None, None, None)
