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

"""Tests for Hydra model configs."""

import inspect
import sys

from contextlib import contextmanager
from pathlib import Path

from absl import flags
from absl.testing import absltest
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from trax import layers as tl
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
def _compose_model_cfg(config_name):
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


def _should_skip_model(target):
    return target in {
        "trax.models.data_streams",
    }


def _maybe_build(model_obj):
    if isinstance(model_obj, tl.Layer):
        return model_obj
    if callable(model_obj):
        try:
            sig = inspect.signature(model_obj)
            required = [
                param
                for param in sig.parameters.values()
                if param.default is param.empty
                and param.kind
                in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
            ]
            if not required:
                return model_obj()
        except (TypeError, ValueError):
            return model_obj
    return model_obj


class HydraModelTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        _ensure_flags_parsed()

    def test_builds_models_from_yaml(self):
        repo_root = Path(__file__).resolve().parents[5]
        model_root = (
            repo_root
            / "resources"
            / "learning"
            / "supervised"
            / "configs"
            / "yaml"
            / "model"
        )
        skip_paths = {
            model_root / "base.yaml",
        }

        for path in sorted(model_root.rglob("*.yaml")):
            if path in skip_paths:
                continue
            rel = path.relative_to(model_root)
            config_name = f"model/{rel.with_suffix('').as_posix()}"
            with self.subTest(config=config_name):
                ctx = _compose_model_cfg(config_name)
                cfg = ctx.__enter__()
                try:
                    model_cfg = OmegaConf.select(cfg, "model")
                    if model_cfg is None:
                        continue
                    for key, value in model_cfg.items():
                        if not isinstance(value, DictConfig):
                            continue
                        target = value.get("_target_")
                        if not target or _should_skip_model(target):
                            continue
                        model_obj = instantiate(value)
                        built = _maybe_build(model_obj)
                        self.assertIsNotNone(
                            built,
                            msg=f"Model '{key}' did not build for {config_name}",
                        )
                finally:
                    ctx.__exit__(None, None, None)
