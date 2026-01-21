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

"""Tests for trainer config helpers."""

import os
import sys

from absl import flags
from absl.testing import absltest

from trax.utils.learning.supervised.trainer import gini as gini_utils
from trax.utils.learning.supervised.trainer import hydra as hydra_utils


def _ensure_output_dir_flag():
    try:
        flags.FLAGS["output_dir"]
    except KeyError:
        flags.DEFINE_string("output_dir", "", "Auto-added test flag: --output_dir")
    try:
        flags.FLAGS(sys.argv)
    except flags.Error:
        flags.FLAGS([sys.argv[0]], known_only=True)


class TrainerConfigTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        _ensure_output_dir_flag()

    def test_gini_output_dir_override(self):
        flag = flags.FLAGS["output_dir"]
        original = flag.value
        try:
            flag.value = "~/tmp/trax_run"
            self.assertEqual(
                gini_utils.output_dir_or_default(),
                os.path.expanduser("~/tmp/trax_run"),
            )
        finally:
            flag.value = original

    def test_hydra_output_dir_override(self):
        flag = flags.FLAGS["output_dir"]
        original = flag.value
        try:
            flag.value = "~/tmp/trax_run_hydra"
            self.assertEqual(
                hydra_utils.output_dir_or_default(cfg={}),
                os.path.expanduser("~/tmp/trax_run_hydra"),
            )
        finally:
            flag.value = original


if __name__ == "__main__":
    absltest.main()
