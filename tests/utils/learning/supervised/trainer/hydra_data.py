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

from unittest import mock

from absl import flags
from absl.testing import absltest
from hydra.utils import instantiate

from trax.utils.learning.supervised.trainer import hydra as hydra_utils

FLAGS = flags.FLAGS


def _ensure_flags_parsed():
    try:
        flags.FLAGS(sys.argv)
    except flags.Error:
        flags.FLAGS([sys.argv[0]], known_only=True)


def _fake_tfds_stream(texts):
    def _tfds(*args, **kwargs):
        def _gen(_=None):
            for text in texts:
                yield text

        return _gen

    return _tfds


class HydraDataTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        _ensure_flags_parsed()

    def test_bert_pretraining_builds_streams(self):
        overrides = [
            "experiment=bert_pretraining",
            "data.Tokenize.vocab_type=char",
            "data.Tokenize.keys=[0,1]",
            "data.PadToLength.len_map={0:16,1:16,2:16}",
            "data.PadToLength.pad_value={0:0,1:0,2:0}",
            "data.Batch.batch_size=2",
            "data.Shuffle.queue_size=2",
        ]
        texts = [
            "Alpha beta gamma. Delta epsilon zeta.",
            "Eta theta iota. Kappa lambda mu.",
            "Nu xi omicron. Pi rho sigma.",
            "Tau upsilon phi. Chi psi omega.",
        ]

        config_name_flag = FLAGS["hydra_config_name"]
        overrides_flag = FLAGS["hydra_overrides"]
        original_name = config_name_flag.value
        original_overrides = overrides_flag.value
        try:
            config_name_flag.value = "config"
            overrides_flag.value = overrides
            cfg = hydra_utils.compose_config()

            with mock.patch(
                "trax.data.preprocessing.modules.bert.TFDS",
                side_effect=_fake_tfds_stream(texts),
            ) as tfds_mock:
                inputs = instantiate(cfg.data.make_inputs)
                batch = next(inputs.train_stream(1))

            self.assertTrue(tfds_mock.called)
            self.assertLen(batch, 7)
            self.assertEqual(batch[0].shape[0], 2)
        finally:
            config_name_flag.value = original_name
            overrides_flag.value = original_overrides


if __name__ == "__main__":
    absltest.main()
