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

"""Tests for T5 preprocessors."""

import numpy as np
from absl.testing import absltest

from trax.data.preprocessing.modules import t5 as serial_t5


class T5PreprocessTest(absltest.TestCase):
    def test_select_random_chunk_t5_returns_targets(self):
        tokens = np.arange(10, dtype=np.int32)
        processed = list(
            serial_t5.select_random_chunk_t5(sequence_length=20)(
                iter([{"inputs": tokens}])
            )
        )

        self.assertLen(processed, 1)
        np.testing.assert_array_equal(processed[0]["inputs"], tokens)
        np.testing.assert_array_equal(processed[0]["targets"], tokens)

    def test_split_tokens_t5_halves(self):
        tokens = np.arange(6, dtype=np.int32)
        processed = list(
            serial_t5.split_tokens_t5(sequence_length=6)(iter([{"inputs": tokens}]))
        )

        self.assertLen(processed, 1)
        np.testing.assert_array_equal(processed[0]["inputs"], tokens[:3])
        np.testing.assert_array_equal(processed[0]["targets"], tokens[3:])

    def test_denoise_t5_noise_zero(self):
        tokens = np.arange(8, dtype=np.int32)
        processed = list(
            serial_t5.denoise_t5(sequence_length=8, noise_density=0.0)(
                iter([{"inputs": tokens}])
            )
        )

        self.assertLen(processed, 1)
        np.testing.assert_array_equal(processed[0]["inputs"], tokens)
        np.testing.assert_array_equal(processed[0]["targets"], tokens)

    def test_denoise_t5_noise_full(self):
        tokens = np.arange(5, dtype=np.int32)
        processed = list(
            serial_t5.denoise_t5(sequence_length=5, noise_density=1.0)(
                iter([{"inputs": tokens}])
            )
        )

        self.assertLen(processed, 1)
        np.testing.assert_array_equal(processed[0]["inputs"], np.zeros_like(tokens))
        np.testing.assert_array_equal(processed[0]["targets"], tokens)


if __name__ == "__main__":
    absltest.main()
