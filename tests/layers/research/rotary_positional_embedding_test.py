# coding=utf-8
# Copyright 2026 The Trax Authors.
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

"""Tests for rotary positional embeddings."""

import numpy as np

from absl.testing import absltest

from trax.layers.research import rotary_positional_embedding as rope


class RotaryPositionalEmbeddingTest(absltest.TestCase):
    def test_apply_rotary_embedding_shapes(self):
        q = np.zeros((2, 4, 8, 16), dtype=np.float32)
        k = np.zeros((2, 4, 8, 16), dtype=np.float32)
        q_rot, k_rot = rope.apply_rotary_embedding(q, k, rotary_dim=8, base=10000.0)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)

    def test_apply_rotary_embedding_zero_input(self):
        q = np.zeros((1, 2, 4, 8), dtype=np.float32)
        k = np.zeros((1, 2, 4, 8), dtype=np.float32)
        q_rot, k_rot = rope.apply_rotary_embedding(q, k)
        np.testing.assert_allclose(q_rot, q, atol=1e-6)
        np.testing.assert_allclose(k_rot, k, atol=1e-6)


if __name__ == "__main__":
    absltest.main()
