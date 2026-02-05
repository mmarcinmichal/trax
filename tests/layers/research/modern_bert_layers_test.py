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

"""Tests for ModernBERT helper layers."""

import numpy as np

from absl.testing import absltest

from trax.layers.research import modern_bert_layers
from trax.utils import shapes


class ModernBertLayersTest(absltest.TestCase):
    def test_parallel_glu_shape(self):
        layer = modern_bert_layers.ParallelGlu(32, use_bias=False)
        x = np.ones((2, 4, 16), dtype=np.float32)
        layer.init(shapes.signature(x))
        out = layer(x)
        self.assertEqual(out.shape, (2, 4, 32))

    def test_mlp_parallel_glu_shape(self):
        layer = modern_bert_layers.ModernBertMlp(
            d_model=16,
            mlp_dim=32,
            mlp_type="parallel_glu",
            mlp_in_bias=False,
            mlp_out_bias=False,
            dropout=0.0,
            mode="eval",
        )
        x = np.ones((2, 4, 16), dtype=np.float32)
        layer.init(shapes.signature(x))
        out = layer(x)
        self.assertEqual(out.shape, (2, 4, 16))

    def test_mlp_standard_shape(self):
        layer = modern_bert_layers.ModernBertMlp(
            d_model=16,
            mlp_dim=32,
            mlp_type="mlp",
            mlp_in_bias=False,
            mlp_out_bias=False,
            dropout=0.0,
            mode="eval",
        )
        x = np.ones((2, 4, 16), dtype=np.float32)
        layer.init(shapes.signature(x))
        out = layer(x)
        self.assertEqual(out.shape, (2, 4, 16))


if __name__ == "__main__":
    absltest.main()
