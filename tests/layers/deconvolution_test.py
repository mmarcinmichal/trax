# coding=utf-8
# Copyright 2022 The Trax Authors.
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

"""Tests for Deconvolution layers."""

import numpy as np

from absl.testing import absltest

import trax.layers as tl

from trax.utils import shapes


class ConvTransposeTest(absltest.TestCase):
    def test_call(self):
        layer = tl.ConvTranspose(30, (3, 3))
        x = np.ones((9, 5, 5, 20))
        layer.init(shapes.signature(x))

        y = layer(x)
        self.assertEqual(y.shape, (9, 7, 7, 30))


if __name__ == "__main__":
    absltest.main()
