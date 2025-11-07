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

"""Tests for Graph Neural Network models."""

import numpy as np

from absl.testing import absltest

from trax.models import gnn
from trax.utils import shapes


class GNNTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        base_adj = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.float32,
        )
        self.adj = np.stack([base_adj, base_adj])
        self.features = np.ones((2, 4, 3), dtype=np.float32)
        self.edge_features = np.ones((2, 4, 4, 1), dtype=np.float32)

    def test_graph_conv_net_forward_shape(self):
        model = gnn.GraphConvNet(hidden_sizes=(5, 2))
        _, _ = model.init([shapes.signature(self.features), shapes.signature(self.adj)])
        out_features, out_adj = model([self.features, self.adj])
        self.assertEqual(out_features.shape, (2, 4, 2))
        self.assertEqual(out_adj.shape, (2, 4, 4))

    def test_graph_attention_net_forward_shape(self):
        model = gnn.GraphAttentionNet(hidden_sizes=(5, 2), num_heads=2)
        _, _ = model.init([shapes.signature(self.features), shapes.signature(self.adj)])
        out_features, out_adj = model([self.features, self.adj])
        self.assertEqual(out_features.shape, (2, 4, 2))
        self.assertEqual(out_adj.shape, (2, 4, 4))

    def test_graph_edge_net_forward_shape(self):
        model = gnn.GraphEdgeNet(node_sizes=(5, 2), edge_sizes=(3, 2))
        model.init(
            [
                shapes.signature(self.features),
                shapes.signature(self.edge_features),
                shapes.signature(self.adj),
            ]
        )
        out_features, out_edges, out_adj = model(
            [self.features, self.edge_features, self.adj]
        )
        self.assertEqual(out_features.shape, (2, 4, 2))
        self.assertEqual(out_edges.shape, (2, 4, 4, 2))
        self.assertEqual(out_adj.shape, (2, 4, 4))


if __name__ == "__main__":
    absltest.main()
