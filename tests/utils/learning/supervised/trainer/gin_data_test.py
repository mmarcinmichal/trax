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

"""Tests for gin data configs using make_inputs."""

from pathlib import Path

from absl.testing import absltest

_GIN_DATA_WITH_BATCHER = [
    "hourglass/hourglass_cifar10.gin",
    "hourglass/hourglass_enwik8.gin",
    "hourglass/hourglass_imagenet32.gin",
    "hourglass/hourglass_imagenet64.gin",
    "lstm/lstm_lm1b.gin",
    "mlp/mlp_mnist.gin",
    "reformer/reformer_bair_robot_pushing.gin",
    "reformer/reformer_cifar10.gin",
    "reformer/reformer_enwik8.gin",
    "reformer/reformer_imagenet64.gin",
    "reformer/reformer_imagenet64_testing.gin",
    "reformer/reformer_pc_enpl.gin",
    "resnet/resnet50_frn_imagenet_8gb.gin",
    "resnet/resnet50_imagenet_8gb_testing.gin",
    "resnet/wide_resnet_cifar10_8gb.gin",
    "scientific_papers/scientific_papers_terraformer.gin",
    "scientific_papers/scientific_papers_terraformer_favor.gin",
    "scientific_papers/scientific_papers_terraformer_pretrained.gin",
    "sparse/sparse_lm1b_pretrain_16gb.gin",
    "transformer/transformer_big_lm1b_8gb.gin",
    "transformer/transformer_finetune_squad_16gb.gin",
    "transformer/transformer_imdb_tfds.gin",
    "transformer/transformer_lm1b_8gb_testing.gin",
    "transformer/transformer_lm1b_cond_skipping.gin",
    "transformer/transformer_lm1b_layerdrop.gin",
    "transformer/transformer_lm1b_layerdrop_every.gin",
    "transformer/transformer_lm1b_layerdrop_ushape.gin",
    "transformer/transformer_lm1b_skipping.gin",
    "transformer/transformer_ptb_16gb.gin",
]


class GinDataTest(absltest.TestCase):
    def test_gin_configs_no_batcher(self):
        repo_root = Path(__file__).resolve().parents[5]
        gin_root = (
            repo_root
            / "resources"
            / "learning"
            / "supervised"
            / "configs"
            / "gini"
        )
        for rel_path in _GIN_DATA_WITH_BATCHER:
            path = gin_root / rel_path
            with self.subTest(config=rel_path):
                text = path.read_text()
                self.assertNotIn("batcher.", text)
                self.assertNotIn("make_inputs_from_data_streams.", text)
                self.assertTrue(
                    "train.inputs = @trax.data.make_inputs" in text
                    or "train.inputs = @data.make_inputs" in text
                )


if __name__ == "__main__":
    absltest.main()
