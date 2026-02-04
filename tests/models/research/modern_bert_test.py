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

"""Tests for ModernBERT model."""

import numpy as np

from absl.testing import absltest

from trax.models.research import modern_bert
from trax.utils import shapes


class ModernBertModelTest(absltest.TestCase):
    def _make_inputs(self, batch_size=2, seq_len=8):
        input_ids = np.arange(batch_size * seq_len, dtype=np.int32).reshape(
            batch_size, seq_len
        )
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int32)
        position_ids = np.tile(np.arange(seq_len, dtype=np.int32), (batch_size, 1))
        cu_seqlens = np.asarray([[0, seq_len], [0, seq_len]], dtype=np.int32)
        max_seqlen = np.asarray([seq_len, seq_len], dtype=np.int32)
        return input_ids, attention_mask, position_ids, cu_seqlens, max_seqlen

    def test_encoder_forward_shape(self):
        config = modern_bert.ModernBertConfig(
            vocab_size=128,
            max_len=8,
            num_layers=2,
            d_model=32,
            num_heads=4,
            mlp_dim=64,
            use_rope=True,
            use_flash_attention=True,
            flash_block_size=4,
            embed_dropout=0.0,
            dropout=0.0,
            attention_dropout=0.0,
            parallel_block=True,
        )
        model = modern_bert.create_modern_bert(config, mode="eval")
        inputs = self._make_inputs()
        input_sig = tuple(shapes.signature(x) for x in inputs)
        model.init(input_sig)
        out = model(inputs)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_mlm_head_shape(self):
        config = modern_bert.ModernBertConfig(
            vocab_size=64,
            max_len=8,
            num_layers=1,
            d_model=16,
            num_heads=4,
            mlp_dim=32,
            use_rope=True,
            use_flash_attention=True,
            flash_block_size=4,
            embed_dropout=0.0,
            dropout=0.0,
            attention_dropout=0.0,
            parallel_block=True,
        )
        model = modern_bert.create_modern_bert_mlm(config, mode="eval")
        inputs = self._make_inputs()
        input_sig = tuple(shapes.signature(x) for x in inputs)
        model.init(input_sig)
        logits = model(inputs)
        self.assertEqual(logits.shape, (2, 8, 64))

    def test_forward_with_gating_enabled(self):
        config = modern_bert.ModernBertConfig(
            vocab_size=32,
            max_len=8,
            num_layers=1,
            d_model=16,
            num_heads=4,
            mlp_dim=32,
            use_rope=True,
            use_flash_attention=True,
            flash_block_size=4,
            embed_dropout=0.0,
            dropout=0.0,
            attention_dropout=0.0,
            use_gating=True,
            gating_position="attention",
            parallel_block=True,
        )
        model = modern_bert.create_modern_bert(config, mode="eval")
        inputs = self._make_inputs()
        input_sig = tuple(shapes.signature(x) for x in inputs)
        model.init(input_sig)
        out = model(inputs)
        self.assertEqual(out.shape, (2, 8, 16))

    def test_config_string_coercion(self):
        config = modern_bert.ModernBertConfig(
            mlp_type="parallel_glu",
            normalization="rmsnorm",
            gating_position="both",
        )
        self.assertEqual(config.mlp_type.value, "parallel_glu")
        self.assertEqual(config.normalization.value, "rmsnorm")
        self.assertEqual(config.gating_position.value, "both")

    def test_segment_mask_matches_slow_path(self):
        config = modern_bert.ModernBertConfig(
            vocab_size=32,
            max_len=8,
            num_layers=1,
            d_model=16,
            num_heads=4,
            mlp_dim=32,
            use_rope=True,
            use_flash_attention=True,
            use_segment_mask=False,
            flash_block_size=4,
            embed_dropout=0.0,
            dropout=0.0,
            attention_dropout=0.0,
            parallel_block=True,
        )
        block_slow = modern_bert.ModernBertParallelBlock(config, mode="eval")
        inputs = self._make_inputs()
        input_ids, attention_mask, _, cu_seqlens, max_seqlen = inputs
        x = np.tile(input_ids[:, :, None], (1, 1, 16)).astype(np.float32)
        sig = (
            shapes.signature(x),
            shapes.signature(attention_mask),
            shapes.signature(cu_seqlens),
            shapes.signature(max_seqlen),
        )
        block_slow.init(sig)
        slow_out, *_ = block_slow((x, attention_mask, cu_seqlens, max_seqlen))

        config_fast = modern_bert.ModernBertConfig(
            vocab_size=32,
            max_len=8,
            num_layers=1,
            d_model=16,
            num_heads=4,
            mlp_dim=32,
            use_rope=True,
            use_flash_attention=True,
            use_segment_mask=True,
            flash_block_size=4,
            embed_dropout=0.0,
            dropout=0.0,
            attention_dropout=0.0,
            parallel_block=True,
        )
        block_fast = modern_bert.ModernBertParallelBlock(config_fast, mode="eval")
        block_fast.weights = block_slow.weights
        fast_out, *_ = block_fast((x, attention_mask, cu_seqlens, max_seqlen))
        np.testing.assert_allclose(fast_out, slow_out, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    absltest.main()
