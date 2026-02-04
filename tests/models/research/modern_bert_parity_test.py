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

"""Parity tests for ModernBERT model shapes."""

import numpy as np

from absl.testing import absltest

from trax.models.research import modern_bert
from trax.utils import shapes


class ModernBertParityTest(absltest.TestCase):
    def _make_inputs(self, batch_size=2, seq_len=8):
        input_ids = np.arange(batch_size * seq_len, dtype=np.int32).reshape(batch_size, seq_len)
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int32)
        position_ids = np.tile(np.arange(seq_len, dtype=np.int32), (batch_size, 1))
        cu_seqlens = np.asarray([[0, seq_len], [0, seq_len]], dtype=np.int32)
        max_seqlen = np.asarray([seq_len, seq_len], dtype=np.int32)
        return input_ids, attention_mask, position_ids, cu_seqlens, max_seqlen

    def test_model_shape_parity_with_modernbert(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch not available.")

        module = None
        try:
            import importlib.util
            import os

            path = "/mnt/d/Projects/trax/resources/ModernBERT-main/src/bert_layers/model.py"
            if os.path.exists(path):
                spec = importlib.util.spec_from_file_location("modernbert_model", path)
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
        except Exception:
            module = None

        if module is None:
            self.skipTest("ModernBERT model module not available.")

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
            parallel_block=True,
        )
        trax_model = modern_bert.create_modern_bert(config, mode="eval")
        inputs = self._make_inputs()
        input_sig = tuple(shapes.signature(x) for x in inputs)
        trax_model.init(input_sig)
        trax_out = trax_model(inputs)

        try:
            mb_config = module.FlexBertConfig(
                vocab_size=32,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=32,
                attention_layer="parallel",
                bert_layer="parallel_prenorm",
                mlp_layer="parallel_glu",
                normalization="rmsnorm",
                padding="unpadded",
                embed_dropout_prob=0.0,
                mlp_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                attn_out_dropout_prob=0.0,
                attn_qkv_bias=False,
                attn_out_bias=False,
            )
            mb_model = module.FlexBertModel(mb_config)
            input_ids = torch.zeros((2, 8), dtype=torch.long)
            attention_mask = torch.ones((2, 8), dtype=torch.long)
            cu_seqlens = torch.tensor([[0, 8], [0, 8]], dtype=torch.int32)
            max_seqlen = torch.tensor([8, 8], dtype=torch.int32)
            mb_out = mb_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        except Exception:
            self.skipTest("ModernBERT model could not be instantiated.")

        self.assertEqual(tuple(trax_out.shape), tuple(mb_out.shape))

    def test_fast_and_slow_attention_api_parity(self):
        config_slow = modern_bert.ModernBertConfig(
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
        block_slow = modern_bert.ModernBertParallelBlock(config_slow, mode="eval")
        block_fast = modern_bert.ModernBertParallelBlock(config_fast, mode="eval")

        input_ids = np.arange(16, dtype=np.int32).reshape(2, 8)
        attention_mask = np.ones((2, 8), dtype=np.int32)
        cu_seqlens = np.asarray([[0, 4, 8], [0, 3, 8]], dtype=np.int32)
        max_seqlen = np.asarray([8, 8], dtype=np.int32)
        x = np.tile(input_ids[:, :, None], (1, 1, 16)).astype(np.float32)

        sig = (
            shapes.signature(x),
            shapes.signature(attention_mask),
            shapes.signature(cu_seqlens),
            shapes.signature(max_seqlen),
        )
        block_slow.init(sig)
        block_fast.weights = block_slow.weights
        slow_out, *_ = block_slow((x, attention_mask, cu_seqlens, max_seqlen))
        fast_out, *_ = block_fast((x, attention_mask, cu_seqlens, max_seqlen))

        np.testing.assert_allclose(fast_out, slow_out, atol=5e-3, rtol=5e-4)


if __name__ == "__main__":
    absltest.main()
