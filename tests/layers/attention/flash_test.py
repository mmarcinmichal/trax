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

"""Tests for flash attention layers."""

import os
import time

import jax
import numpy as np

from absl.testing import absltest

from trax.layers.attention import flash as flash_attention
from trax.layers.attention.base import DotProductAttention
from trax.utils import shapes


class FlashAttentionLayerTest(absltest.TestCase):
    def test_attention_implementations_match_reference(self):
        rng = np.random.default_rng(0)
        b, h, l, d = 2, 2, 8, 4
        q = rng.normal(size=(b, h, l, d)).astype(np.float32)
        k = rng.normal(size=(b, h, l, d)).astype(np.float32)
        v = rng.normal(size=(b, h, l, d)).astype(np.float32)
        mask = np.ones((b, l), dtype=np.int32)

        ref_layer = DotProductAttention(dropout=0.0, mode="eval")
        ref_layer.init((
            shapes.signature(q),
            shapes.signature(k),
            shapes.signature(v),
            shapes.signature(mask[:, None, None, :]),
        ))
        ref = ref_layer((q, k, v, mask[:, None, None, :]))

        for impl in ("naive", "jax", "flash"):
            out = flash_attention.scaled_dot_product_attention(
                q,
                k,
                v,
                implementation=impl,
                attention_mask=None,
                causal=False,
                dropout=0.0,
                mode="eval",
                block_q=4,
                block_k=4,
            )
            np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)

    def test_packed_sequences_with_padding(self):
        rng = np.random.default_rng(0)
        b, h, l, d = 2, 2, 8, 4
        q = rng.normal(size=(b, h, l, d)).astype(np.float32)
        k = rng.normal(size=(b, h, l, d)).astype(np.float32)
        v = rng.normal(size=(b, h, l, d)).astype(np.float32)
        attention_mask = np.ones((b, l), dtype=np.int32)
        cu_seqlens = np.asarray([[0, 4, 8], [0, 3, 8]], dtype=np.int32)
        seg_ids = flash_attention.segment_ids_from_cu(cu_seqlens, l)

        ref = flash_attention.scaled_dot_product_attention(
            q,
            k,
            v,
            implementation="naive",
            attention_mask=attention_mask,
            seg_ids=seg_ids,
            causal=False,
            dropout=0.0,
            mode="eval",
            block_q=4,
            block_k=4,
        )
        for impl in ("jax", "flash"):
            out = flash_attention.scaled_dot_product_attention(
                q,
                k,
                v,
                implementation=impl,
                attention_mask=attention_mask,
                seg_ids=seg_ids,
                causal=False,
                dropout=0.0,
                mode="eval",
                block_q=4,
                block_k=4,
            )
            np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-4)

    def test_flash_self_attention_shape(self):
        layer = flash_attention.FlashSelfAttention(
            d_model=16,
            n_heads=4,
            block_size=4,
            attention_dropout=0.0,
            use_rope=True,
            use_segment_mask=True,
            mode="eval",
        )
        x = np.ones((2, 8, 16), dtype=np.float32)
        attention_mask = np.ones((2, 8), dtype=np.int32)
        cu_seqlens = np.asarray([[0, 8], [0, 8]], dtype=np.int32)
        max_seqlen = np.asarray([8, 8], dtype=np.int32)
        sig = (
            shapes.signature(x),
            shapes.signature(attention_mask),
            shapes.signature(cu_seqlens),
            shapes.signature(max_seqlen),
        )
        layer.init(sig)
        out = layer((x, attention_mask, cu_seqlens, max_seqlen))
        self.assertEqual(out.shape, (2, 8, 16))

    def test_segment_mask_matches_slow_path(self):
        slow = flash_attention.FlashSelfAttention(
            d_model=16,
            n_heads=4,
            block_size=4,
            attention_dropout=0.0,
            use_rope=True,
            use_segment_mask=True,
            implementation="naive",
            mode="eval",
        )
        fast = flash_attention.FlashSelfAttention(
            d_model=16,
            n_heads=4,
            block_size=4,
            attention_dropout=0.0,
            use_rope=True,
            use_segment_mask=True,
            implementation="flash",
            mode="eval",
        )
        x = np.arange(2 * 8 * 16, dtype=np.float32).reshape(2, 8, 16)
        attention_mask = np.ones((2, 8), dtype=np.int32)
        cu_seqlens = np.asarray([[0, 4, 8], [0, 3, 8]], dtype=np.int32)
        max_seqlen = np.asarray([8, 8], dtype=np.int32)
        sig = (
            shapes.signature(x),
            shapes.signature(attention_mask),
            shapes.signature(cu_seqlens),
            shapes.signature(max_seqlen),
        )
        slow.init(sig)
        fast.weights = slow.weights
        slow_out = slow((x, attention_mask, cu_seqlens, max_seqlen))
        fast_out = fast((x, attention_mask, cu_seqlens, max_seqlen))
        np.testing.assert_allclose(fast_out, slow_out, atol=5e-3, rtol=5e-4)

    def test_attention_impls_benchmark(self):
        if not os.getenv("TRAX_BENCHMARKS"):
            self.skipTest("Set TRAX_BENCHMARKS=1 to enable benchmark.")
        rng = np.random.default_rng(0)
        b, h, l, d = 2, 4, 64, 32
        q = rng.normal(size=(b, h, l, d)).astype(np.float32)
        k = rng.normal(size=(b, h, l, d)).astype(np.float32)
        v = rng.normal(size=(b, h, l, d)).astype(np.float32)
        mask = np.ones((b, l), dtype=np.int32)

        def _time_impl(impl):
            # Warmup
            out = flash_attention.scaled_dot_product_attention(
                q,
                k,
                v,
                implementation=impl,
                attention_mask=mask,
                causal=False,
                dropout=0.0,
                mode="eval",
                block_q=16,
                block_k=16,
            )
            jax.device_get(out)
            start = time.perf_counter()
            out = flash_attention.scaled_dot_product_attention(
                q,
                k,
                v,
                implementation=impl,
                attention_mask=mask,
                causal=False,
                dropout=0.0,
                mode="eval",
                block_q=16,
                block_k=16,
            )
            jax.device_get(out)
            return time.perf_counter() - start

        times = {impl: _time_impl(impl) for impl in ("naive", "jax", "flash")}
        self.assertTrue(all(t > 0 for t in times.values()))


if __name__ == "__main__":
    absltest.main()
