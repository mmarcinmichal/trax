# coding=utf-8
"""Tests for :mod:`trax.layers.research.flash_attention`."""

import jax

from tensorflow import test

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers.research import efficient_attention, flash_attention
from trax.utils import shapes


class FlashAttentionTest(test.TestCase):
    def test_matches_reference_self_attention(self):
        with fastmath.use_backend(fastmath.Backend.JAX):
            layer_flash = flash_attention.FlashSelfAttention(
                n_heads=2,
                d_qk=4,
                d_v=4,
                causal=True,
                attention_dropout=0.0,
                output_dropout=0.0,
                block_size=4,
                mode="train",
            )
            layer_ref = efficient_attention.SelfAttention(
                n_heads=2,
                d_qk=4,
                d_v=4,
                causal=True,
                chunk_len=None,
                n_chunks_before=0,
                n_chunks_after=0,
                attention_dropout=0.0,
                output_dropout=0.0,
                use_reference_code=True,
                mode="train",
            )
            x = jax.random.uniform(jax.random.PRNGKey(0), (2, 8, 8), dtype=jnp.float32)
            sig = shapes.signature(x)
            layer_flash.init(sig)
            layer_ref.init(sig)
            y_flash = layer_flash(x)
            y_ref = layer_ref(x)
            self.assertAllClose(y_flash, y_ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test.main()
