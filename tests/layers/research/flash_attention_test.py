# coding=utf-8
"""Tests for flash_attention."""

import numpy as np
from absl.testing import absltest

import jax
from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers.research import flash_attention


def _naive_attention(q, k, v, mask=None):
    logits = jnp.einsum("bqd,bkd->bqk", q, k)
    if mask is not None:
        logits = jnp.where(mask[:, None, :], -1e9, logits)
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bqk,bkd->bqd", weights, v)


class FlashAttentionTest(absltest.TestCase):
    def test_matches_naive(self):
        with fastmath.use_backend(fastmath.Backend.JAX):
            batch, seqlen, d = 2, 7, 4
            q = jnp.arange(batch * seqlen * d).reshape((batch, seqlen, d)) / 100.0
            k = jnp.arange(batch * seqlen * d).reshape((batch, seqlen, d)) / 50.0
            v = jnp.arange(batch * seqlen * d).reshape((batch, seqlen, d)) / 25.0
            mask = jnp.arange(seqlen)[None, :] >= 5
            out_ref = _naive_attention(q, k, v, mask)
            out_flash = flash_attention.flash_attention(
                q, k, v, block_size=4, mask=mask
            )
            self.assertEqual(out_ref.shape, out_flash.shape)
            np.testing.assert_allclose(out_ref, out_flash, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    absltest.main()
