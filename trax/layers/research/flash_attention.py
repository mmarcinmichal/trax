# coding=utf-8
"""Flash attention implementation for Trax.

This module provides a ``FlashSelfAttention`` layer that computes
self-attention using the FlashAttention algorithm. It streams the
computation in tiled blocks so the full attention matrix is never
materialised in memory. The layer can be used as a drop in replacement
for :class:`trax.layers.research.efficient_attention.SelfAttention` when
causal attention without masking is required.

Flash attention is most beneficial on long sequences where regular
self-attention becomes memory bound. For short sequences the standard
implementation may be faster.
"""

import math

import jax

from jax import lax

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers.research.efficient_attention import apply_broadcasted_dropout


class FlashSelfAttention(base.Layer):
    """Memory efficient self-attention using the FlashAttention algorithm."""

    def __init__(
        self,
        n_heads: int = 2,
        d_qk: int = 64,
        d_v: int = 64,
        causal: bool = False,
        masked: bool = False,
        bias: bool = False,
        block_size: int = 128,
        attention_dropout: float = 0.0,
        output_dropout: float = 0.0,
        mode: str = "train",
    ) -> None:
        super().__init__(n_in=(2 if masked else 1), n_out=1)
        self._n_heads = n_heads
        self._d_qk = d_qk
        self._d_v = d_v
        self._causal = causal
        self._masked = masked
        self._bias = bias
        self._block_size = block_size
        self._attention_dropout = attention_dropout
        self._output_dropout = output_dropout
        self._mode = mode

    def init_weights_and_state(self, input_signature):
        d_model = input_signature.shape[-1]
        rng = fastmath.random.get_prng(0)
        w_q = jax.random.normal(rng, (d_model, self._n_heads * self._d_qk)) / math.sqrt(d_model)
        w_k = jax.random.normal(rng, (d_model, self._n_heads * self._d_qk)) / math.sqrt(d_model)
        w_v = jax.random.normal(rng, (d_model, self._n_heads * self._d_v)) / math.sqrt(d_model)
        w_o = jax.random.normal(rng, (self._n_heads * self._d_v, d_model)) / math.sqrt(self._n_heads * self._d_v)
        if self._bias:
            b_q = jnp.zeros((self._n_heads * self._d_qk,))
            b_k = jnp.zeros((self._n_heads * self._d_qk,))
            b_v = jnp.zeros((self._n_heads * self._d_v,))
            b_o = jnp.zeros((d_model,))
            self.weights = (w_q, w_k, w_v, w_o, b_q, b_k, b_v, b_o)
        else:
            self.weights = (w_q, w_k, w_v, w_o)
        self.state = ()

    # ---------------------------------------------------------------------
    def forward(self, inputs):
        if self._masked:
            x, mask = inputs
            del mask  # Masking not implemented in this version.
        else:
            x = inputs

        if self._bias:
            w_q, w_k, w_v, w_o, b_q, b_k, b_v, b_o = self.weights
        else:
            w_q, w_k, w_v, w_o = self.weights
        q = jnp.dot(x, w_q)
        k = jnp.dot(x, w_k)
        v = jnp.dot(x, w_v)
        if self._bias:
            q += b_q
            k += b_k
            v += b_v

        batch, seqlen, _ = q.shape
        q = q.reshape(batch, seqlen, self._n_heads, self._d_qk).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seqlen, self._n_heads, self._d_qk).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seqlen, self._n_heads, self._d_v).transpose(0, 2, 1, 3)

        out = _flash_attention(
            q, k, v, causal=self._causal, block_size=self._block_size, dropout=self._attention_dropout, mode=self._mode
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, seqlen, self._n_heads * self._d_v)
        out = jnp.dot(out, w_o)
        if self._bias:
            out += b_o
        out = apply_broadcasted_dropout(out, self._output_dropout, self.rng if self._mode == "train" else None)
        return out


# -----------------------------------------------------------------------------

def _flash_attention(q, k, v, *, causal: bool, block_size: int, dropout: float, mode: str):
    """FlashAttention reference implementation.

    Args:
      q: Query tensor ``[batch, heads, length, depth]``.
      k: Key tensor ``[batch, heads, length, depth]``.
      v: Value tensor ``[batch, heads, length, depth_v]``.
      causal: Whether to apply a causal mask.
      block_size: Block size used for tiling the sequence dimension.
      dropout: Attention dropout rate.
      mode: ``"train"`` or ``"eval"``/``"predict"``.

    Returns:
      Output tensor of shape ``[batch, heads, length, depth_v]``.
    """
    batch, n_heads, seqlen, d = q.shape
    d_v = v.shape[-1]
    scale = 1.0 / math.sqrt(d)

    def process_query_block(qs):
        q_start = qs * block_size
        q_end = jnp.minimum(q_start + block_size, seqlen)
        q_block = lax.dynamic_slice(q, (0, 0, q_start, 0), (batch, n_heads, q_end - q_start, d))
        m = jnp.full((batch, n_heads, q_end - q_start, 1), -jnp.inf)
        row_sum = jnp.zeros((batch, n_heads, q_end - q_start, 1))
        o = jnp.zeros((batch, n_heads, q_end - q_start, d_v))
        def body(ks, carry):
            m, row_sum, o = carry
            k_start = ks * block_size
            k_end = jnp.minimum(k_start + block_size, seqlen)
            k_block = lax.dynamic_slice(k, (0, 0, k_start, 0), (batch, n_heads, k_end - k_start, d))
            v_block = lax.dynamic_slice(v, (0, 0, k_start, 0), (batch, n_heads, k_end - k_start, d_v))
            dots = lax.dot_general(
                q_block,
                k_block,
                (((3,), (3,)), ((0, 1), (0, 1))),
            )
            dots = dots * scale
            if causal:
                q_idx = jnp.arange(q_start, q_end).reshape(1, 1, -1, 1)
                k_idx = jnp.arange(k_start, k_end).reshape(1, 1, 1, -1)
                dots = jnp.where(k_idx > q_idx, -jnp.inf, dots)
            block_max = jnp.max(dots, axis=-1, keepdims=True)
            m_new = jnp.maximum(m, block_max)
            exp_m = jnp.exp(m - m_new)
            exp_d = jnp.exp(dots - m_new)
            row_sum_new = exp_m * row_sum + jnp.sum(exp_d, axis=-1, keepdims=True)
            o = o * (exp_m * row_sum / row_sum_new) + lax.dot_general(
                exp_d,
                v_block,
                (((3,), (2,)), ((0, 1, 2), (0, 1, 2))),
            ) / row_sum_new
            if dropout > 0.0 and mode == "train":
                keep = fastmath.random.bernoulli(fastmath.random.get_prng(0), 1.0 - dropout, o.shape)
                o = jnp.where(keep, o / (1.0 - dropout), 0)
            return (m_new, row_sum_new, o)
        (m, row_sum, o) = jax.lax.fori_loop(0, (seqlen + block_size - 1) // block_size, body, (m, row_sum, o))
        return o

    outputs = []
    for i in range((seqlen + block_size - 1) // block_size):
        outputs.append(process_query_block(i))
    return jnp.concatenate(outputs, axis=2)
