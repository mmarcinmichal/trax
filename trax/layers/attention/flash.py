# coding=utf-8
"""Flash attention implementations and backends for Trax."""

import inspect

import jax
import jax.numpy as jnp

from jax import lax

from trax import fastmath
from trax.layers import base
from trax.layers import initializers as init
from trax.layers.research import rotary_positional_embedding as rotary


def _dropout(x, rate, rng, mode):
    if rate <= 0.0 or mode != "train":
        return x
    keep_prob = 1.0 - rate
    mask = fastmath.random.bernoulli(rng, keep_prob, x.shape)
    return x * mask / keep_prob


def _split_heads(x, n_heads):
    b, l, d = x.shape
    head_dim = d // n_heads
    return x.reshape(b, l, n_heads, head_dim).transpose(0, 2, 1, 3)


def _merge_heads(x):
    b, h, l, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, l, h * d)


def _scale_for_depth(depth, dtype):
    return jnp.asarray(1.0 / jnp.sqrt(depth), dtype=dtype)


def segment_ids_from_cu(cu_seqlens, length):
    if cu_seqlens is None:
        return None
    valid = cu_seqlens >= 0
    cu = jnp.where(valid, cu_seqlens, length)
    positions = jnp.arange(length)

    def _single(cu_row, valid_row):
        seg = jnp.searchsorted(cu_row, positions, side="right") - 1
        valid_count = jnp.maximum(jnp.sum(valid_row) - 1, 1)
        seg = jnp.clip(seg, 0, valid_count - 1)
        return seg.astype(jnp.int32)

    return jax.vmap(_single)(cu, valid)


def _make_attention_mask(attention_mask, max_seqlen, length, batch_size):
    if attention_mask is None:
        mask = jnp.ones((batch_size, length), dtype=bool)
    else:
        mask = attention_mask.astype(bool)
    if max_seqlen is not None:
        max_seqlen = jnp.asarray(max_seqlen)
        if max_seqlen.ndim == 0:
            max_seqlen = jnp.full((mask.shape[0],), max_seqlen)
        pos = jnp.arange(length)[None, :]
        mask = mask & (pos < max_seqlen[:, None])
    return mask


def _apply_mask(scores, mask_valid, dtype):
    neg_inf = jnp.asarray(-jnp.inf, dtype=dtype)
    return jnp.where(mask_valid, scores, neg_inf)


def _full_attention_mask(attention_mask, seg_ids, causal, length):
    mask = attention_mask.astype(bool)
    q_mask = mask[:, :, None]
    k_mask = mask[:, None, :]
    valid = q_mask & k_mask
    if seg_ids is not None:
        valid = valid & (seg_ids[:, :, None] == seg_ids[:, None, :])
    if causal:
        idx = jnp.arange(length)
        causal_mask = idx[None, :, None] >= idx[None, None, :]
        valid = valid & causal_mask
    return valid


def scaled_dot_product_attention_naive(
    q,
    k,
    v,
    *,
    attention_mask=None,
    seg_ids=None,
    causal=False,
    dropout=0.0,
    mode="train",
    rng=None,
):
    """Naive reference attention over full QK (O(L^2) memory)."""
    depth = q.shape[-1]
    scale = _scale_for_depth(depth, q.dtype)
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    if attention_mask is not None:
        full_mask = _full_attention_mask(attention_mask, seg_ids, causal, q.shape[2])
        full_mask = full_mask[:, None, :, :]
        scores = _apply_mask(scores, full_mask, scores.dtype)
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(scores.dtype)
    if dropout > 0.0 and mode == "train":
        if rng is None:
            rng = fastmath.random.get_prng(0)
        weights = _dropout(weights, dropout, rng, mode)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)


def scaled_dot_product_attention_jax(
    q,
    k,
    v,
    *,
    attention_mask=None,
    seg_ids=None,
    causal=False,
    dropout=0.0,
    mode="train",
    rng=None,
):
    """JAX backend attention using jax.nn.dot_product_attention if available."""
    if attention_mask is None:
        attention_mask = jnp.ones((q.shape[0], q.shape[2]), dtype=bool)
    mask = _full_attention_mask(attention_mask, seg_ids, causal, q.shape[2])
    mask = mask[:, None, :, :]
    mask = jnp.broadcast_to(mask, (q.shape[0], q.shape[1], q.shape[2], q.shape[2]))
    kwargs = {}
    sig = inspect.signature(jax.nn.dot_product_attention)
    if "mask" in sig.parameters:
        kwargs["mask"] = mask
    if "is_causal" in sig.parameters:
        kwargs["is_causal"] = False
    if "dropout_rate" in sig.parameters:
        kwargs["dropout_rate"] = dropout
    if "deterministic" in sig.parameters:
        kwargs["deterministic"] = mode != "train"
    if "rng" in sig.parameters:
        kwargs["rng"] = rng
    q_t = jnp.transpose(q, (0, 2, 1, 3))
    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))
    out = jax.nn.dot_product_attention(q_t, k_t, v_t, **kwargs)
    return jnp.transpose(out, (0, 2, 1, 3))


def scaled_dot_product_attention_flash(
    q,
    k,
    v,
    *,
    attention_mask=None,
    seg_ids=None,
    causal=False,
    block_q=128,
    block_k=128,
    dropout=0.0,
    mode="train",
    rng=None,
):
    """Tiled FlashAttention with online softmax (no O(L^2) materialization)."""
    b, h, length, d = q.shape
    scale = _scale_for_depth(d, q.dtype)
    pad_q = (block_q - length % block_q) % block_q
    pad_k = (block_k - length % block_k) % block_k
    pad = max(pad_q, pad_k)
    if pad:
        pad_spec = ((0, 0), (0, 0), (0, pad), (0, 0))
        q = jnp.pad(q, pad_spec)
        k = jnp.pad(k, pad_spec)
        v = jnp.pad(v, pad_spec)
    length_padded = q.shape[2]

    if attention_mask is None:
        attention_mask = jnp.ones((b, length), dtype=bool)
    else:
        attention_mask = attention_mask.astype(bool)
    if pad:
        attention_mask = jnp.pad(attention_mask, ((0, 0), (0, pad)), constant_values=False)
    if seg_ids is not None and pad:
        seg_ids = jnp.pad(seg_ids, ((0, 0), (0, pad)), constant_values=-1)

    num_q = length_padded // block_q
    num_k = length_padded // block_k

    def q_loop(q_idx, carry):
        out = carry
        q_start = q_idx * block_q
        q_block = lax.dynamic_slice(q, (0, 0, q_start, 0), (b, h, block_q, d))
        q_mask = lax.dynamic_slice(attention_mask, (0, q_start), (b, block_q))
        seg_q = None
        if seg_ids is not None:
            seg_q = lax.dynamic_slice(seg_ids, (0, q_start), (b, block_q))
        q_pos = jnp.arange(block_q) + q_start

        m = jnp.full((b, h, block_q, 1), jnp.finfo(q.dtype).min, dtype=jnp.float32)
        l = jnp.zeros((b, h, block_q, 1), dtype=jnp.float32)
        out_block = jnp.zeros((b, h, block_q, d), dtype=jnp.float32)

        def k_loop(k_idx, carry_k):
            m_k, l_k, out_k = carry_k
            k_start = k_idx * block_k
            k_block = lax.dynamic_slice(k, (0, 0, k_start, 0), (b, h, block_k, d))
            v_block = lax.dynamic_slice(v, (0, 0, k_start, 0), (b, h, block_k, d))
            k_mask = lax.dynamic_slice(attention_mask, (0, k_start), (b, block_k))
            seg_k = None
            if seg_ids is not None:
                seg_k = lax.dynamic_slice(seg_ids, (0, k_start), (b, block_k))
            k_pos = jnp.arange(block_k) + k_start

            scores = jnp.einsum("bhqd,bhkd->bhqk", q_block, k_block) * scale
            scores = scores.astype(jnp.float32)

            valid = q_mask[:, None, :, None] & k_mask[:, None, None, :]
            if seg_ids is not None:
                valid = valid & (seg_q[:, None, :, None] == seg_k[:, None, None, :])
            if causal:
                causal_mask = q_pos[None, :, None] >= k_pos[None, None, :]
                valid = valid & causal_mask[None, None, :, :]

            scores = _apply_mask(scores, valid, scores.dtype)

            max_scores = jnp.max(scores, axis=-1, keepdims=True)
            m_new = jnp.maximum(m_k, max_scores)
            is_finite = jnp.isfinite(m_new)
            exp_m = jnp.where(is_finite, jnp.exp(m_k - m_new), 0.0)
            exp_scores = jnp.where(is_finite, jnp.exp(scores - m_new), 0.0)
            if dropout > 0.0 and mode == "train":
                if rng is None:
                    local_rng = fastmath.random.get_prng(0)
                else:
                    local_rng = jax.random.fold_in(rng, q_idx * num_k + k_idx)
                keep_prob = 1.0 - dropout
                mask = fastmath.random.bernoulli(local_rng, keep_prob, exp_scores.shape)
                exp_scores = exp_scores * mask / keep_prob
            l_new = l_k * exp_m + jnp.sum(exp_scores, axis=-1, keepdims=True)
            out_new = out_k * exp_m + jnp.einsum("bhqk,bhkd->bhqd", exp_scores, v_block)
            return m_new, l_new, out_new

        m_final, l_final, out_final = lax.fori_loop(0, num_k, k_loop, (m, l, out_block))
        denom = jnp.where(l_final == 0, 1.0, l_final)
        out_final = out_final / denom
        out_final = out_final.astype(q.dtype)
        out = lax.dynamic_update_slice(out, out_final, (0, 0, q_start, 0))
        return out

    output = jnp.zeros((b, h, length_padded, d), dtype=q.dtype)
    output = lax.fori_loop(0, num_q, q_loop, output)
    output = output[:, :, :length, :]
    return output


def scaled_dot_product_attention(
    q,
    k,
    v,
    *,
    implementation="flash",
    attention_mask=None,
    seg_ids=None,
    causal=False,
    dropout=0.0,
    mode="train",
    rng=None,
    block_q=128,
    block_k=128,
):
    if implementation == "jax":
        return scaled_dot_product_attention_jax(
            q,
            k,
            v,
            attention_mask=attention_mask,
            seg_ids=seg_ids,
            causal=causal,
            dropout=dropout,
            mode=mode,
            rng=rng,
        )
    if implementation == "naive":
        return scaled_dot_product_attention_naive(
            q,
            k,
            v,
            attention_mask=attention_mask,
            seg_ids=seg_ids,
            causal=causal,
            dropout=dropout,
            mode=mode,
            rng=rng,
        )
    if implementation == "flash":
        return scaled_dot_product_attention_flash(
            q,
            k,
            v,
            attention_mask=attention_mask,
            seg_ids=seg_ids,
            causal=causal,
            block_q=block_q,
            block_k=block_k,
            dropout=dropout,
            mode=mode,
            rng=rng,
        )
    raise ValueError("implementation must be one of: 'flash', 'jax', 'naive'.")


class FlashSelfAttention(base.Layer):
    """Self-attention with selectable implementation."""

    def __init__(
        self,
        d_model,
        n_heads,
        block_size,
        *,
        attention_dropout=0.0,
        use_rope=True,
        rotary_dim=None,
        rotary_base=10000.0,
        use_segment_mask=True,
        attn_qkv_bias=False,
        attn_out_bias=False,
        implementation="flash",
        causal=False,
        mode="train",
    ):
        super().__init__(n_in=4, n_out=1)
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self._d_model = d_model
        self._n_heads = n_heads
        self._block_size = block_size
        self._attention_dropout = attention_dropout
        self._use_rope = use_rope
        self._rotary_dim = rotary_dim
        self._rotary_base = rotary_base
        self._use_segment_mask = use_segment_mask
        self._attn_qkv_bias = attn_qkv_bias
        self._attn_out_bias = attn_out_bias
        self._implementation = implementation
        self._causal = causal
        self._mode = mode

    def init_weights_and_state(self, input_signature):
        x_sig = input_signature[0]
        d_model = x_sig.shape[-1]
        w_init = init.RandomNormalInitializer(0.02)
        rng_q, rng_k, rng_v, rng_o = fastmath.random.split(self.rng, 4)
        wq = w_init((d_model, d_model), rng_q)
        wk = w_init((d_model, d_model), rng_k)
        wv = w_init((d_model, d_model), rng_v)
        wo = w_init((d_model, d_model), rng_o)
        if self._attn_qkv_bias:
            bq = jnp.zeros((d_model,), dtype=x_sig.dtype)
            bk = jnp.zeros((d_model,), dtype=x_sig.dtype)
            bv = jnp.zeros((d_model,), dtype=x_sig.dtype)
        else:
            bq = bk = bv = None
        bo = jnp.zeros((d_model,), dtype=x_sig.dtype) if self._attn_out_bias else None
        self.weights = (wq, wk, wv, wo, bq, bk, bv, bo)
        self.state = ()

    def forward(self, inputs):
        x, attention_mask, cu_seqlens, max_seqlen = inputs
        wq, wk, wv, wo, bq, bk, bv, bo = self.weights
        q = jnp.dot(x, wq) + (bq if bq is not None else 0)
        k = jnp.dot(x, wk) + (bk if bk is not None else 0)
        v = jnp.dot(x, wv) + (bv if bv is not None else 0)
        q = _split_heads(q, self._n_heads)
        k = _split_heads(k, self._n_heads)
        v = _split_heads(v, self._n_heads)
        if self._use_rope:
            q, k = rotary.apply_rotary_embedding(
                q, k, rotary_dim=self._rotary_dim, base=self._rotary_base
            )
        if attention_mask is None and max_seqlen is not None:
            attention_mask = _make_attention_mask(
                attention_mask, max_seqlen, q.shape[2], q.shape[0]
            )
        if attention_mask is None:
            attention_mask = jnp.ones((q.shape[0], q.shape[2]), dtype=bool)
        seg_ids = None
        if cu_seqlens is not None and self._use_segment_mask:
            seg_ids = segment_ids_from_cu(cu_seqlens, q.shape[2])
        attn_out = scaled_dot_product_attention(
            q,
            k,
            v,
            implementation=self._implementation,
            attention_mask=attention_mask,
            seg_ids=seg_ids,
            causal=self._causal,
            dropout=self._attention_dropout,
            mode=self._mode,
            rng=self.rng,
            block_q=self._block_size,
            block_k=self._block_size,
        )
        attn_out = _merge_heads(attn_out)
        attn_out = jnp.dot(attn_out, wo) + (bo if bo is not None else 0)
        return attn_out


def make_self_attention(**kwargs):
    """Factory that returns a FlashSelfAttention instance."""
    return FlashSelfAttention(**kwargs)
