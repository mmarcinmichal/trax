# coding=utf-8
"""Flash attention implementation for Trax."""

import jax
import jax.numpy as jnp

from jax import lax


def flash_attention(q, k, v, *, block_size, mask=None):
    """Memory efficient dot-product attention.

    Args:
      q: Queries array of shape [batch, len, depth].
      k: Keys array of shape [batch, len, depth].
      v: Values array of shape [batch, len, depth_v].
      block_size: Integer block size used for computation.
      mask: Optional boolean mask of shape [batch, len] where ``True`` values
        indicate positions that should be masked out.

    Returns:
      Array of shape [batch, len, depth_v] with the attention outputs.
    """
    seqlen = q.shape[1]
    pad_len = (block_size - seqlen % block_size) % block_size
    if pad_len:
        pad = ((0, 0), (0, pad_len), (0, 0))
        q = jnp.pad(q, pad)
        k = jnp.pad(k, pad)
        v = jnp.pad(v, pad)
        if mask is not None:
            mask = jnp.pad(mask, ((0, 0), (0, pad_len)), constant_values=True)
    total_len = q.shape[1]

    if mask is not None:
        if mask.ndim == 2:
            mask_b = mask[:, None, :]
        elif mask.ndim == 3:
            mask_b = mask
        else:
            raise ValueError("mask must be 2D or 3D.")
    else:
        mask_b = None

    outputs = []
    for start in range(0, total_len, block_size):
        q_block = lax.dynamic_slice(
            q, (0, start, 0), (q.shape[0], block_size, q.shape[2])
        )
        logits = jnp.einsum("bqd,bkd->bqk", q_block, k)
        if mask_b is not None:
            if mask_b.ndim == 3:
                block_mask = mask_b[:, start : start + block_size, :]
                logits = jnp.where(block_mask, -1e9, logits)
            else:
                logits = jnp.where(mask_b, -1e9, logits)
        weights = jax.nn.softmax(logits, axis=-1)
        out_block = jnp.einsum("bqk,bkd->bqd", weights, v)
        outputs.append(out_block)
    output = jnp.concatenate(outputs, axis=1)
    if pad_len:
        output = output[:, :seqlen, :]
    return output
