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

"""Modern BERT-style encoder stack for Trax."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import jax
import jax.numpy as jnp

from trax import fastmath
from trax.layers import base
from trax.layers import initializers as init
from trax.layers.research import rotary_positional_embedding as rotary
from trax.utils.shapes import ShapeDtype


class MlpType(str, Enum):
    MLP = "mlp"
    PARALLEL_GLU = "parallel_glu"


class NormalizationType(str, Enum):
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"


class GatingPosition(str, Enum):
    ATTENTION = "attention"
    MLP = "mlp"
    BOTH = "both"


def _coerce_enum(value, enum_cls, field_name):
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"{field_name} must be one of {[e.value for e in enum_cls]}."
        ) from exc


@dataclass
class ModernBertConfig:
    """Configuration for a modernized BERT-style encoder stack.

    Allowed string values:
      - mlp_type: "mlp", "parallel_glu"
      - normalization: "layernorm", "rmsnorm"
      - gating_position: "attention", "mlp", "both"
    """

    vocab_size: int = 30522
    max_len: int = 512
    num_layers: int = 12
    d_model: int = 768
    num_heads: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    embed_dropout: float = 0.1
    use_rope: bool = True
    rotary_dim: Optional[int] = None
    rotary_base: float = 10000.0
    use_flash_attention: bool = True
    flash_block_size: int = 128
    use_segment_mask: bool = True
    attn_qkv_bias: bool = False
    attn_out_bias: bool = False
    mlp_in_bias: bool = False
    mlp_out_bias: bool = False
    mlp_type: Union[MlpType, str] = MlpType.PARALLEL_GLU
    normalization: Union[NormalizationType, str] = NormalizationType.RMSNORM
    parallel_block: bool = True
    final_norm: bool = True
    tie_word_embeddings: bool = True
    use_gating: bool = False
    gating_position: Union[GatingPosition, str] = GatingPosition.ATTENTION

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.gating_position = _coerce_enum(
            self.gating_position, GatingPosition, "gating_position"
        )
        self.normalization = _coerce_enum(
            self.normalization, NormalizationType, "normalization"
        )
        self.mlp_type = _coerce_enum(self.mlp_type, MlpType, "mlp_type")

    @staticmethod
    def allowed_values():
        return {
            "gating_position": [e.value for e in GatingPosition],
            "normalization": [e.value for e in NormalizationType],
            "mlp_type": [e.value for e in MlpType],
        }


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


def _flash_attention_multihead(q, k, v, mask, block_size):
    # q, k, v: [batch, heads, length, head_dim]
    from trax.layers.research import flash_attention as fa

    b, h, l, d = q.shape
    qh = q.reshape(b * h, l, d)
    kh = k.reshape(b * h, l, d)
    vh = v.reshape(b * h, l, d)
    if mask is not None:
        mask = jnp.repeat(mask, h, axis=0)
    out = fa.flash_attention(qh, kh, vh, block_size=block_size, mask=mask)
    return out.reshape(b, h, l, d)


def _segment_ids_from_cu(cu_seqlens, length):
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


def _segment_mask_from_cu(attention_mask, cu_seqlens, max_seqlen, length):
    seg_ids = _segment_ids_from_cu(cu_seqlens, length)
    if seg_ids is None:
        return None
    seg_mismatch = seg_ids[:, :, None] != seg_ids[:, None, :]

    pad_mask = None
    if attention_mask is not None:
        pad_mask = attention_mask == 0
        seg_mismatch = seg_mismatch | pad_mask[:, :, None] | pad_mask[:, None, :]

    if max_seqlen is not None:
        max_seqlen = jnp.asarray(max_seqlen)
        if max_seqlen.ndim == 0:
            max_seqlen = jnp.full((seg_ids.shape[0],), max_seqlen)
        pos = jnp.arange(length)[None, :]
        over = pos >= max_seqlen[:, None]
        seg_mismatch = seg_mismatch | over[:, :, None] | over[:, None, :]

    return seg_mismatch


def _segment_attention(q, k, v, attention_mask, cu_seqlens, block_size):
    # Fallback, correctness-first attention per segment (not JIT-friendly).
    b, h, l, d = q.shape
    out = jnp.zeros_like(q)
    cu_seqlens = jax.device_get(cu_seqlens)
    if attention_mask is not None:
        attention_mask = jax.device_get(attention_mask)
    for batch_idx in range(b):
        cu = cu_seqlens[batch_idx]
        cu = cu[cu >= 0]
        for i in range(len(cu) - 1):
            start = int(cu[i])
            end = int(cu[i + 1])
            if end <= start:
                continue
            if attention_mask is not None:
                mask_slice = attention_mask[batch_idx, start:end]
                if mask_slice.sum() == 0:
                    continue
                end = start + int(mask_slice.sum())
                if end <= start:
                    continue
            q_seg = q[batch_idx : batch_idx + 1, :, start:end, :]
            k_seg = k[batch_idx : batch_idx + 1, :, start:end, :]
            v_seg = v[batch_idx : batch_idx + 1, :, start:end, :]
            seg_out = _flash_attention_multihead(q_seg, k_seg, v_seg, None, block_size)
            out = out.at[batch_idx, :, start:end, :].set(seg_out[0])
    return out


class ModernBertParallelBlock(base.Layer):
    """Parallel pre-norm encoder block with optional gating."""

    def __init__(self, config: ModernBertConfig, mode: str = "train"):
        super().__init__(n_in=4, n_out=4)
        self._config = config
        self._mode = mode

    def _norm(self, x, weights):
        scale = weights[0]
        if self._config.normalization == NormalizationType.RMSNORM:
            mean_square = jnp.mean(x * x, axis=-1, keepdims=True)
            x = x * jnp.reciprocal(jnp.sqrt(mean_square + 1e-6))
            return x * scale
        bias = weights[1]
        mean = jnp.mean(x, axis=-1, keepdims=True)
        centered = x - mean
        variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
        norm_inputs = centered / jnp.sqrt(variance + 1e-6)
        return norm_inputs * scale + bias

    def _attention(self, x, attention_mask, cu_seqlens, max_seqlen, weights, rng):
        wq, wk, wv, wo, bq, bk, bv, bo = weights
        q = jnp.dot(x, wq) + (bq if bq is not None else 0)
        k = jnp.dot(x, wk) + (bk if bk is not None else 0)
        v = jnp.dot(x, wv) + (bv if bv is not None else 0)
        q = _split_heads(q, self._config.num_heads)
        k = _split_heads(k, self._config.num_heads)
        v = _split_heads(v, self._config.num_heads)
        if self._config.use_rope:
            q, k = rotary.apply_rotary_embedding(
                q, k, rotary_dim=self._config.rotary_dim, base=self._config.rotary_base
            )
        mask = attention_mask == 0 if attention_mask is not None else None
        if self._config.use_flash_attention:
            if cu_seqlens is not None and self._config.use_segment_mask:
                length = q.shape[2]
                mask = _segment_mask_from_cu(attention_mask, cu_seqlens, max_seqlen, length)
                attn_out = _flash_attention_multihead(
                    q, k, v, mask, self._config.flash_block_size
                )
            elif cu_seqlens is not None:
                attn_out = _segment_attention(
                    q, k, v, attention_mask, cu_seqlens, self._config.flash_block_size
                )
            else:
                attn_out = _flash_attention_multihead(
                    q, k, v, mask, self._config.flash_block_size
                )
        else:
            scale = 1.0 / jnp.sqrt(q.shape[-1])
            dots = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
            if mask is not None:
                dots = jnp.where(mask[:, None, None, :], -1e9, dots)
            weights_attn = jax.nn.softmax(dots, axis=-1)
            if self._config.attention_dropout > 0.0 and self._mode == "train":
                weights_attn = _dropout(
                    weights_attn, self._config.attention_dropout, rng, self._mode
                )
            attn_out = jnp.einsum("bhqk,bhkd->bhqd", weights_attn, v)
        attn_out = _merge_heads(attn_out)
        attn_out = jnp.dot(attn_out, wo) + (bo if bo is not None else 0)
        attn_out = _dropout(attn_out, self._config.dropout, rng, self._mode)
        return attn_out

    def _mlp(self, x, weights, rng):
        w_in, b_in, w_out, b_out = weights
        if self._config.mlp_type == MlpType.PARALLEL_GLU:
            hidden = jnp.dot(x, w_in) + (b_in if b_in is not None else 0)
            gate, value = jnp.split(hidden, 2, axis=-1)
            hidden = jax.nn.gelu(gate) * value
        else:
            hidden = jnp.dot(x, w_in) + (b_in if b_in is not None else 0)
            hidden = jax.nn.gelu(hidden)
        hidden = _dropout(hidden, self._config.dropout, rng, self._mode)
        out = jnp.dot(hidden, w_out) + (b_out if b_out is not None else 0)
        out = _dropout(out, self._config.dropout, rng, self._mode)
        return out

    def init_weights_and_state(self, input_signature):
        x_sig = input_signature[0]
        d_model = x_sig.shape[-1]
        w_init = init.RandomNormalInitializer(0.02)
        rng_q, rng_k, rng_v, rng_o, rng_mlp, rng_gate = fastmath.random.split(
            self.rng, 6
        )
        norm_scale = jnp.ones((d_model,), dtype=x_sig.dtype)
        if self._config.normalization == NormalizationType.LAYERNORM:
            norm_bias = jnp.zeros((d_model,), dtype=x_sig.dtype)
            norm_weights = (norm_scale, norm_bias)
        else:
            norm_weights = (norm_scale,)

        wq = w_init((d_model, d_model), rng_q)
        wk = w_init((d_model, d_model), rng_k)
        wv = w_init((d_model, d_model), rng_v)
        wo = w_init((d_model, d_model), rng_o)
        if self._config.attn_qkv_bias:
            bq = jnp.zeros((d_model,), dtype=x_sig.dtype)
            bk = jnp.zeros((d_model,), dtype=x_sig.dtype)
            bv = jnp.zeros((d_model,), dtype=x_sig.dtype)
        else:
            bq = bk = bv = None
        if self._config.attn_out_bias:
            bo = jnp.zeros((d_model,), dtype=x_sig.dtype)
        else:
            bo = None
        attn_weights = (wq, wk, wv, wo, bq, bk, bv, bo)

        if self._config.mlp_type == MlpType.PARALLEL_GLU:
            w_in = w_init((d_model, self._config.mlp_dim * 2), rng_mlp)
        else:
            w_in = w_init((d_model, self._config.mlp_dim), rng_mlp)
        if self._config.mlp_in_bias:
            b_in = jnp.zeros((w_in.shape[-1],), dtype=x_sig.dtype)
        else:
            b_in = None
        w_out = w_init((self._config.mlp_dim, d_model), rng_mlp)
        if self._config.mlp_out_bias:
            b_out = jnp.zeros((d_model,), dtype=x_sig.dtype)
        else:
            b_out = None
        mlp_weights = (w_in, b_in, w_out, b_out)

        gate_weights = None
        if self._config.use_gating:
            gate_w = w_init((d_model, d_model), rng_gate)
            gate_b = jnp.zeros((d_model,), dtype=x_sig.dtype)
            gate_weights = (gate_w, gate_b)

        self.weights = (norm_weights, attn_weights, mlp_weights, gate_weights)
        self.state = ()

    def forward(self, inputs):
        x, attention_mask, cu_seqlens, max_seqlen = inputs
        norm_weights, attn_weights, mlp_weights, gate_weights = self.weights
        rng_attn, rng_mlp, rng_gate = fastmath.random.split(self.rng, 3)
        x_norm = self._norm(x, norm_weights)
        attn_out = self._attention(
            x_norm, attention_mask, cu_seqlens, max_seqlen, attn_weights, rng_attn
        )
        mlp_out = self._mlp(x_norm, mlp_weights, rng_mlp)
        if self._config.use_gating and gate_weights is not None:
            gate_w, gate_b = gate_weights
            gate = jax.nn.sigmoid(jnp.dot(x_norm, gate_w) + gate_b)
            if self._config.gating_position in (
                GatingPosition.ATTENTION,
                GatingPosition.BOTH,
            ):
                attn_out = gate * x + (1.0 - gate) * attn_out
            if self._config.gating_position in (
                GatingPosition.MLP,
                GatingPosition.BOTH,
            ):
                mlp_out = gate * x + (1.0 - gate) * mlp_out
            x = attn_out + mlp_out
        else:
            x = x + attn_out + mlp_out
        return (x, attention_mask, cu_seqlens, max_seqlen)


class ModernBertEncoder(base.Layer):
    """ModernBERT encoder with parallel blocks."""

    def __init__(self, config: ModernBertConfig, mode: str = "train"):
        super().__init__(n_in=5, n_out=1)
        self._config = config
        self._mode = mode
        self._blocks = [
            ModernBertParallelBlock(config, mode=mode) for _ in range(config.num_layers)
        ]

    def init_weights_and_state(self, input_signature):
        input_ids_sig = input_signature[0]
        d_model = self._config.d_model
        w_init = init.RandomNormalInitializer(0.02)
        tok_embed = w_init((self._config.vocab_size, d_model), self.rng)
        pos_embed = None
        if not self._config.use_rope:
            pos_embed = w_init((self._config.max_len, d_model), self.rng)

        block_weights = []
        x_sig = ShapeDtype(
            (input_ids_sig.shape[0], input_ids_sig.shape[1], d_model),
            input_ids_sig.dtype,
        )
        block_sig = (x_sig, input_signature[1], input_signature[3], input_signature[4])
        for block in self._blocks:
            bw, _ = block.init(block_sig)
            block_weights.append(bw)

        if self._config.final_norm:
            norm_scale = jnp.ones((d_model,), dtype=input_ids_sig.dtype)
            if self._config.normalization == NormalizationType.LAYERNORM:
                norm_bias = jnp.zeros((d_model,), dtype=input_ids_sig.dtype)
                final_norm = (norm_scale, norm_bias)
            else:
                final_norm = (norm_scale,)
        else:
            final_norm = None

        self.weights = (tok_embed, pos_embed, block_weights, final_norm)
        self.state = ()

    def _embed(self, input_ids, position_ids, weights, rng):
        tok_embed, pos_embed = weights
        x = tok_embed[input_ids]
        if pos_embed is not None and position_ids is not None:
            x = x + pos_embed[position_ids]
        x = _dropout(x, self._config.embed_dropout, rng, self._mode)
        return x

    def _final_norm(self, x, weights):
        if weights is None:
            return x
        scale = weights[0]
        if self._config.normalization == NormalizationType.RMSNORM:
            mean_square = jnp.mean(x * x, axis=-1, keepdims=True)
            x = x * jnp.reciprocal(jnp.sqrt(mean_square + 1e-6))
            return x * scale
        bias = weights[1]
        mean = jnp.mean(x, axis=-1, keepdims=True)
        centered = x - mean
        variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
        norm_inputs = centered / jnp.sqrt(variance + 1e-6)
        return norm_inputs * scale + bias

    def forward(self, inputs):
        input_ids, attention_mask, position_ids, cu_seqlens, max_seqlen = inputs
        tok_embed, pos_embed, block_weights, final_norm = self.weights
        rng_embed, rng_block = fastmath.random.split(self.rng, 2)
        x = self._embed(input_ids, position_ids, (tok_embed, pos_embed), rng_embed)
        for block, bw in zip(self._blocks, block_weights):
            block.weights = bw
            x, attention_mask, cu_seqlens, max_seqlen = block(
                (x, attention_mask, cu_seqlens, max_seqlen)
            )
        x = self._final_norm(x, final_norm)
        return x


class ModernBertForMaskedLM(base.Layer):
    """ModernBERT with MLM head."""

    def __init__(self, config: ModernBertConfig, mode: str = "train"):
        super().__init__(n_in=5, n_out=1)
        self._config = config
        self._mode = mode
        self._encoder = ModernBertEncoder(config, mode=mode)

    def init_weights_and_state(self, input_signature):
        enc_w, _ = self._encoder.init(input_signature)
        d_model = self._config.d_model
        w_init = init.RandomNormalInitializer(0.02)
        if self._config.tie_word_embeddings:
            proj_w = None
        else:
            proj_w = w_init((d_model, self._config.vocab_size), self.rng)
        proj_b = jnp.zeros((self._config.vocab_size,), dtype=input_signature[0].dtype)
        self.weights = (enc_w, proj_w, proj_b)
        self.state = ()

    def forward(self, inputs):
        enc_w, proj_w, proj_b = self.weights
        self._encoder.weights = enc_w
        x = self._encoder(inputs)
        if self._config.tie_word_embeddings:
            tok_embed = enc_w[0]
            logits = jnp.einsum("bld,vd->blv", x, tok_embed)
        else:
            logits = jnp.dot(x, proj_w) + proj_b
        return logits


def create_modern_bert(config: ModernBertConfig, mode: str = "train"):
    """Creates a ModernBERT encoder stack."""
    return ModernBertEncoder(config, mode=mode)


def create_modern_bert_mlm(config: ModernBertConfig, mode: str = "train"):
    """Creates a ModernBERT model with MLM head."""
    return ModernBertForMaskedLM(config, mode=mode)
