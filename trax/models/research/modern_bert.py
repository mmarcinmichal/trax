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

"""Modern BERT-style encoder stack for Trax.

Stack conventions (Serial stack semantics):
  - Encoder inputs: (input_ids, attention_mask, position_ids, cu_seqlens, max_seqlen)
  - Embeddings output: (x, attention_mask, cu_seqlens, max_seqlen)
  - Block inputs/outputs: (x, attention_mask, cu_seqlens, max_seqlen)
  - Encoder output: x
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import jax
import jax.numpy as jnp

from trax import fastmath
from trax import layers as tl
from trax.layers import base, normalization
from trax.layers import combinators as cb
from trax.layers import initializers as init
from trax.layers.attention import flash as flash_attention
from trax.layers.research import modern_bert_layers
from trax.layers.research import rotary_positional_embedding as rotary


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
    attention_implementation: str = "flash"
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
        if self.attention_implementation not in ("flash", "jax", "naive"):
            raise ValueError(
                "attention_implementation must be one of: 'flash', 'jax', 'naive'."
            )

    @staticmethod
    def allowed_values():
        return {
            "gating_position": [e.value for e in GatingPosition],
            "normalization": [e.value for e in NormalizationType],
            "mlp_type": [e.value for e in MlpType],
            "attention_implementation": ["flash", "jax", "naive"],
        }


def _norm_layer(config):
    if config.normalization == NormalizationType.RMSNORM:
        return normalization.RMSNorm(epsilon=1e-6)
    return normalization.LayerNorm(epsilon=1e-6)


def _split_heads(x, n_heads):
    b, l, d = x.shape
    head_dim = d // n_heads
    return x.reshape(b, l, n_heads, head_dim).transpose(0, 2, 1, 3)


def _merge_heads(x):
    b, h, l, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, l, h * d)


def _dropout(x, rate, rng, mode):
    if rate <= 0.0 or mode != "train":
        return x
    keep_prob = 1.0 - rate
    mask = fastmath.random.bernoulli(rng, keep_prob, x.shape)
    return x * mask / keep_prob


class SimpleSelfAttention(base.Layer):
    """Standard dot-product self-attention (non-flash fallback)."""

    def __init__(
        self,
        config: ModernBertConfig,
        *,
        mode: str = "train",
    ):
        super().__init__(n_in=4, n_out=1)
        self._config = config
        self._mode = mode

    def init_weights_and_state(self, input_signature):
        x_sig = input_signature[0]
        d_model = x_sig.shape[-1]
        w_init = init.RandomNormalInitializer(0.02)
        wq = w_init((d_model, d_model), self.rng)
        wk = w_init((d_model, d_model), self.rng)
        wv = w_init((d_model, d_model), self.rng)
        wo = w_init((d_model, d_model), self.rng)
        if self._config.attn_qkv_bias:
            bq = jnp.zeros((d_model,), dtype=x_sig.dtype)
            bk = jnp.zeros((d_model,), dtype=x_sig.dtype)
            bv = jnp.zeros((d_model,), dtype=x_sig.dtype)
        else:
            bq = bk = bv = None
        bo = (
            jnp.zeros((d_model,), dtype=x_sig.dtype)
            if self._config.attn_out_bias
            else None
        )
        self.weights = (wq, wk, wv, wo, bq, bk, bv, bo)
        self.state = ()

    def forward(self, inputs):
        x, attention_mask, _, _ = inputs
        wq, wk, wv, wo, bq, bk, bv, bo = self.weights
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
        scale = 1.0 / jnp.sqrt(q.shape[-1])
        dots = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if mask is not None:
            dots = jnp.where(mask[:, None, None, :], -1e9, dots)
        weights_attn = jax.nn.softmax(dots, axis=-1)
        if self._config.attention_dropout > 0.0 and self._mode == "train":
            weights_attn = _dropout(
                weights_attn, self._config.attention_dropout, self.rng, self._mode
            )
        attn_out = jnp.einsum("bhqk,bhkd->bhqd", weights_attn, v)
        attn_out = _merge_heads(attn_out)
        attn_out = jnp.dot(attn_out, wo) + (bo if bo is not None else 0)
        return attn_out


class ModernBertEmbeddings(base.Layer):
    """Token + position embeddings for ModernBERT."""

    def __init__(self, config: ModernBertConfig):
        super().__init__(n_in=5, n_out=4)
        self._config = config

    def init_weights_and_state(self, input_signature):
        input_ids_sig = input_signature[0]
        d_model = self._config.d_model
        w_init = init.RandomNormalInitializer(0.02)
        tok_embed = w_init((self._config.vocab_size, d_model), self.rng)
        pos_embed = None
        if not self._config.use_rope:
            pos_embed = w_init((self._config.max_len, d_model), self.rng)
        self.weights = (tok_embed, pos_embed)
        self.state = ()

    def forward(self, inputs):
        input_ids, attention_mask, position_ids, cu_seqlens, max_seqlen = inputs
        tok_embed, pos_embed = self.weights
        x = tok_embed[input_ids]
        if pos_embed is not None:
            if position_ids is None:
                position_ids = jnp.arange(x.shape[1])[None, :]
            x = x + pos_embed[position_ids]
        return x, attention_mask, cu_seqlens, max_seqlen


class ModernBertCombine(base.Layer):
    """Combines attention and MLP outputs with optional gating."""

    def __init__(self, config: ModernBertConfig):
        super().__init__(n_in=7, n_out=4)
        self._config = config

    def init_weights_and_state(self, input_signature):
        x_sig = input_signature[0]
        if self._config.use_gating:
            w_init = init.RandomNormalInitializer(0.02)
            gate_w = w_init((x_sig.shape[-1], x_sig.shape[-1]), self.rng)
            gate_b = jnp.zeros((x_sig.shape[-1],), dtype=x_sig.dtype)
            self.weights = (gate_w, gate_b)
        else:
            self.weights = ()
        self.state = ()

    def forward(self, inputs):
        x, x_norm, attn_out, mlp_out, attention_mask, cu_seqlens, max_seqlen = inputs
        if self._config.use_gating:
            gate_w, gate_b = self.weights
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


class ModernBertParallelBlock(tl.Serial):
    """Parallel pre-norm encoder block with optional gating."""

    def __init__(self, config: ModernBertConfig, mode: str = "train"):
        if not config.parallel_block:
            raise ValueError("Only parallel blocks are supported in ModernBERT.")
        norm = _norm_layer(config)
        if config.use_flash_attention:
            attention = flash_attention.FlashSelfAttention(
                d_model=config.d_model,
                n_heads=config.num_heads,
                block_size=config.flash_block_size,
                attention_dropout=config.attention_dropout,
                use_rope=config.use_rope,
                rotary_dim=config.rotary_dim,
                rotary_base=config.rotary_base,
                use_segment_mask=config.use_segment_mask,
                attn_qkv_bias=config.attn_qkv_bias,
                attn_out_bias=config.attn_out_bias,
                implementation=config.attention_implementation,
                mode=mode,
            )
        else:
            attention = SimpleSelfAttention(config, mode=mode)
        mlp = modern_bert_layers.ModernBertMlp(
            d_model=config.d_model,
            mlp_dim=config.mlp_dim,
            mlp_type=config.mlp_type.value,
            mlp_in_bias=config.mlp_in_bias,
            mlp_out_bias=config.mlp_out_bias,
            dropout=config.dropout,
            mode=mode,
        )
        attn_dropout = tl.Dropout(rate=config.dropout, mode=mode)
        super().__init__(
            cb.Branch(
                cb.Select([0], n_in=4),
                cb.Serial(cb.Select([0], n_in=4), norm),
                cb.Select([1, 2, 3], n_in=4),
            ),
            cb.Branch(
                cb.Select([0], n_in=5),
                cb.Select([1], n_in=5),
                cb.Serial(
                    cb.Select([1, 2, 3, 4], n_in=5),
                    attention,
                    attn_dropout,
                ),
                cb.Serial(cb.Select([1], n_in=5), mlp),
                cb.Select([2, 3, 4], n_in=5),
            ),
            ModernBertCombine(config),
        )


def ModernBertFinalNorm(config: ModernBertConfig, mode: str):
    if not config.final_norm:
        return cb.Select([0, 1, 2, 3], n_in=4)
    return cb.Parallel(
        _norm_layer(config),
        cb.Serial(),
        cb.Serial(),
        cb.Serial(),
    )


class ModernBertEncoder(tl.Serial):
    """ModernBERT encoder with parallel blocks."""

    def __init__(self, config: ModernBertConfig, mode: str = "train"):
        embeddings = ModernBertEmbeddings(config)
        embed_dropout = cb.Parallel(
            tl.Dropout(rate=config.embed_dropout, mode=mode),
            cb.Serial(),
            cb.Serial(),
            cb.Serial(),
        )
        blocks = [
            ModernBertParallelBlock(config, mode=mode) for _ in range(config.num_layers)
        ]
        final_norm = ModernBertFinalNorm(config, mode)
        super().__init__(
            embeddings,
            embed_dropout,
            *blocks,
            final_norm,
            cb.Select([0], n_in=4),
        )
        self._embeddings = embeddings

    @property
    def embeddings_layer(self):
        return self._embeddings


class ModernBertMlmHead(base.Layer):
    """MLM head with optional tied embeddings."""

    def __init__(self, config: ModernBertConfig, embedding_layer: ModernBertEmbeddings):
        super().__init__(n_in=1, n_out=1)
        self._config = config
        self._embedding_layer = embedding_layer

    def init_weights_and_state(self, input_signature):
        vocab_size = self._config.vocab_size
        if self._config.tie_word_embeddings:
            proj_w = None
        else:
            w_init = init.RandomNormalInitializer(0.02)
            proj_w = w_init((input_signature.shape[-1], vocab_size), self.rng)
        proj_b = jnp.zeros((vocab_size,), dtype=input_signature.dtype)
        self.weights = (proj_w, proj_b)
        self.state = ()

    def forward(self, inputs):
        proj_w, proj_b = self.weights
        if self._config.tie_word_embeddings:
            tok_embed = self._embedding_layer.weights[0]
            logits = jnp.einsum("bld,vd->blv", inputs, tok_embed) + proj_b
        else:
            logits = jnp.dot(inputs, proj_w) + proj_b
        return logits


class ModernBertForMaskedLM(tl.Serial):
    """ModernBERT with MLM head."""

    def __init__(self, config: ModernBertConfig, mode: str = "train"):
        encoder = ModernBertEncoder(config, mode=mode)
        head = ModernBertMlmHead(config, embedding_layer=encoder.embeddings_layer)
        super().__init__(encoder, head)


def create_modern_bert(config: ModernBertConfig, mode: str = "train"):
    """Creates a ModernBERT encoder stack."""
    return ModernBertEncoder(config, mode=mode)


def create_modern_bert_mlm(config: ModernBertConfig, mode: str = "train"):
    """Creates a ModernBERT model with MLM head."""
    return ModernBertForMaskedLM(config, mode=mode)
