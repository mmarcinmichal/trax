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

from trax import layers as tl
from trax.layers.research import rotary_positional_embedding as rotary


@dataclass
class ModernBertConfig:
    """Configuration for a modernized BERT-style encoder stack."""

    vocab_size: int = 30522
    max_len: int = 512
    num_layers: int = 12
    d_model: int = 768
    num_heads: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_rope: bool = False
    use_gating: bool = False
    gating_position: str = "attention"

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.gating_position not in ("attention", "mlp"):
            raise ValueError("gating_position must be 'attention' or 'mlp'.")


def _maybe_rope(use_rope: bool):
    if not use_rope:
        return tl.Serial()
    return tl.Fn("ApplyRoPE", rotary.rotate)


def _gated_residual(layer, d_model, use_gating):
    if not use_gating:
        return tl.Residual(layer)
    gate = tl.Serial(tl.LayerNorm(), tl.Dense(d_model), tl.Sigmoid())
    return tl.Serial(
        tl.Branch([], gate, layer),
        tl.Gate(),
    )


def _encoder_block(config, mode="train"):
    attention = tl.Serial(
        tl.LayerNorm(),
        _maybe_rope(config.use_rope),
        tl.Attention(
            config.d_model,
            n_heads=config.num_heads,
            dropout=config.attention_dropout,
            mode=mode,
        ),
        tl.Dropout(rate=config.dropout, mode=mode),
    )
    mlp = tl.Serial(
        tl.LayerNorm(),
        tl.Dense(config.mlp_dim),
        tl.Gelu(),
        tl.Dropout(rate=config.dropout, mode=mode),
        tl.Dense(config.d_model),
        tl.Dropout(rate=config.dropout, mode=mode),
    )

    attention_block = _gated_residual(
        attention,
        config.d_model,
        config.use_gating and config.gating_position == "attention",
    )
    mlp_block = _gated_residual(
        mlp,
        config.d_model,
        config.use_gating and config.gating_position == "mlp",
    )
    return [attention_block, mlp_block]


def create_modern_bert(config: ModernBertConfig, mode: str = "train"):
    """Creates a full ModernBert-style encoder with token embeddings."""
    positional = []
    if not config.use_rope:
        positional = [tl.PositionalEncoding(config.max_len, mode=mode)]
    return tl.Serial(
        tl.Branch([], tl.PaddingMask()),
        tl.Embedding(config.vocab_size, config.d_model),
        tl.Dropout(rate=config.dropout, mode=mode),
        positional,
        [_encoder_block(config, mode=mode) for _ in range(config.num_layers)],
        tl.Select([0], n_in=2),
        tl.LayerNorm(),
    )
