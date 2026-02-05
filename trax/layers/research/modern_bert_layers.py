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

"""ModernBERT helper layers."""

import jax

from trax.layers import activation_fns
from trax.layers import combinators as cb
from trax.layers import core
from trax.layers import initializers as init
from trax.layers import base


def _bias_initializer(use_bias):
    if use_bias:
        return init.RandomNormalInitializer(0.0)
    return None


def ParallelGlu(hidden_dim, *, use_bias=True):
    """Parallel GLU projection used in ModernBERT MLPs."""
    kernel_init = init.RandomNormalInitializer(0.02)
    bias_init = _bias_initializer(use_bias)

    def _glu(x):
        gate, value = jax.numpy.split(x, 2, axis=-1)
        return jax.nn.gelu(gate) * value

    return cb.Serial(
        core.Dense(
            hidden_dim * 2,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            use_bias=use_bias,
        ),
        base.Fn("ParallelGlu", _glu),
    )


def ModernBertMlp(
    *,
    d_model,
    mlp_dim,
    mlp_type,
    mlp_in_bias,
    mlp_out_bias,
    dropout,
    mode,
):
    """Returns a ModernBERT MLP block."""
    kernel_init = init.RandomNormalInitializer(0.02)
    bias_init_in = _bias_initializer(mlp_in_bias)
    bias_init_out = _bias_initializer(mlp_out_bias)
    if mlp_type == "parallel_glu":
        return cb.Serial(
            ParallelGlu(mlp_dim, use_bias=mlp_in_bias),
            core.Dropout(rate=dropout, mode=mode),
            core.Dense(
                d_model,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init_out,
                use_bias=mlp_out_bias,
            ),
            core.Dropout(rate=dropout, mode=mode),
        )
    return cb.Serial(
        core.Dense(
            mlp_dim,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init_in,
            use_bias=mlp_in_bias,
        ),
        activation_fns.Gelu(),
        core.Dropout(rate=dropout, mode=mode),
        core.Dense(
            d_model,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init_out,
            use_bias=mlp_out_bias,
        ),
        core.Dropout(rate=dropout, mode=mode),
    )
