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

"""Serial-friendly T5 preprocessing utilities."""

import random

import gin
import numpy as np


@gin.configurable(module="trax.data")
def select_random_chunk_t5(  # pylint: disable=invalid-name
    sequence_length=None, training=True, seed=None
):
    """Selects a random input chunk and mirrors it as targets."""
    del training
    if sequence_length is None:
        return lambda g: g

    rng = random.Random(seed) if seed is not None else random

    def _select(generator):
        for example in generator:
            if isinstance(example, dict):
                tokens = np.asarray(example["inputs"])
                seq_len = tokens.shape[0]
                max_start = max(seq_len - sequence_length, 0)
                start = rng.randint(0, max_start) if max_start > 0 else 0
                chunk = tokens[start : start + sequence_length]
                updated = dict(example)
                updated["inputs"] = chunk
                updated["targets"] = chunk
                yield updated
            elif isinstance(example, (list, tuple)):
                tokens = np.asarray(example[0])
                seq_len = tokens.shape[0]
                max_start = max(seq_len - sequence_length, 0)
                start = rng.randint(0, max_start) if max_start > 0 else 0
                chunk = tokens[start : start + sequence_length]
                yield chunk, chunk
            else:
                raise ValueError(
                    "select_random_chunk_t5 expects dict or tuple/list examples."
                )

    return _select


@gin.configurable(module="trax.data")
def split_tokens_t5(  # pylint: disable=invalid-name
    sequence_length=None, training=True
):
    """Splits input tokens in half, using the second half as targets."""
    del training
    if sequence_length is None:
        return lambda g: g

    def _split(generator):
        for example in generator:
            if isinstance(example, dict):
                tokens = np.asarray(example["inputs"])
                split_point = tokens.shape[0] // 2
                updated = dict(example)
                updated["inputs"] = tokens[:split_point]
                updated["targets"] = tokens[split_point:]
                yield updated
            elif isinstance(example, (list, tuple)):
                tokens = np.asarray(example[0])
                split_point = tokens.shape[0] // 2
                yield tokens[:split_point], tokens[split_point:]
            else:
                raise ValueError("split_tokens_t5 expects dict or tuple/list examples.")

    return _split


@gin.configurable(module="trax.data")
def denoise_t5(  # pylint: disable=invalid-name
    sequence_length=None, training=True, noise_density=0.15, seed=None
):
    """Applies random token masking, returning masked inputs and clean targets."""
    del training
    if sequence_length is None:
        return lambda g: g

    rng = np.random.default_rng(seed)

    def _denoise(generator):
        for example in generator:
            if isinstance(example, dict):
                tokens = np.asarray(example["inputs"])
                mask = rng.random(tokens.shape) < noise_density
                noisy = np.where(mask, np.zeros_like(tokens), tokens)
                updated = dict(example)
                updated["inputs"] = noisy
                updated["targets"] = tokens
                yield updated
            elif isinstance(example, (list, tuple)):
                tokens = np.asarray(example[0])
                mask = rng.random(tokens.shape) < noise_density
                noisy = np.where(mask, np.zeros_like(tokens), tokens)
                yield noisy, tokens
            else:
                raise ValueError("denoise_t5 expects dict or tuple/list examples.")

    return _denoise
