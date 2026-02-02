# coding=utf-8
# Copyright 2022 The Trax Authors.
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

"""Data sources and input processing.

Trax authors recommend constructing input pipelines using layer-like functions
and combinators. For example, following is an input pipeline for training
sentiment analysis tasks on the IMDB dataset::

  from trax import data

  inputs = data.Serial(
    data.TFDS('imdb_reviews', keys=('text', 'label'), train=True),
    data.Tokenize(vocab_file='en_8k.subword', keys=[0]),
    data.Shuffle(),
    data.FilterByLength(max_length=2048, length_keys=[0]),
    data.BucketByLength(boundaries=[  32, 128, 512, 2048],
                        batch_sizes=[128,  32,   8,    2, 1],
                        length_keys=[0]),
    data.AddLossWeights()
  )

Each of these functions creates a Python generator of tuples of data arrays.
For example::

  data.TFDS('imdb_reviews', keys=('text', 'label'), train=True),

creates a generator of examples (tuples of NumPy :py:class:`ndarray` objects)
from the TFDS imdb_reviews dataset, see here:
https://www.tensorflow.org/datasets/catalog/imdb_reviews

As you can see on the website above, this dataset has 'text' and 'label' fields
and we create tuples containing the text and the label from the training split
by specifying keys=('text', 'label'), train=True.

Other functions, like ``Tokenize`` and ``Shuffle``, take a generator and output
another generator, in this way converting tuples into other tuples or mixing
the training stream. For example, ``Tokenize(..., keys=[0])`` tokenizes the
first element of a tuple -- converting it from text to a NumPy integer array.
And ``Shuffle`` randomizes the order of examples.

Note that all elements in the data pipeline are just functions on generators,
so you can use Python's `map` and `filter` and other native functions too.
For example, you can create an input pipeline for a language model reading
lines from `my_file.txt` as follows::

  inputs = data.Serial(
    lambda _: open('my_file.txt'),
    lambda g: map(lambda line: line.strip(), g),
    data.Tokenize(vocab_file='en_8k.subword'),
    lambda g: filter(lambda x: x.shape[0] < 513, g),  # At most 512 tokens.
    data.Shuffle(),
    lambda g: map(lambda x: (x, x)),  # Language models have inputs = targets.
    data.BucketByLength(boundaries=[  32, 64, 128, 256, 512],
                        batch_sizes=[ 32, 16,  8,    4,   2, 1]),
    data.AddLossWeights(id_to_mask=0)
  )

"""

import math
import multiprocessing.dummy as mp  # using threads for now
import os
import pickle
import random
import time

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import gin
import jax
import numpy as np
import tensorflow as tf

from trax.utils import logging as trax_logging

from trax import fastmath
from trax.data.debugger import data_pipeline as debug_data_pipeline
from trax.data.encoder.encoder import SentencePieceEncoder
from trax.data.loader.tf.interface import DatasetLoader, DatasetStreams
from trax.utils import shapes

# Alias fastmath.numpy for local use without relying on import-as syntax.
jnp = fastmath.numpy


# For now, we skip at most 100K examples for efficiency.
_MAX_SKIP_EXAMPLES = 100_000

def Serial(*fns):  # pylint: disable=invalid-name
    """Combines generator functions into one that runs them serially."""

    def composed_fns(generator=None):
        for f in fastmath.tree_flatten(fns):
            generator = f(generator)
        return generator

    return composed_fns


def _identity_preprocess(generator):
    return generator


@gin.configurable(module="trax.data")
def unsupervised_preprocessors(preprocessors=None):  # pylint: disable=invalid-name
    """Combines preprocessors into a Serial pipeline."""
    if preprocessors is None:
        return lambda g: g
    return Serial(*preprocessors)


@gin.configurable(module="trax.data")
def NextSentencePrediction(  # pylint: disable=invalid-name
    text_key="text",
    buffer_size=50000,
    seed=None,
):
    """Builds next sentence prediction pairs from a text stream."""
    rng = random.Random(seed) if seed is not None else random

    def _extract_text(example):
        if isinstance(example, dict):
            return example.get(text_key)
        if isinstance(example, (list, tuple)) and example:
            return example[0]
        return example

    def _first_two_sentences(text):
        text = _text_to_str(text)
        if isinstance(text, np.ndarray):
            return None, None
        parts = text.split(". ")
        first = parts[0].strip()
        second = parts[1].strip() if len(parts) > 1 else first
        return first, second

    def _nsp(stream):
        buffer = []
        for example in stream:
            text = _extract_text(example)
            if text is None:
                continue
            sent1, sent2 = _first_two_sentences(text)
            if sent1 is None:
                continue
            use_random = rng.random() < 0.5
            if use_random:
                if buffer:
                    other_text = buffer[rng.randrange(len(buffer))]
                else:
                    other_text = text
                other_sent1, _ = _first_two_sentences(other_text)
                if other_sent1 is None:
                    continue
                sent2 = other_sent1
            label = not use_random
            yield sent1, sent2, label
            buffer.append(text)
            if len(buffer) > buffer_size:
                buffer.pop(0)

    return _nsp


def Parallel(  # pylint: disable=invalid-name
    fns=None,
    counters=None,
    reweight_by_minimum=False,
    gradually_reweight=False,
    use_remainders=False,
):
    """Combines generator functions into one that runs them in parallel.

    Args:
      fns: a sequence of datasets which are combined in parallel.
      counters: a sequence of ints with same length as fns, please see comments on
        its use below.
      reweight_by_minimum: if set to True, then we re-weight every counter by the
        minimal counter. E.g. counters (10000, 100000) are translated to (1, 10)
        and hence for every 10 examples from the second dataset we are getting
        1 example from the first dataset. Without reweighting first we would see
        20 examples from the first and second dataset and then 90 thousand eamples
        only from the first dataset.
      gradually_reweight: if set to True, then we loop through the generators
        using a recursive rule defined in emit_examples. First we sort generators
        by the counters. If we have datasets with counters 1, 20, 40
        (after sorting) then we yield examples (a(b c^2)^20)^*, where examples of
        type a come from the first dataset, of type b from the second and of type
        c from the third. The exponents are obtained through divisions of
        subsequent counters.
      use_remainders: if set to True as weell as gradually_reweight is set to
        True and counters are 1, 20, 45 then after dealing with all examples in
        the format (a(b c^2)^20)^*, the generator yields the remaining 5 examples
        from the dataset with counter 45.
    Returns:
      parallel_generator: the generator yields samples according to given;
      if counters are not given then samples are genereted uniformly.

    Example 1:

      gen = data.Parallel([dataset1, dataset2, dataset3], counters=(2, 1, 3))

    defines a generator that yields 33% examples from dataset1, 16% examples from
    dataset2 and 50% examples from dataset3.

    Example 2:

      gen = data.Parallel([dataset1, dataset2, dataset3], counters=(20, 50, 30))

    defines a generator that yields 20% examples from dataset1, 50% examples from
    dataset2 and 30% examples from dataset3.
    """

    if counters:
        assert len(counters) == len(fns)
        # Remove generators with zero counters
        counters = list(counters)
        fns = list(fns)
        non_zeros = [j for j in range(len(counters)) if counters[j] != 0]
        counters = [counters[j] for j in non_zeros]
        fns = [fns[j] for j in non_zeros]
    else:
        counters = [1] * len(fns)

    if reweight_by_minimum:
        counters = [math.floor(counter / min(counters)) for counter in counters]

    def emit_examples(sorted_counters_with_gens, prev_counter):
        if sorted_counters_with_gens:
            _, counter, generator = sorted_counters_with_gens[0]
            repeats = math.floor(counter / prev_counter)
            for _ in range(repeats):
                yield next(generator)
                yield from emit_examples(sorted_counters_with_gens[1:], counter)

    def parallel_generator(gen=None):
        # If gradually_reweight is set to False then
        # current_counters are increased step by step; they are reset to 0s when
        # current_counters[idx] == counters[idx] for all idx. See
        # test_parallel_with_weights_three_datasets for an example of how
        # current_counters are changed during computation.
        # If gradually_reweight is set to False then we loop using a
        # recursive rule defined in emit_examples.

        generators = []
        for f in fns:
            if gen:
                generators.append(f(gen))
            else:
                # This handles the case when the function f cannot be
                # called on None.
                generators.append(f())

        if gradually_reweight:
            counters_with_gens = zip(range(len(generators)), counters, generators)
            sorted_counters_with_gens = sorted(counters_with_gens, key=lambda x: x[1])
            while True:
                yield from emit_examples(sorted_counters_with_gens, min(counters))
                if use_remainders:
                    # Below we are dealing with remainders.
                    fractions = []
                    for i in range(len(sorted_counters_with_gens)):
                        _, counter, generator = sorted_counters_with_gens[i]
                        processed = 1
                        for fraction in fractions:
                            processed *= fraction
                        remainder = counter - processed
                        for _ in range(remainder):
                            yield next(generator)
                        if i < len(sorted_counters_with_gens) - 1:
                            _, next_counter, _ = sorted_counters_with_gens[i + 1]
                            fractions.append(math.floor(next_counter / counter))
        else:
            current_counters = [0] * len(generators)
            while True:
                for idx, generator in enumerate(generators):
                    if current_counters[idx] < counters[idx]:
                        current_counters[idx] += 1
                        # instead of checking current_counters[idx] == counters[idx] for
                        # all idx, we check the equivalent condition:
                        if sum(current_counters) == sum(counters):
                            current_counters = [0] * len(generators)
                        yield next(generator)

    return parallel_generator


@gin.configurable(module="trax.data")
def Shuffle(queue_size=1024):  # pylint: disable=invalid-name
    """Returns a shuffle function with the given queue size."""
    return lambda g: shuffle(g, queue_size)


@gin.configurable(module="trax.data")
def Batch(batch_size):  # pylint: disable=invalid-name
    """Returns a batching function with given batch size."""
    return lambda g: batch(g, batch_size)


@gin.configurable(module="trax.data")
def Dup():  # pylint: disable=invalid-name
    """Duplicates (copies) the top element (inputs).

    The generator stream is augmented in the following way:

    - If the stream consists of a single element `(inputs, )`,
      the inputs simply get copied to `(inputs, inputs)`.
    - If the stream consists of multiple elements, for example
      `(inputs, weights)`, the rest of elements get moved toward
      the right side `(inputs, inputs, weights)`.

    Returns:
      the duplicating function.
    """

    def _copy(xs):
        x, *rest = xs
        return (x, x, *rest)

    return lambda g: map(lambda x: _copy(x), g)  # pylint: disable=unnecessary-lambda


@gin.configurable(module="trax.data")
def FilterEmptyExamples(axes=None, debug=False):  # pylint: disable=invalid-name
    """Filters empty examples.

    Filters any example that has an array of size (0,) (if axes=None).
    Alternatively, checks only axes provided in `axes' list. Contrary to
    FilterByLength used with several elements with length_axis, here the example
    would be filtered if ANY of the dimensions listed in `axes' contains an empty
    array.

    Args:
      axes: list of indices to check, if None, all of them.
      debug: If true, emits a log everytime we filter out an empty example.

    Returns:
      Function filtering empty examples.
    """

    def _filter_examples(generator):
        for example in generator:
            correct = True
            for i, unused_tuple_element in enumerate(example):
                if axes is None or i in axes:
                    if example[i].shape == (0,):
                        correct = False
                        break
            if correct:
                yield example
            elif debug:
                trax_logging.info("Filtered example: %r", example)

    return _filter_examples


@gin.configurable(module="trax.data")
def FilterByLength(
    max_length,
    min_length=0,  # pylint: disable=invalid-name
    length_keys=None,
    length_axis=0,
):
    """Returns a function that filters out examples by length.

    Args:
      max_length: int. If not None, indicates maximum length.
      min_length: int. If not None, indicates minimum length.
      length_keys: (list) which example keys to take into account.
      length_axis: which shape axis to take into account.
    Returns:
      a function that filters out examples by length.
    """

    assert max_length is not None or min_length is not None
    length_keys = length_keys or [0, 1]
    def length_fn(x):
        return length_fn_(x, length_axis, length_keys)

    def filtered(gen):
        for example in gen:
            example_len = length_fn(example)

            # Checking max length boundary.
            if max_length is not None:
                if example_len > max_length:
                    continue
            # Checking min length boundary.
            if min_length is not None:
                if example_len < min_length:
                    continue
            # Within bounds.
            yield example

    return filtered


@gin.configurable(module="trax.data")
def TruncateToLength(len_map=None):  # pylint: disable=invalid-name
    """Returns a stream function that resizes items as specified by ``len_map``.

    Args:
      len_map: Dictionary that specifies maximum shapes for potentially multiple
          features per stream item. For example, given a stream of tokenized
          string pairs, one could enforce a maximum length of 256 tokens for each
          string by using ``len_map={0: (256,), 1: (256,)}``.
    """

    @debug_data_pipeline.debug_pipeline
    def _truncate_to_length(generator):
        for example in generator:
            if isinstance(example, np.ndarray):
                example = (example,)
            if isinstance(example, (list, tuple)):
                example = list(example)
                if len_map is not None:
                    for key, max_len in len_map.items():
                        example_len = example[key].shape
                        if example_len > max_len:
                            example[key] = np.resize(example[key], max_len)
                output = tuple(example)
            else:
                raise ValueError(f"Unknown example type: {example}")
            yield output

    return _truncate_to_length


@gin.configurable(module="trax.data")
def PadToLength(  # pylint: disable=invalid-name
    len_map=None, pad_value=0, multiple=False
):
    """Pads the values to lengths given in `len_map'.

    len_map contains a dictionary of example keys to dimension sizes.

    Args:
      len_map: dict of int to int, we pad examples to lengths
        given by the values of the dict. If multiple is True, the dimensions are
        padded to multiple of this value.
      pad_value: dict of int to int. The value gets applied to
        constant_values on numpy.pad per given dimension.
      multiple: boolean. If False, pads to the value of len_map. If True, pads to
        closest multiple of value of len_map.
    Returns:
      Function to pad examples to given lengths.
    """

    @debug_data_pipeline.debug_pipeline
    def _pad_to_length(generator):
        for example in generator:
            if isinstance(example, (list, tuple)):
                example = list(example)
                for key, value in len_map.items():
                    array_length = example[key].shape[0]
                    if multiple:
                        padding_len = array_length - ((array_length // value) * value)
                    else:
                        padding_len = max([0, value - example[key].shape[0]])
                    example[key] = np.pad(
                        example[key],
                        pad_width=(0, padding_len),
                        mode="constant",
                        constant_values=pad_value[key],
                    )
                output = tuple(example)
            else:
                if not isinstance(example, np.ndarray):
                    raise ValueError(f"example isn't nparray, but should be: {example}")
                array_length = example.shape[0]
                if multiple:
                    padding_len = array_length - (
                        (array_length // len_map[0]) * len_map[0]
                    )
                else:
                    padding_len = max(0, len_map[0] - array_length)
                output = np.pad(
                    example,
                    pad_width=(0, padding_len),
                    mode="constant",
                    constant_values=pad_value[0],
                )
            yield output

    if len_map is None:
        raise ValueError("len_map parameter should be provided.")
    return _pad_to_length


@gin.configurable(module="trax.data")
def ConcatInputsTargets(pad_symbol=0):  # pylint: disable=invalid-name
    """Concatenates inputs and targets with a pad symbol between."""

    def _concat(generator):
        for example in generator:
            if isinstance(example, (list, tuple)) and len(example) == 2:
                features, targets = example
                if isinstance(features, dict):
                    inputs = np.asarray(features.get("inputs"))
                else:
                    inputs = np.asarray(features)
                targets = np.asarray(targets)
                pad = np.zeros_like(inputs[:1]) + pad_symbol
                concatenated = np.concatenate([pad, inputs, pad, targets], axis=0)
                if isinstance(features, dict):
                    updated = dict(features)
                    updated["inputs"] = concatenated
                    yield updated, concatenated
                else:
                    yield concatenated, concatenated
            elif isinstance(example, dict):
                inputs = np.asarray(example.get("inputs"))
                targets = np.asarray(example.get("targets"))
                pad = np.zeros_like(inputs[:1]) + pad_symbol
                concatenated = np.concatenate([pad, inputs, pad, targets], axis=0)
                updated = dict(example)
                updated["inputs"] = concatenated
                yield updated, concatenated
            else:
                raise ValueError(
                    "ConcatInputsTargets expects (features, targets) or dict examples."
                )

    return _concat


@gin.configurable(module="trax.data")
def SqueezeTargets():  # pylint: disable=invalid-name
    """Squeezes the last axis of targets if it is size 1."""

    def _squeeze(generator):
        for example in generator:
            if isinstance(example, (list, tuple)) and len(example) == 2:
                features, targets = example
                targets = np.asarray(targets)
                if targets.shape[-1] == 1:
                    targets = np.squeeze(targets, axis=-1)
                yield features, targets
            elif isinstance(example, dict):
                updated = dict(example)
                targets = np.asarray(updated["targets"])
                if targets.shape[-1] == 1:
                    targets = np.squeeze(targets, axis=-1)
                updated["targets"] = targets
                yield updated
            else:
                raise ValueError("SqueezeTargets expects dict or (features, targets).")

    return _squeeze


@gin.configurable(module="trax.data")
def LM1BFilterByLength(  # pylint: disable=invalid-name
    max_target_length=-1, max_eval_target_length=-1, training=True, target_key=1
):
    """Filters examples by target length for LM1B-style datasets."""
    max_len = max_target_length if training else max_eval_target_length
    if max_len <= 0:
        return lambda g: g

    def _filter(generator):
        for example in generator:
            if isinstance(example, dict):
                target = np.asarray(example["targets"])
            elif isinstance(example, (list, tuple)) and len(example) > target_key:
                target = np.asarray(example[target_key])
            else:
                raise ValueError("LM1BFilterByLength expects dict or tuple example.")
            if target.shape[0] < max_len + 1:
                yield example

    return _filter


@gin.configurable(module="trax.data")
def LMTokenPreprocess():  # pylint: disable=invalid-name
    """Concatenates inputs, pad, targets and adds loss mask for targets."""

    def _process(generator):
        for example in generator:
            if isinstance(example, dict):
                inputs = np.asarray(example["inputs"])
                targets = np.asarray(example["targets"])
                pad = np.zeros_like(inputs[:1])
                concatenated = np.concatenate([inputs, pad, targets], axis=0)
                mask = np.concatenate(
                    [np.zeros_like(inputs), pad, np.ones_like(targets)], axis=0
                ).astype(np.float32)
                updated = dict(example)
                updated["inputs"] = concatenated
                updated["targets"] = concatenated
                updated["mask"] = mask
                yield updated
            elif isinstance(example, (list, tuple)) and len(example) == 2:
                inputs = np.asarray(example[0])
                targets = np.asarray(example[1])
                pad = np.zeros_like(inputs[:1])
                concatenated = np.concatenate([inputs, pad, targets], axis=0)
                mask = np.concatenate(
                    [np.zeros_like(inputs), pad, np.ones_like(targets)], axis=0
                ).astype(np.float32)
                yield concatenated, concatenated, mask
            else:
                raise ValueError("LMTokenPreprocess expects dict or (inputs, targets).")

    return _process


@gin.configurable(module="trax.data")
def RandomSplitText(  # pylint: disable=invalid-name
    max_words_per_segment=512, text_key="text", seed=None
):
    """Randomly selects a contiguous word chunk from text examples."""
    rng = random.Random(seed) if seed is not None else random

    def _chunk_text(text_value):
        if isinstance(text_value, bytes):
            text_value = text_value.decode("utf-8")
        words = str(text_value).split()
        length = len(words)
        if length == 0:
            return ""
        max_len = min(length, max_words_per_segment)
        max_start = length - max_len
        start = rng.randint(0, max_start) if max_start > 0 else 0
        return " ".join(words[start : start + max_len])

    def _split(generator):
        for example in generator:
            if isinstance(example, dict):
                updated = dict(example)
                updated[text_key] = _chunk_text(updated[text_key])
                yield updated
            elif isinstance(example, (list, tuple)):
                if not isinstance(text_key, int):
                    raise ValueError(
                        "RandomSplitText expects integer text_key for tuple examples."
                    )
                updated = list(example)
                updated[text_key] = _chunk_text(updated[text_key])
                yield tuple(updated)
            else:
                raise ValueError("RandomSplitText expects dict or tuple examples.")

    return _split


@gin.configurable(module="trax.data")
def FilterByLengthMap(  # pylint: disable=invalid-name
    len_map=None, filter_on_eval=False, training=True
):
    """Filters examples by min/max lengths per key from len_map."""
    if len_map is None or (not training and not filter_on_eval):
        return lambda g: g

    def _filter(generator):
        for example in generator:
            ok = True
            for key, bounds in len_map.items():
                min_len, max_len = bounds
                if isinstance(example, dict):
                    value = example[key]
                else:
                    value = example[key]
                size = np.asarray(value).shape[0]
                if size < min_len or size > max_len:
                    ok = False
                    break
            if ok:
                yield example

    return _filter


@gin.configurable(module="trax.data")
def TruncateByLengthMap(  # pylint: disable=invalid-name
    len_map=None, truncate_on_eval=False, training=True
):
    """Truncates examples to max lengths per key from len_map."""
    if len_map is None or (not training and not truncate_on_eval):
        return lambda g: g

    def _truncate(generator):
        for example in generator:
            if isinstance(example, dict):
                updated = dict(example)
                for key, max_len in len_map.items():
                    value = np.asarray(updated[key])
                    if value.shape[0] > max_len:
                        updated[key] = value[:max_len, ...]
                yield updated
            elif isinstance(example, (list, tuple)):
                updated = list(example)
                for key, max_len in len_map.items():
                    value = np.asarray(updated[key])
                    if value.shape[0] > max_len:
                        updated[key] = value[:max_len, ...]
                yield tuple(updated)
            else:
                raise ValueError("TruncateByLengthMap expects dict or tuple example.")

    return _truncate


@gin.configurable(module="trax.data")
def PadToLengthMap(len_map=None, pad_value=0):  # pylint: disable=invalid-name
    """Pads examples to max lengths per key from len_map."""
    if len_map is None:
        raise ValueError("len_map parameter should be provided.")

    def _pad_value_for(key):
        if isinstance(pad_value, dict):
            return pad_value.get(key, 0)
        return pad_value

    def _pad(generator):
        for example in generator:
            if isinstance(example, dict):
                updated = dict(example)
                for key, max_len in len_map.items():
                    value = np.asarray(updated[key])
                    pad_len = max(0, max_len - value.shape[0])
                    if pad_len:
                        updated[key] = np.pad(
                            value,
                            pad_width=(0, pad_len),
                            mode="constant",
                            constant_values=_pad_value_for(key),
                        )
                yield updated
            elif isinstance(example, (list, tuple)):
                updated = list(example)
                for key, max_len in len_map.items():
                    value = np.asarray(updated[key])
                    pad_len = max(0, max_len - value.shape[0])
                    if pad_len:
                        updated[key] = np.pad(
                            value,
                            pad_width=(0, pad_len),
                            mode="constant",
                            constant_values=_pad_value_for(key),
                        )
                yield tuple(updated)
            else:
                raise ValueError("PadToLengthMap expects dict or tuple example.")

    return _pad


@gin.configurable(module="trax.data")
def AddEOS(output_features="targets", eos=1):  # pylint: disable=invalid-name
    """Appends EOS to specified output features."""
    if not isinstance(output_features, (list, tuple)):
        output_features = [output_features]

    def _append(generator):
        for example in generator:
            if isinstance(example, dict):
                updated = dict(example)
                for key in output_features:
                    value = np.asarray(updated[key])
                    updated[key] = np.concatenate([value, [eos]], axis=0)
                yield updated
            elif isinstance(example, (list, tuple)):
                updated = list(example)
                for key in output_features:
                    value = np.asarray(updated[key])
                    updated[key] = np.concatenate([value, [eos]], axis=0)
                yield tuple(updated)
            else:
                raise ValueError("AddEOS expects dict or tuple example.")

    return _append


@gin.configurable(module="trax.data")
def BucketByLength(
    boundaries,
    batch_sizes,  # pylint: disable=invalid-name
    length_keys=None,
    length_axis=0,
    strict_pad_on_len=False,
):
    """Returns a function for bucketing inputs, see `bucket_by_length`."""
    length_keys = length_keys or [0, 1]
    # In all cases so far, we use a length function of the following form.
    def length_fn(x):
        return length_fn_(x, length_axis, length_keys)
    return lambda g: bucket_by_length(  # pylint: disable=g-long-lambda
        g, length_fn, boundaries, batch_sizes, strict_pad_on_len
    )


@gin.configurable(module="trax.data")
def BucketByLengthFromBatcher(  # pylint: disable=invalid-name
    bucket_length=32,
    batch_size_per_device=32,
    eval_batch_size=32,
    max_eval_length=None,
    buckets=None,
    buckets_include_inputs_in_length=False,
    training=True,
    length_axis=0,
    strict_pad_on_len=False,
):
    """Bucket by length using batcher-style heuristics."""
    if buckets is None:
        batch_size = batch_size_per_device if training else eval_batch_size
        boundaries, batch_sizes = buckets_for_length(
            bucket_length, batch_size, max_eval_length, n_devices=1, training=training
        )
    else:
        boundaries, batch_sizes = buckets

    length_keys = [0, 1] if buckets_include_inputs_in_length else [1]
    return BucketByLength(
        boundaries=boundaries,
        batch_sizes=batch_sizes,
        length_keys=length_keys,
        length_axis=length_axis,
        strict_pad_on_len=strict_pad_on_len,
    )


@gin.configurable(module="trax.data")
def MLM(
    vocab_size=None,  # pylint:disable=invalid-name
    max_length=None,
    noise_density=0.15,
    mean_noise_span_length=3.0,
):
    """Pipeline that just does MLM."""
    return Serial(
        # Generate sequential chunks.
        generate_sequential_chunks(max_length=max_length),
        # Generate mask and chunk.
        generate_random_noise_mask(
            noise_density=noise_density, mean_noise_span_length=mean_noise_span_length
        ),
        # Consume mask and chunk to give (input, targets).
        consume_noise_mask(vocab_size=vocab_size),
    )


@gin.configurable(module="trax.data")
def PrefixLM(input_length=128, output_length=512):  # pylint:disable=invalid-name
    """Chunks examples to make inputs/outputs of specified lengths."""

    def _f(generator):
        for example in generator:
            n_tokens = len(example)
            # Iterate:
            # |--------|<---- input_length ---->|<- output_length ->|--------------|
            # ^        ^                        ^                   ^
            # |        |                        |                   |
            # 0        input_begin_idx          input_end_idx       output_end_idx
            input_begin_idx = 0
            # While you can make an input batch, keep going.
            while input_begin_idx + input_length < n_tokens:
                input_end_idx = input_begin_idx + input_length
                output_end_idx = min(input_end_idx + output_length, n_tokens)
                yield (
                    example[input_begin_idx:input_end_idx],
                    example[input_end_idx:output_end_idx],
                )
                # Update the indices.
                input_begin_idx = output_end_idx

    return _f


@gin.configurable(module="trax.data")
def ConcatenateToLMInput(pad_to_length=None):  # pylint: disable=invalid-name
    """Prepares the input needed for training of Language Models.

    Each example needs to contain two elements (input and target).
    Input is concatenated to target and, if pad_to_length is given, padded to
    length provided.
    The loss_weights indicates only the target, without input nor padding.

    Args:
      pad_to_length: int, total length of padding of input and target arrays.
    Returns:
      Function to return input for a LM.
    """

    @debug_data_pipeline.debug_pipeline
    def _concatenate_to_lm_input(generator):
        for example in generator:
            if isinstance(example, (list, tuple)) and (len(example) == 2):
                concatenated = np.concatenate((example[0], example[1]), axis=-1)
                loss_weights = np.concatenate(
                    (np.zeros_like(example[0]), np.ones_like(example[1]))
                )
                if pad_to_length is not None:
                    padding_len = pad_to_length - (
                        example[0].shape[0] + example[1].shape[0]
                    )
                    if padding_len < 0:
                        raise ValueError(
                            "Example lengths "
                            f"({example[0].shape[0]}, {example[1].shape[0]}) "
                            f"longer than pad_to_length ({pad_to_length})."
                        )
                    loss_weights = np.pad(loss_weights, (0, padding_len), "constant")
                    concatenated = np.pad(concatenated, (0, padding_len), "constant")
                output = (concatenated, concatenated, loss_weights)
            elif isinstance(example, (list, tuple)) and (len(example) == 1):
                # Make x into (x, x)
                output = (example[0], example[0])
            elif isinstance(example, np.ndarray):
                # Make x into (x, x)
                output = (example, example)
            else:
                raise ValueError(f"Unknown input to ConcatenateToLMInput: {example}")
            yield output

    return _concatenate_to_lm_input


@gin.configurable(module="trax.data")
def CastTo(
    dtype=np.int32,
    indices=(
        0,
        1,
    ),
    debug=False,
):  # pylint: disable=invalid-name
    """Casts the given indices to the given dtype."""

    def _cast_fn(generator):
        debug_count = 0
        for example in generator:
            debug_count += 1
            assert isinstance(example, tuple)
            example = list(example)
            dtype_mismatch = False
            original_index_and_dtype = []
            for i in range(len(example)):
                if i not in indices:
                    continue
                original_type = example[i].dtype
                if original_type != dtype:
                    if not (original_type == np.int64 and dtype == np.int32):
                        # Downcasting from np.int64 to np.int32 is OK
                        original_index_and_dtype.append((i, original_type))
                    example[i] = example[i].astype(dtype)
                    dtype_mismatch = True
            if debug and dtype_mismatch and original_index_and_dtype:
                trax_logging.info(
                    "dtype mismatch in example[%d] = %r was earlier: %r",
                    debug_count,
                    example,
                    original_index_and_dtype,
                )
            yield tuple(example)

    return _cast_fn


@gin.configurable(module="trax.data")
def AppendValue(val=None):  # pylint: disable=invalid-name
    """Appends values provided in 'val` to inputs.

    val are keyed by example keys, its values contain appended tensors.

    Args:
      val: dict of int to tensors. Specific keys get the tensors specified in
        values appended.
    Returns:
      Funtion to append tensors to examples.
    """

    @debug_data_pipeline.debug_pipeline
    def _append_value(generator):
        for example in generator:
            if isinstance(example, tuple):
                example = list(example)
                if val is not None:
                    for key, value in val.items():
                        example[key] = np.append(example[key], value, -1)
                output = tuple(example)
            else:
                if not isinstance(example, np.ndarray):
                    raise ValueError(f"example isn't nparray, but should be: {example}")
                output = np.append(example, val[0])
            yield output

    return _append_value


@gin.configurable(module="trax.data")
def AddLossWeights(id_to_mask=None):  # pylint: disable=invalid-name
    """Returns a function to add loss weights; see `add_loss_weights`."""
    return lambda g: add_loss_weights(g, id_to_mask=id_to_mask)


@gin.configurable(module="trax.data")
def UnBatch():  # pylint: disable=invalid-name
    """Returns a function which unbatches."""

    def _unbatch(generator):
        for batched_example in generator:
            # batched_example is usually like:
            # (batched_inputs, batched_outputs) or
            # (batched_inputs, batched_outputs, batched_weights)
            assert isinstance(batched_example, tuple)
            # assert all lengths are the same.
            batch_sizes = list(
                set(map(lambda example: example.shape[0], batched_example))
            )
            assert len(batch_sizes) == 1
            # Now unbatch examples.
            for example_idx in range(batch_sizes[0]):
                yield tuple(
                    map(lambda x: x[example_idx], batched_example)
                )  # pylint: disable=cell-var-from-loop

    return _unbatch


@gin.configurable(module="trax.data")
def Prefetch(n_prefetch=2):  # pylint: disable=invalid-name
    """Pre-fetches a number of examples from generator in a separate process."""

    def prefetch(generator):
        in_q, out_q = mp.Queue(), mp.Queue()
        p = mp.Process(target=generator_process, args=(generator, in_q, out_q))
        for _ in range(n_prefetch):
            in_q.put(None)
        p.start()
        while True:
            yield out_q.get()
            in_q.put(None)

    return prefetch


@gin.configurable(module="trax.data")
def UniformlySeek(
    name=None, host_id=None, n_hosts=None, dataset_size=None
):  # pylint: disable=invalid-name
    """Sets each host at (dataset_size/n_hosts)-th of the dataset."""
    if not dataset_size:
        dataset_size = 2**18  # 512 * 512
        trax_logging.error(
            "No dataset size given to Uniformly seek, assuming: %d", dataset_size
        )
    assert name
    host_id = jax.process_index() if host_id is None else host_id
    n_hosts = n_hosts or jax.host_count()
    each_host = int(dataset_size / n_hosts)

    def _f(generator):
        # Each host seeks to the appropriate point in the dataset.
        num_to_seek = int(host_id * each_host)
        start_time = time.time()
        trax_logging.info(
            "Dataset[%s] host_id[%d] is seeking to position[%d]",
            name,
            host_id,
            num_to_seek,
        )
        for _ in range(num_to_seek):
            next(generator)
        trax_logging.info(
            "Dataset[%s] host_id[%d] reached position[%d]. " "Time taken [%s] seconds",
            name,
            host_id,
            num_to_seek,
            time.time() - start_time,
        )
        for example in generator:
            yield example

    return _f


@gin.configurable(module="trax.data")
def CountAndSkip(name):  # pylint: disable=invalid-name
    """Returns a function that counts and skips examples (see above)."""
    return lambda g: count_and_skip(g, name)


@gin.configurable(module="trax.data")
def Log(n_steps_per_example=1, only_shapes=True):  # pylint: disable=invalid-name
    """Creates a logging component of the input pipeline."""

    def log(stream):
        counter = 0
        for example in stream:
            item_to_log = example
            if only_shapes:
                item_to_log = fastmath.nested_map(shapes.signature, example)
            if counter % n_steps_per_example == 0:
                trax_logging.info(str(item_to_log), stdout=True)
            counter += 1
            yield example

    return log


@gin.configurable(module="trax.data")
def ConvertToUnicode(keys=None):  # pylint: disable=invalid-name
    """Converts to Unicode UTF-8 elements of an example.

    Useful for when TFDS outputs byte arrays. All the errors of the conversion
    are ignored.

    Args:
      keys: tuple/list of example dimensions to convert.

    Returns:
      Function converting chosen elements of an example to UTF-8.
    """

    @debug_data_pipeline.debug_pipeline
    def _convert_to_unicode_str(stream):
        for example in stream:
            if isinstance(example, (list, tuple)):
                new_example = []
                for i, x in enumerate(example):
                    if keys is None or i in keys:
                        new_example.append(to_unicode(x))
                    else:
                        new_example.append(x)
                output = tuple(new_example)
                yield output
            elif isinstance(example, dict):
                new_example = {}
                for k in example:
                    if keys is None or k in keys:
                        new_example[k] = to_unicode(example[k])
                    else:
                        new_example[k] = example[k]
                yield new_example
            else:
                output = to_unicode(example)
                yield output

    return _convert_to_unicode_str


def _text_to_str(text):
    if isinstance(text, np.ndarray):
        if text.shape == ():
            text = text.item()
        else:
            return text
    if isinstance(text, np.bytes_):
        text = text.tobytes()
    if isinstance(text, bytes):
        return to_unicode(text)
    if isinstance(text, str):
        return text
    return str(text)


@gin.configurable(module="trax.data")
def SentencePieceTokenize(  # pylint: disable=invalid-name
    spm_path=gin.REQUIRED, extra_ids=0
):
    """Tokenizes a stream of text using SentencePiece."""
    tokenizer = SentencePieceEncoder(spm_path, extra_ids=extra_ids)

    def _tokenize(stream):
        for example in stream:
            if isinstance(example, dict):
                text = example.get("text", example.get("inputs", example.get("targets")))
            elif isinstance(example, (list, tuple)) and example:
                text = example[0]
            else:
                text = example
            text = _text_to_str(text)
            if isinstance(text, np.ndarray):
                tokens = np.asarray(text, dtype=np.int64)
            else:
                tokens = np.array(tokenizer.encode(text), dtype=np.int64)
            yield tokens

    return _tokenize


@gin.configurable(module="trax.data")
def Rekey(key_map=None):  # pylint: disable=invalid-name
    """Replaces example keys according to the mapping in `key_map`."""
    if not key_map:
        return lambda g: g

    def _rekey(stream):
        for example in stream:
            if not isinstance(example, dict):
                raise ValueError("Rekey expects dict examples.")
            yield {
                new_key: example[old_key] if old_key else ""
                for new_key, old_key in key_map.items()
            }

    return _rekey


@gin.configurable(module="trax.data")
def SentencePieceTokenizePairs(  # pylint: disable=invalid-name
    spm_path=gin.REQUIRED,
    input_key="inputs",
    target_key="targets",
    copy_pretokenized=False,
):
    """Tokenizes `inputs` and `targets` fields using SentencePiece."""
    tokenizer = SentencePieceEncoder(spm_path)

    def _tokenize(stream):
        for example in stream:
            if not isinstance(example, dict):
                raise ValueError("SentencePieceTokenizePairs expects dict examples.")
            inputs = _text_to_str(example.get(input_key, example.get("targets")))
            targets = _text_to_str(example.get(target_key, example.get("targets")))
            tokenized_inputs = np.array(tokenizer.encode(inputs), dtype=np.int64)
            tokenized_targets = np.array(tokenizer.encode(targets), dtype=np.int64)
            output = {
                input_key: tokenized_inputs,
                target_key: tokenized_targets,
            }
            if copy_pretokenized:
                output[f"{input_key}_pretokenized"] = inputs
                output[f"{target_key}_pretokenized"] = targets
            yield output

    return _tokenize


@gin.configurable(module="trax.data")
def DictToTuple(keys=("inputs", "targets")):  # pylint: disable=invalid-name
    """Converts dict examples to tuples based on `keys`."""
    def _convert(stream):
        for example in stream:
            if not isinstance(example, dict):
                raise ValueError("DictToTuple expects dict examples.")
            yield tuple(example[key] for key in keys)

    return _convert


@gin.configurable(module="trax.data")
def GenericTextPreprocess(  # pylint: disable=invalid-name
    spm_path=gin.REQUIRED,
    text_preprocess_fns=None,
    token_preprocess_fns=None,
    copy_pretokenized=False,
):
    """Serial-friendly text preprocessing with SentencePiece tokenization."""
    steps = []
    if text_preprocess_fns:
        steps.extend(text_preprocess_fns)
    steps.append(
        SentencePieceTokenizePairs(
            spm_path=spm_path, copy_pretokenized=copy_pretokenized
        )
    )
    if token_preprocess_fns:
        steps.extend(token_preprocess_fns)
    steps.append(DictToTuple())
    return Serial(*steps)


def _pad_punctuation_py(text):
    import re
    import string

    pattern = r"([{}])".format(re.escape(string.punctuation))
    text = re.sub(pattern, r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


@gin.configurable(module="trax.data")
def SquadTextPreprocess(include_context=True):  # pylint: disable=invalid-name
    """Converts SQuAD examples to dicts with `inputs` and `targets`."""

    def _squad(stream):
        for example in stream:
            if not isinstance(example, dict):
                raise ValueError("SquadTextPreprocess expects dict examples.")
            question = _pad_punctuation_py(_text_to_str(example.get("question", "")))
            context = _pad_punctuation_py(_text_to_str(example.get("context", "")))
            answers = example.get("answers", {})
            if isinstance(answers, dict):
                answer_list = answers.get("text", [])
            else:
                answer_list = answers
            answer = ""
            if isinstance(answer_list, (list, tuple)) and answer_list:
                answer = _pad_punctuation_py(_text_to_str(answer_list[0]))
            elif isinstance(answer_list, str):
                answer = _pad_punctuation_py(_text_to_str(answer_list))
            if include_context:
                inputs = f"question: {question} context: {context}".strip()
            else:
                inputs = f"squad trivia question: {question}".strip()
            yield {
                "inputs": inputs,
                "targets": answer,
                "id": example.get("id", ""),
                "context": context,
                "question": question,
                "answers": answer_list,
            }

    return _squad


@gin.configurable(module="trax.data")
def ClassificationVector(vocab_size=8192):  # pylint: disable=invalid-name
    """Returns a function to convert token sequences to one-hot vectors."""
    return lambda g: classification_vector(g, vocab_size=vocab_size)


#
# Generator based pure functions
#
def batch(generator, batch_size):
    """Batch and pad generator as in tf.data.Dataset.padded_batch."""
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, but is {batch_size}.")
    buf = []
    i = 0
    for example in generator:
        buf.append(example)  # Examples are tuples of tensors.
        if len(buf) == batch_size:
            # buf is a list of tuples, e.g., [(in1, tgt1), (in2, tgt2), (in3, tgt3)]
            # batch is a tuple of arrays: ([in1, in2, in3], [tgt1, tgt2, tgt3])
            try:
                batched_example = tuple(
                    pad_to_max_dims([np.asarray(tensor) for tensor in x])
                    for x in zip(*buf)
                )
            except ValueError as e:
                for j in range(len(buf)):
                    trax_logging.error(
                        "Batch[%d][%d] input shape: %r output shape: %r",
                        i,
                        j,
                        buf[j][0].shape,
                        buf[j][1].shape,
                    )
                for j in range(len(buf)):
                    trax_logging.error("Batch[%d][%d] input: %r", i, j, buf[j][0])
                    trax_logging.error("Batch[%d][%d] output: %r", i, j, buf[j][1])
                raise e
            i += 1
            yield batched_example
            buf = []


def bucket_by_length(
    generator, length_fn, boundaries, batch_sizes, strict_pad_on_len=False
):
    """Bucket by length, like tf.data.experimental.bucket_by_sequence_length.

    This function draws examples from the provided `generator` and puts an
    example into a bucket depending on `l = length_fn(example)`. Which bucket
    is used depends on between which `boundaries` is l. When a bucket reaches
    its batch size, as specified by `batch_sizes`, generates a batch of
    padded examples from this bucket.

    Args:
      generator: python generator to draw data from.
      length_fn: a function taking the example and returning the length.
      boundaries: a list of bucket boundaries.
      batch_sizes: a list of batch sizes.
      strict_pad_on_len: bool; if true we pad on the length dimension, dim[0]
        strictly as a multiple of boundary.

    Yields:
      An input batch, which comes from one of the buckets.
    """
    buckets = [[] for _ in range(len(batch_sizes))]
    boundaries = boundaries + [math.inf]  # Max boundary is unlimited.
    for example in generator:
        length = length_fn(example)
        # `bucket_idx` will always be < len(boundaries), since boundaries is right
        # padded by `math.inf`.
        bucket_idx = min([i for i, b in enumerate(boundaries) if length <= b])
        buckets[bucket_idx].append(example)
        if len(buckets[bucket_idx]) == batch_sizes[bucket_idx]:
            batched = zip(*buckets[bucket_idx])
            boundary = boundaries[bucket_idx]
            boundary = None if boundary == math.inf else boundary
            padded_batch = tuple(
                pad_to_max_dims(x, boundary, strict_pad_on_len) for x in batched
            )
            yield padded_batch
            buckets[bucket_idx] = []


@debug_data_pipeline.debug_pipeline
def add_loss_weights(generator, id_to_mask=None):
    """Add weights to inputs without weights and masks by id if requested.

    The generator stream is augmented in the following way:

    - If the stream consists of pairs `(inputs, targets)`, a loss mask is added
      that is creates as a tensor of ones of the same shape as targets.
    - If `id_to_mask` is not `None`, and the stream (after the previous point)
      has triples `(inputs, targets, weights)`, the weights are multiplied by a
      0/1 mask that is 0 iff targets is equal to `id_to_mask` (1 otherwise).

    Args:
      generator: Stream of tuples.
      id_to_mask: If not None, int-valued id that represents padding, as opposed
          to true target IDs.

    Yields:
      Examples from the augmented stream.
    """
    for example in generator:
        if len(example) > 3 or len(example) < 2:
            assert id_to_mask is None, "Cannot automatically mask this stream."
            yield example
        else:
            if len(example) == 2:
                weights = np.ones_like(example[1]).astype(np.float32)
            else:
                weights = example[2].astype(np.float32)
            mask = 1.0 - np.equal(example[1], id_to_mask).astype(np.float32)
            weights *= mask
            output = (example[0], example[1], weights)
            yield output


data_counters = {}  # Used by {load,save}_data_counters and count_and_skip


def count_and_skip(generator, name):
    """Count the number of items in the generator, skip already counted ones.

    This function counts the number of processed examples and puts it into
    the global variable `counters`. This variable can be saved and restored,
    and if restored, this function will skip examples until the restored counter
    is reached. When the data generator is deterministic, this allows to restore
    the data reading process from a checkpoint.

    Args:
      generator: generator for examples in the dataset.
      name: string, a unique id that we use to count the examples

    Yields:
      The examples from generator but first skip the number specified in the
      global variable counters[name] and next increment this variable every
      time a new example appears.
    """
    global data_counters
    local_counter = 0
    for example in generator:
        local_counter += 1
        # This check must be inside the loop due to asynchronous initializations.
        if name not in data_counters:
            data_counters[name] = 0
        if local_counter > data_counters[name]:
            data_counters[name] += 1
            yield example


def generator_process(generator, in_q, out_q):
    for example in generator:
        in_q.get()
        out_q.put(example)


#
# Helper functions
#
def shuffle(samples, queue_size):
    """Shuffles a sample stream using a random-out next-in queue of given size.

    Args:
      samples: Stream of samples for eventual use as training data or eval data.
      queue_size: Minimum number of samples within which the streamed shuffling
          takes place.

    Yields:
      Shuffled stream of samples, ready for further processing, e.g., grouping
      into batches.
    """
    if queue_size < 1:
        raise ValueError(f"Arg queue_size ({queue_size}) is less than 1.")
    if queue_size == 1:
        trax_logging.warning("Queue size of 1 results in no shuffling.")
    queue = []
    try:
        # Prep: fill the queue.
        for _ in range(queue_size):
            queue.append(next(samples))

        # Core streaming shuffle: yield sample from random location in queue, then
        # fill that location with new sample from input stream.
        for sample in samples:
            i = np.random.randint(queue_size)
            yield queue[i]
            queue[i] = sample
    except StopIteration:
        # Only get here if the initial queue fill fails.
        trax_logging.warning(
            "Not enough samples (%d) to fill initial queue (size %d).",
            len(queue),
            queue_size,
        )

    # No new samples coming in; shuffle and drain the queue.
    np.random.shuffle(queue)
    for sample in queue:
        yield sample


def pad_tf_tensors(tensors, boundary=None, strict_pad_on_len=False):
    """
    Pad RaggedTensors to a consistent size with advanced padding options.

    Args:
      tensors: A list of TensorFlow RaggedTensors to pad
      boundary: Optional boundary for padding
      strict_pad_on_len: If True, pad strictly to boundary multiples

    Returns:
      A padded batch of tensors
    """
    # Ensure inputs are RaggedTensors or Tensor
    if not all(isinstance(a, (tf.RaggedTensor, tf.Tensor)) for a in tensors):
        raise ValueError("All input tensors must be RaggedTensors or Tensor")

    # Get the number of dimensions
    dim = tensors[0].shape.rank

    # Handle boundary input
    if boundary is not None:
        if not isinstance(boundary, (list, tuple)):
            boundary = [boundary] * dim

        if len(boundary) != dim:
            raise ValueError(
                f"Length of boundary ({len(boundary)}) must match tensor dimensions ({dim})"
            )
    else:
        boundary = [None] * dim

    # Extract lengths for each dimension
    def get_tensor_lengths(tensors, dim_index):
        """Safely extract lengths for a given dimension."""
        lengths = []
        for t in tensors:
            # For the first dimension (row lengths)
            if dim_index == 0:
                lengths.append(t.nrows())
            # For subsequent dimensions, get the row lengths
            else:
                # Flatten and get max length of inner dimension
                flat_values = t.flat_values
                # Handle multi-dimensional ragged tensors
                if dim_index < flat_values.shape.ndims:
                    flat_length = flat_values.shape[dim_index - 1]
                    lengths.append(flat_length)
                else:
                    lengths.append(0)
        return lengths

    # Compute padding lengths
    max_len_to_pad = []
    padding_needed = False

    for i in range(dim):
        lengths = get_tensor_lengths(tensors, i)

        # Determine max length
        max_len = max(lengths)
        min_len = min(lengths)

        # Handle boundary and strict padding
        cur_boundary = boundary[i]

        if cur_boundary is None:
            # No boundary specified, use max length
            max_len_pad = max_len
        elif strict_pad_on_len:
            # Strictly pad to boundary multiples
            max_len_pad = math.ceil(max_len / cur_boundary) * cur_boundary
        else:
            # Use boundary with intelligent power-of-2 adjustment
            if max_len <= 0:
                max_len_pad = 0
            else:
                cur_boundary = max(max_len, cur_boundary)
                if 2 * max_len < cur_boundary:
                    max_len_pad = 2 ** int(np.ceil(np.log2(max_len)))
                else:
                    max_len_pad = cur_boundary

        max_len_to_pad.append(max_len_pad)

        # Check if padding is needed
        if max_len_pad != max_len:
            padding_needed = True

    # If no padding is needed, stack the tensors
    if not padding_needed:
        return tf.stack(tensors)

    # Pad each tensor
    padded_tensors = []
    for t in tensors:
        # Determine padding for each dimension
        padding_spec = []
        for i, max_pad in enumerate(max_len_to_pad):
            if i == 0:
                # Pad rows
                row_padding = max_pad - t.nrows()
                padding_spec.append([0, row_padding])
            else:
                # Pad inner dimensions
                try:
                    flat_values = t.flat_values
                    if i < flat_values.shape.ndims:
                        dim_len = flat_values.shape[i - 1]
                        padding_to_add = max_pad - dim_len
                        padding_spec.append([0, padding_to_add])
                    else:
                        padding_spec.append([0, 0])
                except Exception:
                    padding_spec.append([0, 0])

        # Apply padding
        padded_t = tf.pad_to_max_length(t, max_len_to_pad[0], constant_values=0)
        padded_tensors.append(padded_t)

    return tf.stack(padded_tensors)


def pad_np_tensors(tensors, boundary=None, strict_pad_on_len=False):
    """Pad a tuple of tensors to a joint dimension and return their batch.

    For example, a pair of tensors of shape (2, 10) and (3, 9) will be padded
    to (3, 10) both and the returned tensor will have shape (2, 3, 10).

    When boundary is specified, we try to pad all unknown dimensions to boundary
    if possible, which can help reduce the number of different shapes occurring
    in the tensors and speed up XLA compilation. So, for example, a pair of
    tensors of shapes (8, 10), (8, 9) with boundary=12 will be padded to (8, 12).

    One special case occurs when boundary is much higher than the padding length
    that we'd use without boundary. For example, tensors (2, 10) and (3, 9) with
    boundary=12 could end up padded to (12, 12), but this is very wasteful in
    the first dimension. In that case, we will use the closest power-of-2 instead
    of the boundary, so the we will end up padding to (4, 12) instead of (12, 12).

    Args:
      tensors: a tuple or list of tensors to pad
      boundary: int or None; if given, expand the padded dimensions to this size
      strict_pad_on_len: bool; if true we pad on the length dimension, dim[0]
        strictly as a multiple of boundary.

    Returns:
      a tensor, the tensors padded together
    """
    # TODO(afrozm): Unify this later.
    if not all(isinstance(a, np.ndarray) for a in tensors):
        raise ValueError("All input tensors must be numpuy array")

    if (boundary is not None) and (
        strict_pad_on_len or isinstance(boundary, (list, tuple))
    ):
        ndim = tensors[0].ndim
        if not isinstance(boundary, (list, tuple)):
            boundary = [boundary] * ndim

        if ndim != len(boundary):
            raise ValueError(
                f"ndim != len(boundary) - "
                f"ndim({ndim}) vs boundary({boundary}) "
                f"len(boundary) = {len(boundary)}."
            )

        max_len_per_dim = [0] * ndim
        for tensor in tensors:
            max_len_per_dim = [max(e, s) for e, s in zip(tensor.shape, max_len_per_dim)]

        # Round everything up to a multiple of boundary in the respective dimension.
        len_per_dim = [
            max_len_per_dim[i] if not b else b * math.ceil(max_len_per_dim[i] / b)
            for i, b in enumerate(boundary)
        ]

        padded_tensors = [
            np.pad(
                t,
                [(0, len_per_dim[i] - t.shape[i]) for i in range(ndim)],
                mode="constant",
                constant_values=t.dtype.type(0),
            )
            for t in tensors
        ]

        return np.stack(padded_tensors)

    max_len_to_pad = []
    padding_needed = False
    dim = len(tensors[0].shape)
    for i in range(dim):
        max_len = max([t.shape[i] for t in tensors])
        min_len = min([t.shape[i] for t in tensors])
        if max_len == min_len and max_len == boundary:  # No padding needed.
            max_len_to_pad.append(max_len)
        elif boundary is None:
            max_len_to_pad.append(max_len)
            padding_needed = True
        else:
            padding_needed = True
            cur_boundary = max(max_len, boundary)
            if 2 * max_len < cur_boundary:
                cur_boundary = 2 ** int(np.ceil(np.log2(max_len)))
            max_len_to_pad.append(cur_boundary)
    if not padding_needed:
        return np.stack(tensors)
    padded_tensors = []
    for t in tensors:
        pad_widths = [(0, max_len_to_pad[i] - t.shape[i]) for i in range(dim)]
        padded_t = np.pad(
            t, pad_widths, mode="constant", constant_values=t.dtype.type(0)
        )
        padded_tensors.append(padded_t)
    return np.stack(padded_tensors)


def pad_jax_arrays(
    arrays: Sequence[jax.Array],
    boundary: Optional[Union[int, Sequence[int]]] = None,
    strict_pad_on_len: bool = False,
) -> jax.Array:
    """Pad a sequence of JAX Arrays to a joint dimension and return their batch.

    For example, a pair of arrays of shape (2, 10) and (3, 9) will be padded
    to (3, 10) both and the returned array will have shape (2, 3, 10).

    When boundary is specified, we try to pad all unknown dimensions to boundary
    if possible, which can help reduce the number of different shapes occurring
    in the arrays and speed up XLA compilation. So, for example, a pair of
    arrays of shapes (8, 10), (8, 9) with boundary=12 will be padded to (8, 12).

    One special case occurs when boundary is much higher than the padding length
    that we'd use without boundary. For example, arrays (2, 10) and (3, 9) with
    boundary=12 could end up padded to (12, 12), but this is very wasteful in
    the first dimension. In that case, we will use the closest power-of-2 instead
    of the boundary, so we will end up padding to (4, 12) instead of (12, 12).

    Args:
      arrays: a sequence of JAX Arrays to pad
      boundary: int or None; if given, expand the padded dimensions to this size
        or can be a sequence matching the number of dimensions
      strict_pad_on_len: bool; if true we pad on the length dimension, dim[0]
        strictly as a multiple of boundary.

    Returns:
      a JAX Array, the arrays padded together
    """
    # Ensure inputs are JAX Arrays
    if not all(isinstance(a, jax.Array) for a in arrays):
        raise ValueError("All inputs must be JAX Arrays")

    # Handle case with list/tuple boundary or strict padding
    if (boundary is not None) and (
        strict_pad_on_len or isinstance(boundary, (list, tuple))
    ):
        ndim = arrays[0].ndim
        if not isinstance(boundary, (list, tuple)):
            boundary = [boundary] * ndim

        if ndim != len(boundary):
            raise ValueError(
                f"ndim != len(boundary) - "
                f"ndim({ndim}) vs boundary({boundary}) "
                f"len(boundary) = {len(boundary)}."
            )

        # Find maximum length per dimension
        max_len_per_dim = [0] * ndim
        for array in arrays:
            max_len_per_dim = [max(e, s) for e, s in zip(array.shape, max_len_per_dim)]

        # Round everything up to a multiple of boundary in the respective dimension
        len_per_dim = [
            max_len_per_dim[i] if not b else b * math.ceil(max_len_per_dim[i] / b)
            for i, b in enumerate(boundary)
        ]

        # Pad each array to the target dimensions
        padded_arrays = [
            jnp.pad(
                a,
                [(0, len_per_dim[i] - a.shape[i]) for i in range(ndim)],
                mode="constant",
                constant_values=a.dtype.type(0),
            )
            for a in arrays
        ]

        return jnp.stack(padded_arrays)

    # Handle the simpler case (similar to pad_np_tensors second part)
    max_len_to_pad = []
    padding_needed = False
    dim = arrays[0].ndim

    for i in range(dim):
        max_len = max([a.shape[i] for a in arrays])
        min_len = min([a.shape[i] for a in arrays])

        if max_len == min_len and max_len == boundary:  # No padding needed
            max_len_to_pad.append(max_len)
        elif boundary is None:
            max_len_to_pad.append(max_len)
            padding_needed = True
        else:
            padding_needed = True
            cur_boundary = max(max_len, boundary)
            if 2 * max_len < cur_boundary:
                cur_boundary = 2 ** int(jnp.ceil(jnp.log2(max_len)))
            max_len_to_pad.append(cur_boundary)

    if not padding_needed:
        return jnp.stack(arrays)

    padded_arrays = []
    for a in arrays:
        pad_widths = [(0, max_len_to_pad[i] - a.shape[i]) for i in range(dim)]
        padded_a = jnp.pad(
            a, pad_widths, mode="constant", constant_values=a.dtype.type(0)
        )
        padded_arrays.append(padded_a)

    return jnp.stack(padded_arrays)


def pad_to_max_dims(tensors, boundary=None, strict_pad_on_len=False):
    """
    Unified padding function. Depending on the type of input tensors, it either applies
    dense padding (using NumPy) or uses TensorFlow operations for RaggedTensors.

    Args:
      tensors: A list or tuple of tensors to pad. They must be either all np.ndarray or all tf.RaggedTensor.
      boundary: Optional boundary for padding.
      strict_pad_on_len: If True, pad strictly to boundary multiples.

    Returns:
      A batched tensor with consistent dimensions.
    """
    if all(isinstance(t, tf.RaggedTensor) or isinstance(t, tf.Tensor) for t in tensors):
        return pad_tf_tensors(tensors, boundary, strict_pad_on_len)
    elif all(isinstance(t, np.ndarray) for t in tensors):
        return pad_np_tensors(tensors, boundary, strict_pad_on_len)
    elif all(isinstance(t, jax.Array) for t in tensors):
        return pad_jax_arrays(tensors, boundary, strict_pad_on_len)
    else:
        raise ValueError(
            "Mixed tensor types not supported. All tensors must be either tf.RaggedTensor, tf.Tensor, jax Array or np.ndarray."
        )


@gin.configurable(module="trax.data")
def generate_random_noise_mask(
    noise_density=0.15, mean_noise_span_length=3.0, seed1=None, seed2=None
):
    """Returns a function that generates a random noise mask."""

    def _f(generator):
        for example in generator:
            length = len(example)
            noise_mask = random_spans_noise_mask(
                length,
                noise_density=noise_density,
                mean_noise_span_length=mean_noise_span_length,
                seed1=seed1,
                seed2=seed2,
                example=example,
            )
            yield (example, noise_mask)

    return _f


@gin.configurable(module="trax.data")
def consume_noise_mask(vocab_size=32100):
    """Consumes (tokens, noise mask) and returns (inputs, targets)."""

    def _noise_span_to_unique_sentinel(tokens, noise_mask):
        prev_token_is_noise = np.pad(
            noise_mask[:-1], [1, 0], mode="constant", constant_values=False
        )
        first_noise_tokens = np.logical_and(
            noise_mask, np.logical_not(prev_token_is_noise)
        )
        subsequent_noise_tokens = np.logical_and(noise_mask, prev_token_is_noise)
        sentinel = vocab_size - np.cumsum(first_noise_tokens)
        tokens = np.where(first_noise_tokens, sentinel, tokens)
        return tokens[np.logical_not(subsequent_noise_tokens)]

    def _f(generator):
        for tokens, noise_mask in generator:
            # Returns inputs and targets.
            yield (
                _noise_span_to_unique_sentinel(tokens, noise_mask),
                _noise_span_to_unique_sentinel(tokens, np.logical_not(noise_mask)),
            )

    return _f


@gin.configurable(module="trax.data")
def generate_sequential_chunks(max_length=None):
    """Returns a function that generates chunks of atmost max_length length."""

    def _f(generator):
        for example in generator:
            n_tokens = len(example)
            if n_tokens <= max_length:
                yield example
            else:
                n_segments = int(math.ceil(float(n_tokens) / float(max_length)))
                for i in range(n_segments):
                    start = max_length * i
                    end = min(start + max_length, n_tokens)
                    yield example[start:end]

    return _f


@gin.configurable(module="trax.data")
def addition_input_stream(
    vocab_size=gin.REQUIRED,
    batch_size=gin.REQUIRED,
    min_length=gin.REQUIRED,
    max_length=gin.REQUIRED,
    pad_to_multiple=32,
    encdec=False,
):
    """Data stream for the add problem: <S>x+y<S>(x+y).

    Args:
      vocab_size: how many symbols to use.
      batch_size: how large are the batches.
      min_length: minimal length of w.
      max_length: maximal length of w.
      pad_to_multiple: int, pad length to be multiple of this number.
      encdec: bool, if True return encoder-decoder style inputs (default: False)

    Returns:
      python generator of tuples of data examples
    """
    base = vocab_size - 3  # We use 0 to pad, training+1 as "+" and training+2 as "<S>".

    def single_example(max_length, min_length):
        """Generate a stream of random mini-batches."""
        add_len = (min_length - 1) // 2
        l1 = np.random.randint((max_length - add_len + 1) // 2) + add_len
        l2 = np.random.randint(max_length - l1 - 1) + 1
        n1 = random_number_lower_endian(l1, base)
        n2 = random_number_lower_endian(l2, base)
        result = lower_endian_to_number(n1, base) + lower_endian_to_number(n2, base)
        inp = n1 + [base] + n2
        tgt = number_to_lower_endian(result, base)
        if encdec:
            x = [i + 1 for i in inp]
            y = [i + 1 for i in tgt]
            weights = [1] * len(tgt)
            candidate_example = (np.array(x), np.array(y), np.array(weights))
            if any(len(sample) > max_length for sample in candidate_example):
                # sample too long, try again
                return single_example(max_length, min_length)
            return (np.array(x), np.array(y), np.array(weights))
        else:
            x = [base + 2] + [i + 1 for i in inp] + [base + 2] + [i + 1 for i in tgt]
            weights = ([0] * (len(inp) + 2)) + ([1] * len(tgt))
            return (np.array(x), np.array(x), np.array(weights))

    def batches(max_length, min_length):
        """Batches of examples."""
        if max_length < 3 or min_length < 3:
            raise ValueError("Maximum/minimum length must be at least 3.")
        while True:
            ex = [single_example(max_length, min_length) for _ in range(batch_size)]
            padded_batch = [
                pad_to_max_dims(x, boundary=pad_to_multiple, strict_pad_on_len=True)
                for x in zip(*ex)
            ]
            yield tuple(padded_batch)

    return batches(max_length, min_length)


@gin.configurable(module="trax.data")
def make_additional_stream(stream=gin.REQUIRED):
    """Create a stream mostly for use in gin configs for additional tasks."""
    return Serial(stream)()


@gin.configurable(module="trax.data")
def make_parallel_stream(streams=gin.REQUIRED, counters=None):
    """Create a parallel stream for use in gin configs for additional tasks."""
    return Parallel(streams, counters=counters)()


@debug_data_pipeline.debug_pipeline
def classification_vector(generator, vocab_size=8192):
    """Convert token sequences to classification vectors.

    The generator stream is transformed by replacing token sequences with
    vectors where each position contains the token ID if that token appears
    in the text, otherwise 0.

    Args:
      generator: Stream of tuples where the first element is a token sequence.
      vocab_size: Size of the vocabulary (defines length of the vector).

    Yields:
      Examples with token sequences converted to classification vectors.
    """
    for example in generator:
        tokens = example[0]

        # Create a zero vector of vocab_size length
        class_vector = np.zeros(vocab_size, dtype=np.int32)

        # Set token ID at positions corresponding to tokens
        for token_id in tokens:
            if 0 <= token_id < vocab_size:  # Ensure token_id is in valid range
                class_vector[token_id] = token_id

        # Create output tuple with the classification vector replacing tokens
        output = (class_vector,) + example[1:]
        yield output


def random_spans_noise_mask(
    length,
    noise_density=0.15,
    mean_noise_span_length=3.0,
    seed1=None,
    seed2=None,
    example=None,
):
    """Computes span corruption masks given input parameters."""
    """ This is a straightforward translation of T5's random_spans_noise_mask."""
    # Passing this in case if we want to use for debugging/logging
    del example
    orig_length = length
    # increase length to avoid degeneracy
    length = max(length, 2)
    num_noise_tokens = int(round(length * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # Pick the lengths of the noise spans and the non-noise spans
    def randomly_segment(num_items, num_segments, seed):
        x = np.arange(num_items - 1) < num_segments - 1
        # Set random seed if passed (only in tests for now).
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(x)
        first_in_segment = np.pad(x, (1, 0), mode="constant")
        segment_id = np.cumsum(first_in_segment)

        y = np.roll(segment_id, 1)
        y[0] = 0
        idxs = np.pad(
            np.squeeze(np.argwhere(segment_id - y), axis=1), (1, 0), mode="constant"
        )
        segment_lengths = np.add.reduceat(np.ones_like(segment_id), idxs, axis=0)
        return segment_lengths

    noise_span_lengths = randomly_segment(num_noise_tokens, num_noise_spans, seed1)
    nonnoise_span_lengths = randomly_segment(
        num_nonnoise_tokens, num_noise_spans, seed2
    )
    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2],
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros(length)  # all 0s to begin with
    span_start_indicator[span_starts] = 1
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)
    return is_noise[:orig_length]

def to_unicode(s):
    # Errors of the casting are ignored (e.g. sequences not allowed by UTF-8),
    # in order not to stay with incomplete examples (with empty values).
    return str(s, encoding="utf-8", errors="ignore")


def lower_endian_to_number(l, base):
    """Helper function: convert a list of digits in the given training to a number."""
    return sum([d * (base**i) for i, d in enumerate(l)])


def number_to_lower_endian(n, base):
    """Helper function: convert a number to a list of digits in the given training."""
    if n < base:
        return [n]
    return [n % base] + number_to_lower_endian(n // base, base)


def random_number_lower_endian(length, base):
    """Helper function: generate a random number as a lower-endian digits list."""
    if length == 1:  # Last digit can be 0 only if length is 1.
        return [np.random.randint(base)]
    prefix = [np.random.randint(base) for _ in range(length - 1)]
    return prefix + [np.random.randint(base - 1) + 1]  # Last digit is not 0.


def save_data_counters(output_dir, host_id=None):
    """Checkpoint data counters."""
    global data_counters
    host_id = jax.process_index() if host_id is None else host_id
    fname = os.path.join(output_dir, "data_counters%d.pkl" % host_id)
    with tf.io.gfile.GFile(fname, "wb") as f:
        pickle.dump(data_counters, f)


def load_data_counters(output_dir, host_id=None):
    """Checkpoint data counters."""
    global data_counters
    host_id = jax.process_index() if host_id is None else host_id
    fname = os.path.join(output_dir, "data_counters%d.pkl" % host_id)
    if not tf.io.gfile.exists(fname):
        trax_logging.info("Did not load data counters as %s does not exist.", fname)
        return
    with tf.io.gfile.GFile(fname, "rb") as f:
        obj = pickle.load(f)
    data_counters = obj


def buckets_for_length(
    bucket_length, batch_size, max_eval_length, n_devices, training
):
    """Creates heuristically a set of bucket boundaries and sizes.

    The middle boundary is set to `bucket_length` and the corresponding batch
    size is set to `batch_size`. We also create buckets of 1/2 and 1/4 length
    with 2x and 4x batch size, and buckets of 2x and 4x and larger length with
    1/2 and 1/4 batch size respectively, and batch size 1 for the final one.

    Args:
      bucket_length: the length of the middle bucket.
      batch_size: the batch size for the middle bucket.
      max_eval_length: the longest bucket length if training=False.
      n_devices: number of devices, batch sizes are divisible by that.
      training: bool, whether we are training or evaluating.

    Returns:
      a pair of lists of integers, (bucket_boundaries, bucket_batch_sizes).
    """
    bucket_boundaries = [
        bucket_length // 4,
        bucket_length // 2,
        bucket_length,
        bucket_length * 2,
        bucket_length * 4,
        bucket_length * 8,
        bucket_length * 16,
    ]
    if not training:
        max_eval_length = max_eval_length or bucket_length * 32
        # Set last bucket boundary to be max_eval_length, cut off boundaries
        # that are larger than this.
        bucket_boundaries = [b for b in bucket_boundaries if b < max_eval_length] + [
            max_eval_length
        ]
        bucket_boundaries.append(max_eval_length)
    bucket_batch_sizes = [
        batch_size * 4,
        batch_size * 2,
        batch_size,
        batch_size // 2,
        batch_size // 4,
        batch_size // 8,
        batch_size // 16,
        1,
    ]
    if not training:
        # The last bucket batch size is always 1, but the one-but-last is
        # sized to accommodate the final length = bucket_boundaries[-1], which
        # we changed for eval above -- so adjusting here too.

        # Resize if needed, since bucket_batch_sizes may not be the same size
        # anymore.
        bucket_batch_sizes = bucket_batch_sizes[: len(bucket_boundaries)] + [1]
        bucket_batch_sizes[-2] = batch_size // max_eval_length
    # Make batch sizes divisible by n_devices.
    bucket_batch_sizes = [
        max(b // n_devices, 1) * n_devices for b in bucket_batch_sizes
    ]
    return (bucket_boundaries, bucket_batch_sizes)


def length_fn_(example, length_axis, length_keys):
    """Length is the maximum of shape on length_axis over length_keys."""
    if isinstance(example, (list, tuple)):
        return max([example[i].shape[length_axis] for i in length_keys])
    return example.shape[length_axis]


@dataclass
class StreamBundle:
    """Container for pre-built train/eval streams."""
    train_stream: object
    eval_stream: Optional[object] = None
    train_eval_stream: Optional[object] = None


@gin.configurable(module="trax.data")
def make_streams(train_stream=gin.REQUIRED, eval_stream=None, train_eval_stream=None):
    """Create a StreamBundle from train/eval streams."""
    if isinstance(train_stream, (list, tuple)):
        train_stream = Serial(train_stream)()
    if isinstance(eval_stream, (list, tuple)):
        eval_stream = Serial(eval_stream)()
    if isinstance(train_eval_stream, (list, tuple)):
        train_eval_stream = Serial(train_eval_stream)()
    if eval_stream is None:
        eval_stream = train_stream
    if train_eval_stream is None:
        train_eval_stream = train_stream
    return StreamBundle(
        train_stream=train_stream,
        eval_stream=eval_stream,
        train_eval_stream=train_eval_stream,
    )

@gin.configurable(module="trax.data")
def random_stream(
    input_shape=gin.REQUIRED,
    input_dtype=jnp.int32,
    input_range=(0, 255),
    output_shape=gin.REQUIRED,
    output_dtype=jnp.int32,
    output_range=(0, 9),
):
    """Random batch stream for debugging."""
    if input_dtype in [jnp.float16, jnp.float32, jnp.float64]:
        rand = np.random.uniform
    else:
        rand = np.random.random_integers

    while True:
        inp = rand(input_range[0], input_range[1], input_shape)
        inp = inp.astype(input_dtype)
        out = rand(output_range[0], output_range[1], output_shape)
        out = out.astype(output_dtype)
        yield inp, out


@gin.configurable(module="trax.data")
def random_inputs(
    input_shape=gin.REQUIRED,
    input_dtype=jnp.int32,
    input_range=(0, 255),
    output_shape=gin.REQUIRED,
    output_dtype=jnp.int32,
    output_range=(0, 9),
):
    """Make random StreamBundle for debugging."""
    return make_streams(train_stream=random_stream(
        input_shape=input_shape,
        input_dtype=input_dtype,
        input_range=input_range,
        output_shape=output_shape,
        output_dtype=output_dtype,
        output_range=output_range,
    ))


@gin.configurable(module="trax.data")
def sequence_copy_streams(
    vocab_size=gin.REQUIRED,
    batch_size=gin.REQUIRED,
    train_length=gin.REQUIRED,
    eval_min_length=gin.REQUIRED,
    eval_max_length=gin.REQUIRED,
    reverse=False,
    pad_to_multiple=32,
):
    """Streams for the sequence copy problem: 0w0w for w in [1..vocab_size-1]*."""

    def random_minibatches(length_list):
        """Generate a stream of random mini-batches."""
        while True:
            length = random.choice(length_list)
            assert length % 2 == 0
            w_length = (length // 2) - 1
            w = np.random.randint(
                low=1, high=vocab_size - 1, size=(batch_size, w_length)
            )
            zero = np.zeros([batch_size, 1], np.int32)
            loss_weights = np.concatenate(
                [np.zeros((batch_size, w_length + 2)), np.ones((batch_size, w_length))],
                axis=1,
            )
            if reverse:
                x = np.concatenate([zero, w, zero, jnp.flip(w, axis=1)], axis=1)
            else:
                x = np.concatenate([zero, w, zero, w], axis=1)
            x = _pad_to_multiple_of(x, pad_to_multiple, 1)
            loss_weights = _pad_to_multiple_of(loss_weights, pad_to_multiple, 1)
            yield (x, x, loss_weights)  # Here inputs and targets are the same.

    train_lengths = [2 * (i + 2) for i in range(train_length - 1)]
    eval_lengths = [2 * (i + 1) for i in range(eval_min_length, eval_max_length)]
    return random_minibatches(train_lengths), random_minibatches(eval_lengths)


@gin.configurable(module="trax.data")
def sequence_copy_train_stream(
    vocab_size=gin.REQUIRED,
    batch_size=gin.REQUIRED,
    train_length=gin.REQUIRED,
    eval_min_length=gin.REQUIRED,
    eval_max_length=gin.REQUIRED,
    reverse=False,
    pad_to_multiple=32,
):
    """Train stream for the sequence copy problem."""
    train_stream, _ = sequence_copy_streams(
        vocab_size=vocab_size,
        batch_size=batch_size,
        train_length=train_length,
        eval_min_length=eval_min_length,
        eval_max_length=eval_max_length,
        reverse=reverse,
        pad_to_multiple=pad_to_multiple,
    )
    return train_stream


@gin.configurable(module="trax.data")
def sequence_copy_eval_stream(
    vocab_size=gin.REQUIRED,
    batch_size=gin.REQUIRED,
    train_length=gin.REQUIRED,
    eval_min_length=gin.REQUIRED,
    eval_max_length=gin.REQUIRED,
    reverse=False,
    pad_to_multiple=32,
):
    """Eval stream for the sequence copy problem."""
    _, eval_stream = sequence_copy_streams(
        vocab_size=vocab_size,
        batch_size=batch_size,
        train_length=train_length,
        eval_min_length=eval_min_length,
        eval_max_length=eval_max_length,
        reverse=reverse,
        pad_to_multiple=pad_to_multiple,
    )
    return eval_stream




@gin.configurable(module="trax.data")
def simple_sequence_copy_streams(
    vocab_size=gin.REQUIRED,
    batch_size=gin.REQUIRED,
    train_length=gin.REQUIRED,
    eval_min_length=gin.REQUIRED,
    eval_max_length=gin.REQUIRED,
    pad_to_multiple=32,
):
    """Streams for the sequence copy problem: w for w in [1..vocab_size-1]*."""

    def random_minibatches(length_list):
        """Generate a stream of random mini-batches."""
        while True:
            length = random.choice(length_list)
            x = np.random.randint(low=1, high=vocab_size - 1, size=(batch_size, length))
            loss_weights = np.ones((batch_size, length))
            x = _pad_to_multiple_of(x, pad_to_multiple, 1)
            loss_weights = _pad_to_multiple_of(loss_weights, pad_to_multiple, 1)
            yield (x, x, loss_weights)  # Here inputs and targets are the same.

    train_lengths = list(range(1, train_length + 1))
    eval_lengths = list(range(eval_min_length, eval_max_length + 1))
    return random_minibatches(train_lengths), random_minibatches(eval_lengths)


@gin.configurable(module="trax.data")
def simple_sequence_copy_train_stream(
    vocab_size=gin.REQUIRED,
    batch_size=gin.REQUIRED,
    train_length=gin.REQUIRED,
    eval_min_length=gin.REQUIRED,
    eval_max_length=gin.REQUIRED,
    pad_to_multiple=32,
):
    """Train stream for the simple sequence copy problem."""
    train_stream, _ = simple_sequence_copy_streams(
        vocab_size=vocab_size,
        batch_size=batch_size,
        train_length=train_length,
        eval_min_length=eval_min_length,
        eval_max_length=eval_max_length,
        pad_to_multiple=pad_to_multiple,
    )
    return train_stream


@gin.configurable(module="trax.data")
def simple_sequence_copy_eval_stream(
    vocab_size=gin.REQUIRED,
    batch_size=gin.REQUIRED,
    train_length=gin.REQUIRED,
    eval_min_length=gin.REQUIRED,
    eval_max_length=gin.REQUIRED,
    pad_to_multiple=32,
):
    """Eval stream for the simple sequence copy problem."""
    _, eval_stream = simple_sequence_copy_streams(
        vocab_size=vocab_size,
        batch_size=batch_size,
        train_length=train_length,
        eval_min_length=eval_min_length,
        eval_max_length=eval_max_length,
        pad_to_multiple=pad_to_multiple,
    )
    return eval_stream




@gin.configurable(module="trax.data")
def addition_streams(
    vocab_size=gin.REQUIRED,
    batch_size=gin.REQUIRED,
    train_length=gin.REQUIRED,
    eval_min_length=gin.REQUIRED,
    eval_max_length=gin.REQUIRED,
    pad_to_multiple=32,
    encdec=False,
):
    """Streams for the add problem: <S>x+y<S>(x+y)."""
    train_stream = addition_input_stream(
        vocab_size, batch_size, 3, train_length, pad_to_multiple, encdec
    )
    eval_stream = addition_input_stream(
        vocab_size,
        batch_size,
        eval_min_length,
        eval_max_length,
        pad_to_multiple,
        encdec,
    )
    return train_stream, eval_stream


@gin.configurable(module="trax.data")
def addition_train_stream(
    vocab_size=gin.REQUIRED,
    batch_size=gin.REQUIRED,
    train_length=gin.REQUIRED,
    eval_min_length=gin.REQUIRED,
    eval_max_length=gin.REQUIRED,
    pad_to_multiple=32,
    encdec=False,
):
    """Train stream for the add problem."""
    train_stream, _ = addition_streams(
        vocab_size=vocab_size,
        batch_size=batch_size,
        train_length=train_length,
        eval_min_length=eval_min_length,
        eval_max_length=eval_max_length,
        pad_to_multiple=pad_to_multiple,
        encdec=encdec,
    )
    return train_stream


@gin.configurable(module="trax.data")
def addition_eval_stream(
    vocab_size=gin.REQUIRED,
    batch_size=gin.REQUIRED,
    train_length=gin.REQUIRED,
    eval_min_length=gin.REQUIRED,
    eval_max_length=gin.REQUIRED,
    pad_to_multiple=32,
    encdec=False,
):
    """Eval stream for the add problem."""
    _, eval_stream = addition_streams(
        vocab_size=vocab_size,
        batch_size=batch_size,
        train_length=train_length,
        eval_min_length=eval_min_length,
        eval_max_length=eval_max_length,
        pad_to_multiple=pad_to_multiple,
        encdec=encdec,
    )
    return eval_stream




@gin.configurable(module="trax.data")
def sine_stream(
    batch_size=gin.REQUIRED,
    length=gin.REQUIRED,
    max_phase=(2 * math.pi),
    min_period=0.1,
    max_period=10.0,
):
    """Sinusoids of random period and phase."""

    def random_series():
        while True:
            phase = np.random.uniform(0, max_phase)
            period = np.exp(np.random.uniform(np.log(min_period), np.log(max_period)))
            x = np.arange(length)
            yield np.sin((x - phase) / period)

    minibatch = []
    for series in random_series():
        minibatch.append(series)
        if len(minibatch) == batch_size:
            obs = np.stack(minibatch)
            minibatch.clear()
            act = np.zeros_like(obs, dtype=np.int32)
            mask = np.ones_like(obs)
            yield (obs, act, obs, mask)


@gin.configurable(module="trax.data")
def sine_train_stream(
    batch_size=gin.REQUIRED,
    length=gin.REQUIRED,
    max_phase=(2 * math.pi),
    min_period=0.1,
    max_period=10.0,
):
    """Train stream for sine waves."""
    return sine_stream(
        batch_size=batch_size,
        length=length,
        max_phase=max_phase,
        min_period=min_period,
        max_period=max_period,
    )


@gin.configurable(module="trax.data")
def sine_eval_stream(
    batch_size=gin.REQUIRED,
    length=gin.REQUIRED,
    max_phase=(2 * math.pi),
    min_period=0.1,
    max_period=10.0,
):
    """Eval stream for sine waves."""
    return sine_stream(
        batch_size=batch_size,
        length=length,
        max_phase=max_phase,
        min_period=min_period,
        max_period=max_period,
    )




def _pad_to_multiple_of(x, y, axis):
    """Pads x to multiple of y on the given axis."""
    pad_len = np.ceil(x.shape[axis] / float(y)) * y
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (0, int(pad_len - x.shape[axis]))
    return np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
