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

"""TensorFlow preprocessing utilities for Trax input pipelines."""

import functools
import itertools
import json
import os
import random
import re

import gin
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from absl import logging

from trax import fastmath
from trax.data.encoder.encoder import SentencePieceEncoder
from data.loader.tf.interface import DatasetLoader, DatasetStreams
from trax.data.loader.tf.base import dataset_to_stream
from trax.data.preprocessing.tf.math import (
    convert_float_to_mathqa,
    convert_to_subtract,
)

# How many examples from the stream to skip at random during training.
# For now, we skip at most 100K examples for efficiency.
_MAX_SKIP_EXAMPLES = 100_000


def t5_data():
    """Get the T5 data module if available."""
    module = None
    try:
        import t5.data  # pylint: disable=g-import-not-at-top

        module = t5.data
    except AttributeError as e:
        logging.error("pip install t5")
        raise e
    return module


def random_split_text_tf(max_words_per_segment=512, text_key="text"):
    """Returns a TFDS preprocessing function that chunks long text randomly."""

    def preprocess_fn(dataset):
        def random_chunk(example):
            text = example[text_key]
            tokens = tf.strings.split([text]).values
            length = tf.size(tokens)

            max_len = tf.minimum(length, max_words_per_segment)
            start = tf.random.uniform(
                shape=[], maxval=length - max_len + 1, dtype=tf.int32
            )
            chunk = tokens[start : start + max_len]

            example[text_key] = tf.strings.reduce_join(chunk, separator=" ")
            return example

        return dataset.map(random_chunk, num_parallel_calls=tf.data.AUTOTUNE)

    return preprocess_fn


def next_sentence_prediction_tf(
    text_key="text", label_sentences=True, buffer_size=50000
):
    """Returns a TFDS preprocessing function for NSP."""
    del label_sentences

    def preprocess_fn(dataset):
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
        other_dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
        combined = tf.data.Dataset.zip((dataset, other_dataset))

        def create_nsp_example(a, b):
            text_a = a[text_key]
            text_b = b[text_key]

            def first_two_sentences(text):
                rt = tf.strings.split([text], sep=". ")
                dense = rt.to_tensor(default_value="")
                n = tf.shape(dense)[1]
                first = tf.cond(
                    tf.greater(n, 0),
                    lambda: dense[0, 0],
                    lambda: tf.constant("", dtype=tf.string),
                )
                second = tf.cond(
                    tf.greater(n, 1),
                    lambda: dense[0, 1],
                    lambda: first,
                )
                return first, second

            first_sentence, a_second = first_two_sentences(text_a)
            b_first, _ = first_two_sentences(text_b)

            use_random = tf.random.uniform(()) < 0.5
            second_sentence = tf.cond(use_random, lambda: b_first, lambda: a_second)

            input_text = tf.strings.join(
                ["sentence1: ", first_sentence, " sentence2: ", second_sentence]
            )
            label = tf.where(use_random, "not_next", "next")

            return {"inputs": input_text, "targets": label}

        return combined.map(create_nsp_example)

    return preprocess_fn


def no_preprocess(dataset, training):
    del training
    return dataset


def _append_targets(example, target_names):
    if len(target_names) == 1:
        return (example, example[target_names[0]])
    targets = {}
    for name in target_names:
        targets[name] = example[name]
    return (example, targets)


def _prepare_tf_dataset(
    dataset,
    target_names,
    training,
    shuffle_buffer_size,
    preprocess_fn,
    bare_preprocess_fn,
    seed=None,
    shuffle=True,
):
    if bare_preprocess_fn is not None:
        dataset = bare_preprocess_fn(dataset, training)
    dataset = dataset.map(lambda x: _append_targets(x, target_names))
    dataset = dataset.repeat()
    if training:
        rng = random.Random(seed) if seed is not None else random
        dataset = dataset.skip(rng.randint(0, _MAX_SKIP_EXAMPLES))
    dataset = preprocess_fn(dataset, training)
    if shuffle and shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    return dataset.prefetch(8)


def _resolve_datasets(datasets):
    if isinstance(datasets, DatasetLoader):
        datasets = datasets.datasets()
    if callable(datasets):
        datasets = datasets()
    if isinstance(datasets, DatasetStreams):
        return datasets.train, datasets.eval, datasets.supervised_keys
    if isinstance(datasets, (list, tuple)) and len(datasets) == 3:
        return datasets
    raise ValueError(
        "Expected DatasetLoader, DatasetStreams, or (train, eval, supervised_keys)."
    )


@gin.configurable(module="trax.data")
def tf_dataset_streams(  # pylint: disable=invalid-name
    datasets=gin.REQUIRED,
    preprocess_fn=no_preprocess,
    bare_preprocess_fn=None,
    shuffle_buffer_size=1024,
    shuffle=True,
    seed=None,
    input_name=None,
    target_name=None,
):
    """Apply TF dataset preprocessing and return train/eval numpy streams."""
    train_ds, eval_ds, keys = _resolve_datasets(datasets)
    if train_ds is None:
        raise ValueError("Training requested but train dataset is None.")

    input_names = (
        [input_name]
        if input_name is not None
        else keys[0]
        if keys is not None
        else [None]
    )
    target_names = (
        [target_name]
        if target_name is not None
        else keys[1]
        if keys is not None
        else [None]
    )
    if target_names == [None]:
        raise ValueError("Target name must be provided when supervised keys are missing.")

    train_batches = _prepare_tf_dataset(
        train_ds,
        target_names,
        True,
        shuffle_buffer_size,
        preprocess_fn,
        bare_preprocess_fn,
        seed=seed,
        shuffle=shuffle,
    )
    eval_batches = _prepare_tf_dataset(
        eval_ds,
        target_names,
        False,
        shuffle_buffer_size,
        preprocess_fn,
        bare_preprocess_fn,
        seed=seed,
        shuffle=shuffle,
    )
    input_name_c = input_names[0]

    def stream(which):
        dataset = eval_batches if which == "eval" else train_batches
        return dataset_to_stream(dataset, input_name_c)

    train_stream = lambda: stream("train")
    eval_stream = lambda: stream("eval")
    return train_stream, eval_stream


@gin.configurable(module="trax.data")
def tf_dataset_streams_serial(  # pylint: disable=invalid-name
    datasets=gin.REQUIRED,
    preprocess_fn=gin.REQUIRED,
    shuffle_buffer_size=1024,
    shuffle=True,
    seed=None,
):
    """Return train/eval numpy streams using Serial preprocessing pipelines."""
    train_ds, eval_ds, _ = _resolve_datasets(datasets)
    if train_ds is None:
        raise ValueError("Training requested but train dataset is None.")

    def _prepare(dataset, training):
        dataset = dataset.repeat()
        if training:
            rng = random.Random(seed) if seed is not None else random
            dataset = dataset.skip(rng.randint(0, _MAX_SKIP_EXAMPLES))
        if shuffle and shuffle_buffer_size:
            dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
        return dataset.prefetch(8)

    train_batches = _prepare(train_ds, True)
    eval_batches = _prepare(eval_ds, False)

    def _normalize_example(example):
        if isinstance(example, (list, tuple)) and len(example) == 2:
            features, targets = example
            if isinstance(features, dict):
                normalized = dict(features)
                if "targets" not in normalized:
                    normalized["targets"] = targets
                return normalized
        return example

    def _resolve_pipeline(training):
        try:
            pipeline = preprocess_fn(training=training)
        except TypeError:
            pipeline = preprocess_fn
        return pipeline

    def stream(which):
        dataset = eval_batches if which == "eval" else train_batches
        pipeline = _resolve_pipeline(which != "eval")
        generator = (
            _normalize_example(example)
            for example in fastmath.dataset_as_numpy(dataset)
        )
        return pipeline(generator)

    train_stream = lambda: stream("train")
    eval_stream = lambda: stream("eval")
    return train_stream, eval_stream


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def downsampled_imagenet_flatten_bare_preprocess(dataset, training):
    """Preprocessing for downsampled_imagenet."""
    del training

    def flatten_image(features):
        img = features["image"]
        flat = tf.cast(tf.reshape(img, [-1]), tf.int64)

        new_features = {"image": flat}
        return new_features

    return dataset.map(flatten_image)


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def concat_preprocess(dataset, training, pad_symbol=0):
    """Pre-processing function that concatenates input and target for LM."""
    del training

    def concat(features, targets):
        inp = features["inputs"]
        pad = tf.expand_dims(tf.zeros_like(inp[0]) + pad_symbol, axis=0)
        concat = tf.concat([pad, inp, pad, targets], axis=0)
        features["inputs"] = concat
        return features, concat

    dataset = dataset.map(concat)
    return dataset


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def squeeze_targets_preprocess(dataset, training):
    """Pre-processing function that squeezes last axis of targets."""
    del training

    def squeeze(features, targets):
        if targets.shape[-1] == 1:
            targets = tf.squeeze(targets, axis=-1)
        return features, targets

    dataset = dataset.map(squeeze)
    return dataset


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def lm1b_preprocess(dataset, training, max_target_length=-1, max_eval_target_length=-1):
    """Preprocessing for LM1B: filter out targets exceeding maximum length."""

    def target_right_length(_, target):
        return tf.less(tf.shape(target)[0], max_target_length + 1)

    def eval_target_right_length(_, target):
        return tf.less(tf.shape(target)[0], max_eval_target_length + 1)

    if max_target_length > 0 and training:
        dataset = dataset.filter(target_right_length)

    if max_eval_target_length > 0 and not training:
        dataset = dataset.filter(eval_target_right_length)

    return dataset


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def lm_token_preprocessing(dataset, training):
    """Concatenates inputs, 0, targets, with masking only for targets."""
    del training

    def concat_and_add_mask(x):
        inp = x["inputs"]
        targets = x["targets"]
        pad = tf.expand_dims(tf.zeros_like(inp[0]), axis=0)
        concat = tf.concat([inp, pad, targets], axis=0)
        mask = tf.concat([tf.zeros_like(inp), pad, tf.ones_like(targets)], axis=0)
        x["inputs"] = concat
        x["targets"] = concat
        x["mask"] = mask
        return x

    dataset = dataset.map(concat_and_add_mask)
    return dataset


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def bair_robot_pushing_preprocess(dataset, training):
    """Pre-processing function that concatenates input and target frames."""
    del training

    def concat_and_add_mask(features, targets):
        inp = features["inputs"]
        concat = tf.concat([inp, targets], axis=0)
        mask = tf.concat([tf.zeros_like(inp), tf.ones_like(targets)], axis=0)
        concat = tf.reshape(concat, (-1,))
        mask = tf.reshape(mask, (-1,))
        concat = tf.cast(concat, tf.int32)
        mask = tf.cast(mask, tf.float32)
        features["inputs"] = features["targets"] = concat
        features["mask"] = mask
        return features, concat

    dataset = dataset.map(concat_and_add_mask)
    return dataset


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def filter_dataset_on_len(dataset, training, len_map=None, filter_on_eval=False):
    """Filters a dataset of lengths given in `len_map`."""
    if (len_map is None) or (not training and not filter_on_eval):
        return dataset

    assert isinstance(len_map, dict)
    for k, bounds in len_map.items():
        def within_bounds(x, key, len_bounds):
            size = tf.shape(x[key])[0]
            min_len, max_len = len_bounds
            return (min_len <= size) and (size <= max_len)

        dataset = dataset.filter(lambda x: within_bounds(x, k, bounds))

    return dataset


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def truncate_dataset_on_len(dataset, training, len_map=None, truncate_on_eval=False):
    """Truncates features in an example to lengths given in `len_map`."""
    if (len_map is None) or (not training and not truncate_on_eval):
        return dataset

    assert isinstance(len_map, dict)

    def truncate_example(x):
        for key, max_len in len_map.items():
            x_len = tf.shape(x[key])[0]
            if x_len > max_len:
                x[key] = x[key][:max_len, ...]
        return x

    return dataset.map(truncate_example)


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def pad_dataset_to_length(dataset, training, len_map=None):
    """Pad features less than specified length to specified length."""
    del training
    if len_map is None:
        return dataset

    def pad_to_len(x):
        for key, max_len in len_map.items():
            x_shape = tf.shape(x[key])
            x_len = x_shape[0]
            if x_len < max_len:
                pad_shape = [
                    max_len - x_len,
                ]
                zeros = tf.zeros(pad_shape, dtype=x[key].dtype)
                x[key] = tf.concat([x[key], zeros], 0)
        return x

    return dataset.map(pad_to_len)


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def add_eos_to_output_features(dataset, training, output_features="targets", eos=1):
    """Adds `EOS` to all features in `output_features`."""
    del training
    if not isinstance(output_features, (list, tuple)):
        output_features = [output_features]

    def add_eos(x):
        for output_feature in output_features:
            x[output_feature] = tf.concat([x[output_feature], [eos]], axis=0)
        return x

    return dataset.map(add_eos)


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def select_random_chunk_t5(
    dataset, training, sequence_length=None, output_features=None
):
    """Select a random chunk from the input tokens."""
    del training
    del output_features

    def select_chunk(features):
        if sequence_length is None:
            return features

        tokens = features["inputs"]
        seq_len = tf.shape(tokens)[0]

        max_start = tf.maximum(seq_len - sequence_length, 0)
        start_index = tf.random.uniform(
            [], minval=0, maxval=max_start + 1, dtype=tf.int32
        )

        chunk = tokens[start_index : start_index + sequence_length]

        features["inputs"] = chunk
        features["targets"] = chunk

        return features

    return dataset.map(select_chunk, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def split_tokens_t5(dataset, training, sequence_length=None, output_features=None):
    """Split tokens into two parts."""
    del training
    del output_features

    def split(features):
        if sequence_length is None:
            return features

        tokens = features["inputs"]
        seq_len = tf.shape(tokens)[0]

        split_point = seq_len // 2

        features["inputs"] = tokens[:split_point]
        features["targets"] = tokens[split_point:]

        return features

    return dataset.map(split, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def denoise_t5(
    dataset, training, sequence_length=None, output_features=None, noise_density=0.15
):
    """Apply denoising to the tokens."""
    del training
    del output_features

    def apply_noise(features):
        if sequence_length is None:
            return features

        tokens = features["inputs"]

        mask = tf.random.uniform(tf.shape(tokens), minval=0, maxval=1) < noise_density
        noisy_tokens = tf.where(mask, tf.zeros_like(tokens), tokens)

        features["inputs"] = noisy_tokens
        features["targets"] = tokens

        return features

    return dataset.map(apply_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def _pad_punctuation(text):
    text = tf.strings.regex_replace(text, r"([[:punct:]])", r" \1 ")
    text = tf.strings.regex_replace(text, r"\s+", " ")
    return text


def _string_join(lst):
    out = tf.strings.join(lst, separator=" ")
    return tf.strings.regex_replace(out, r"\s+", " ")


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def squad_t5(dataset, training, include_context=True):
    """Convert SQuAD examples to a text2text pair."""
    del training

    def squad(x):
        a = _pad_punctuation(x["answers"]["text"])
        q = _pad_punctuation(x["question"])
        c = _pad_punctuation(x["context"])
        if include_context:
            inputs = _string_join(["question:", q, "context:", c])
        else:
            inputs = _string_join(["squad trivia question:", q])
        return {
            "inputs": inputs,
            "targets": a[0],
            "id": x["id"],
            "context": c,
            "question": q,
            "answers": a,
        }

    return dataset.map(squad, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def rekey_t5(dataset, training, key_map=None):
    """Replace the feature keys according to the mapping in `key_map`."""
    del training

    def rekey(x):
        if key_map:
            return {
                new_key: x[old_key] if old_key else ""
                for new_key, old_key in key_map.items()
            }
        return x

    return dataset.map(rekey, num_parallel_calls=tf.data.experimental.AUTOTUNE)


_PREPROCESSOR_REGISTRY = {
    "next_sentence_prediction_tf": next_sentence_prediction_tf,
    "random_split_text_tf": random_split_text_tf,
    "select_random_chunk_t5": select_random_chunk_t5,
    "split_tokens_t5": split_tokens_t5,
    "denoise_t5": denoise_t5,
    "squad_t5": squad_t5,
    "rekey_t5": rekey_t5,
}


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def unsupervised_preprocessors(
    dataset, training, sequence_length=None, output_features=None, preprocessors=None
):
    """Apply a series of unsupervised preprocessors."""
    del training

    if preprocessors is None:
        return dataset

    for preprocessor in preprocessors:
        dataset = preprocessor(
            dataset,
            None,
            sequence_length=sequence_length,
            output_features=output_features,
        )

    return dataset


@gin.configurable(module="trax.data", denylist=["dataset", "training"])
def generic_text_dataset_preprocess_fn(
    dataset,
    training=True,
    text_preprocess_fns=None,
    token_preprocess_fns=None,
    spm_path=None,
    copy_pretokenized=False,
    debug_print_examples=False,
    debug_print_examples_rate=0.01,
):
    """Pre-processes, tokenizes and post-processes a `tf.data.Dataset`."""
    if text_preprocess_fns is not None:
        for text_preprocess_fn in text_preprocess_fns:
            dataset = text_preprocess_fn(dataset, training)

    if debug_print_examples:

        def print_examples(x):
            if np.random.uniform() < debug_print_examples_rate:
                tf.print(x, output_stream=logging.info)
            return x

        dataset = dataset.map(print_examples)

    tokenizer = SentencePieceEncoder(spm_path)

    def tokenize_fields(example):
        inputs = example.get("inputs", example["targets"])
        targets = example["targets"]

        tokenized_inputs = tf.cast(tokenizer.encode(inputs), tf.int64)
        tokenized_targets = tf.cast(tokenizer.encode(targets), tf.int64)

        new_example = {
            "inputs": tokenized_inputs,
            "targets": tokenized_targets,
        }
        if copy_pretokenized:
            new_example["inputs_pretokenized"] = inputs
            new_example["targets_pretokenized"] = targets

        return new_example

    dataset = dataset.map(tokenize_fields)

    if token_preprocess_fns is not None:
        for token_preprocess_fn in token_preprocess_fns:
            dataset = token_preprocess_fn(dataset, training)

    if debug_print_examples:

        def print_examples_and_shapes(x):
            if np.random.uniform() < debug_print_examples_rate:
                tf.print(
                    "inputs_shape:",
                    tf.size(x["inputs"]),
                    "targets_shape:",
                    tf.size(x["targets"]),
                    "inputs:",
                    x["inputs"],
                    "targets:",
                    x["targets"],
                    output_stream=logging.info,
                )
            return x

        dataset = dataset.map(print_examples_and_shapes)

    return dataset


@gin.configurable(module="trax.data")
def get_t5_preprocessor_by_name(name=None, fn_kwargs=None):
    """Returns a closure of any T5 preprocessor function with its arguments."""
    if name is None or name not in _PREPROCESSOR_REGISTRY:
        raise ValueError(f"Unknown or missing preprocessor name: '{name}'.")

    fn = _PREPROCESSOR_REGISTRY[name]
    if fn_kwargs:
        fn = functools.partial(fn, **fn_kwargs)

    return lambda ds, training: fn(ds, training)


@gin.configurable(module="trax.data")
def CorpusToRandomChunks(  # pylint: disable=invalid-name
    dataset_name, num_tokens=512, train=True, data_dir=None
):
    datasets = DatasetLoader(
        dataset_name=dataset_name,
        data_dir=data_dir,
        require_train_split=train,
    ).datasets()
    dataset = datasets.train if train else datasets.eval
    dataset = random_split_text_tf(
        max_words_per_segment=num_tokens,
        text_key="text",
    )(dataset)
    dataset = dataset.map(lambda x: (x["text"],))
    dataset = dataset.repeat()

    def gen(generator=None):
        del generator
        for example in fastmath.dataset_as_numpy(dataset):
            yield example

    return gen


@gin.configurable(module="trax.data")
def CreateAquaInputs(  # pylint: disable=invalid-name
    dataset_path=None,
    train=True,
    cumulative=False,
    rationale=False,
    correct_answer=False,
    correct_answer_given_reasoning=False,
    partial_reasoning=True,
    order_prediction=False,
):
    """Prepares Aqua inputs."""
    if train:
        dataset_path = os.path.join(dataset_path, "train.json")
    else:
        dataset_path = os.path.join(dataset_path, "dev.json")
    dataset_handle = tf.io.gfile.GFile(dataset_path, "r")
    dataset = []
    for line in dataset_handle:
        dataset.append(json.loads(line))

    def aqua_yield_examples(generator=None):
        del generator
        while True:
            for example in itertools.cycle(dataset):
                input_prefix = example["question"]
                steps = example["rationale"].split("\n")
                if cumulative:
                    for i in range(len(steps)):
                        input_values = "infer cumulative rationale: " + input_prefix
                        target_values = steps[i]
                        input_prefix += " " + steps[i]
                        yield (
                            input_values,
                            target_values,
                            np.array([1] * len(target_values)),
                        )
                elif rationale:
                    input_values = "infer full rationale: " + input_prefix
                    target_values = example["rationale"]
                    yield (
                        input_values,
                        target_values,
                        np.array([1] * len(target_values)),
                    )
                elif correct_answer:
                    input_values = "infer correct answer: " + input_prefix
                    input_values += " " + " ".join(example["options"])
                    target_values = example["correct"]
                    yield (
                        input_values,
                        target_values,
                        np.array([1] * len(target_values)),
                    )
                elif correct_answer_given_reasoning:
                    input_values = (
                        "infer correct answer given reasoning: " + input_prefix
                    )
                    if partial_reasoning:
                        reasoning_list = example["rationale"].split("\n")
                        reasoning_list = reasoning_list[
                            0 : np.random.randint(0, len(reasoning_list))
                        ]
                        reasoning = "\n".join(reasoning_list)
                    else:
                        reasoning = example["rationale"]
                    input_values += (
                        " " + example["rationale"] + " " + " ".join(example["options"])
                    )
                    target_values = example["correct"]
                    yield (
                        input_values,
                        target_values,
                        np.array([1] * len(target_values)),
                    )
                elif order_prediction:
                    if np.random.uniform() < 0.5 and len(steps) >= 2:
                        idx = range(len(steps))
                        i1, i2 = random.sample(idx, 2)
                        steps[i1], steps[i2] = steps[i2], steps[i1]
                        target_values = "not_ordered"
                    else:
                        target_values = "ordered"
                    input_values = (
                        "order prediction: " + input_prefix + " " + "\n".join(steps)
                    )
                    yield (
                        input_values,
                        target_values,
                        np.array([1] * len(target_values)),
                    )
                else:
                    raise ValueError(
                        "One of the boolean parameters of the Aqua generator must be set to True."
                    )

    return aqua_yield_examples


@gin.configurable(module="trax.data")
def CreateAnnotatedDropInputs(  # pylint: disable=invalid-name
    dataset_path=None,
    train=True,
    single_file=True,
    unique=False,
    total_number_of_samples=None,
    percentile=1.0,
):
    r"""Prepares annotated Drop inputs."""
    if train:
        if single_file:
            dataset_path = os.path.join(dataset_path, "train_annotated.json")
    else:
        dataset_path = os.path.join(dataset_path, "dev_annotated.json")

    def load_dataset():
        dataset = []
        if single_file:
            dataset_handle = tf.io.gfile.GFile(dataset_path, "r")
            for line in dataset_handle:
                dataset.append(json.loads(line))
        else:
            all_files = tf.io.gfile.listdir(dataset_path)
            for filename in all_files:
                if "json" in filename:
                    print("Loading data from file {}".format(filename))
                    with tf.io.gfile.GFile(os.path.join(dataset_path, filename)) as f:
                        for line in f:
                            dataset.append(json.loads(line))
        print("The total size of the dataset {}".format(len(dataset)))
        return dataset[: int(len(dataset) * percentile)]

    def drop_annotated_yield_examples(generator=None):
        del generator
        while True:
            passages = set()
            unique_examples = set()
            dataset = load_dataset()
            for example in dataset:
                if total_number_of_samples:
                    if len(unique_examples) >= total_number_of_samples:
                        break
                if "input" in example.keys():
                    question = example["input"]
                    question = question[question.find(":") + 2 :]
                else:
                    if unique and example["passage"] in passages:
                        continue
                    passages.add(example["passage"])
                    question = example["passage"] + " " + example["question"]
                    list_num = [
                        float(
                            num.replace(",", "").rstrip(".").lstrip(".")
                        )
                        for num in re.findall(
                            r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",
                            question,
                        )
                    ]
                    for i in range(len(list_num)):
                        question += " n{} = {}".format(i, list_num[i])
                input_values = "drop annotated question: " + question
                target_values = example["calculation"]
                unique_examples.add((input_values, target_values))
                yield (
                    input_values,
                    target_values,
                    np.array([1] * len(target_values), dtype=np.int32),
                )

    return drop_annotated_yield_examples


@gin.configurable(module="trax.data")
def CreateDropInputs(train=True, mathqa_format=False):  # pylint: disable=invalid-name
    """Prepares Drop inputs."""
    if train:
        dataset = tfds.load(name="drop", split="train")
    else:
        dataset = tfds.load(name="drop", split="dev")
    dataset = tfds.as_numpy(dataset)

    def drop_yield_examples(generator=None):
        del generator
        while True:
            for example in itertools.cycle(dataset):
                input_values = (
                    "drop question: "
                    + example["passage"].decode("utf-8")
                    + " "
                    + example["question"].decode("utf-8")
                )
                target_values = example["answer"].decode("utf-8")
                if not target_values:
                    continue
                if mathqa_format:
                    if target_values.replace(".", "", 1).isdigit():
                        target_values = convert_to_subtract(
                            convert_float_to_mathqa(target_values)
                        )
                yield input_values, target_values, np.array(
                    [1] * len(target_values), dtype=np.int32
                )

    return drop_yield_examples
