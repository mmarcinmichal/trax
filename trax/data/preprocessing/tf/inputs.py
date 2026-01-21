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

from trax.data.preprocessing.modules.math import (
    convert_float_to_mathqa,
    convert_to_subtract,
)

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
