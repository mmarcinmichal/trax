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

"""DROP-specific preprocessing helpers."""

import itertools
import json
import os
import re

import gin
import numpy as np
import tensorflow_datasets as tfds

from trax.data.preprocessing.modules.math import (
    convert_float_to_mathqa,
    convert_to_subtract,
)
from trax.utils import logging as trax_logging


@gin.configurable(module="trax.data")
def CreateAnnotatedDropInputs(  # pylint: disable=invalid-name
    dataset_path=None,
    train=True,
    single_file=True,
    unique=False,
    total_number_of_samples=None,
    percentile=1.0,
):
    r"""Prepares annotated DROP inputs."""
    if train:
        if single_file:
            dataset_path = os.path.join(dataset_path, "train_annotated.json")
    else:
        dataset_path = os.path.join(dataset_path, "dev_annotated.json")

    def load_dataset():
        dataset = []
        if single_file:
            with open(dataset_path, "r", encoding="utf-8") as dataset_handle:
                for line in dataset_handle:
                    dataset.append(json.loads(line))
        else:
            for filename in os.listdir(dataset_path):
                if "json" in filename:
                    trax_logging.info(
                        "Loading data from file %s", filename, stdout=True
                    )
                    with open(os.path.join(dataset_path, filename), "r", encoding="utf-8") as handle:
                        for line in handle:
                            dataset.append(json.loads(line))
        trax_logging.info(
            "The total size of the dataset %s", len(dataset), stdout=True
        )
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
                        float(num.replace(",", "").rstrip(".").lstrip("."))
                        for num in re.findall(
                            r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",
                            question,
                        )
                    ]
                    for i, num in enumerate(list_num):
                        question += f" n{i} = {num}"
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
    """Prepares DROP inputs."""
    split = "train" if train else "dev"
    dataset = tfds.load(name="drop", split=split)
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
