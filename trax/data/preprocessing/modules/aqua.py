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

"""AQuA-specific preprocessing helpers."""

import itertools
import json
import os
import random

import gin
import numpy as np


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
    """Prepares AQuA inputs."""
    if train:
        dataset_path = os.path.join(dataset_path, "train.json")
    else:
        dataset_path = os.path.join(dataset_path, "dev.json")
    with open(dataset_path, "r", encoding="utf-8") as dataset_handle:
        dataset = [json.loads(line) for line in dataset_handle]

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
                    input_values = "infer correct answer given reasoning: " + input_prefix
                    if partial_reasoning:
                        reasoning_list = example["rationale"].split("\n")
                        reasoning_list = reasoning_list[
                            0 : np.random.randint(0, len(reasoning_list))
                        ]
                        reasoning = "\n".join(reasoning_list)
                    else:
                        reasoning = example["rationale"]
                    input_values += (
                        " " + reasoning + " " + " ".join(example["options"])
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
