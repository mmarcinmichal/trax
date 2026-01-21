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

"""Tests for AQuA preprocessing."""

import json
import os
import tempfile

import numpy as np

from absl.testing import absltest

from trax.data.preprocessing.modules import aqua as aqua_module


class AquaInputsTest(absltest.TestCase):
    def _write_dataset(self, tmp_dir):
        example = {
            "question": "What is 1+1?",
            "rationale": "step1\nstep2",
            "options": ["a", "b", "c"],
            "correct": "a",
        }
        path = os.path.join(tmp_dir, "train.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(example) + "\n")
        return tmp_dir

    def test_create_aqua_inputs_cumulative(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = self._write_dataset(tmp_dir)

            def legacy_create_aqua_inputs(dataset_path):
                dataset_path = os.path.join(dataset_path, "train.json")
                with open(dataset_path, "r", encoding="utf-8") as handle:
                    dataset = [json.loads(line) for line in handle]

                def gen():
                    while True:
                        for example in dataset:
                            input_prefix = example["question"]
                            steps = example["rationale"].split("\n")
                            for i in range(len(steps)):
                                input_values = "infer cumulative rationale: " + input_prefix
                                target_values = steps[i]
                                input_prefix += " " + steps[i]
                                yield (
                                    input_values,
                                    target_values,
                                    np.array([1] * len(target_values)),
                                )
                return gen

            legacy = legacy_create_aqua_inputs(dataset_path)()
            new = aqua_module.CreateAquaInputs(
                dataset_path=dataset_path, train=True, cumulative=True
            )()

            legacy_example = next(legacy)
            new_example = next(new)
            self.assertEqual(legacy_example[0], new_example[0])
            self.assertEqual(legacy_example[1], new_example[1])
            np.testing.assert_array_equal(legacy_example[2], new_example[2])


if __name__ == "__main__":
    absltest.main()
