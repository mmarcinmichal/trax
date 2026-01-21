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

"""Tests for DROP preprocessing."""

import json
import os
import tempfile
from unittest import mock

import numpy as np
import tensorflow_datasets as tfds

from absl.testing import absltest

from trax.data.preprocessing.modules import drop as drop_module


class DropInputsTest(absltest.TestCase):
    def test_create_drop_inputs_parity(self):
        example = {
            "passage": b"passage",
            "question": b"question",
            "answer": b"3",
        }
        dataset = [example]

        def legacy_create_drop_inputs():
            def gen():
                while True:
                    for ex in dataset:
                        input_values = (
                            "drop question: "
                            + ex["passage"].decode("utf-8")
                            + " "
                            + ex["question"].decode("utf-8")
                        )
                        target_values = ex["answer"].decode("utf-8")
                        yield input_values, target_values, np.array(
                            [1] * len(target_values), dtype=np.int32
                        )
            return gen

        with mock.patch.object(tfds, "load", return_value=dataset), mock.patch.object(
            tfds, "as_numpy", lambda x: x
        ):
            new = drop_module.CreateDropInputs(train=True)()
            legacy = legacy_create_drop_inputs()()

            legacy_example = next(legacy)
            new_example = next(new)
            self.assertEqual(legacy_example[0], new_example[0])
            self.assertEqual(legacy_example[1], new_example[1])
            np.testing.assert_array_equal(legacy_example[2], new_example[2])

    def test_create_annotated_drop_inputs_parity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            example = {
                "passage": "passage",
                "question": "question",
                "calculation": "answer",
            }
            path = os.path.join(tmp_dir, "train_annotated.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(example) + "\n")

            def legacy_create_annotated_drop_inputs(dataset_path):
                dataset_path = os.path.join(dataset_path, "train_annotated.json")
                with open(dataset_path, "r", encoding="utf-8") as dataset_handle:
                    dataset = [json.loads(line) for line in dataset_handle]

                def gen():
                    while True:
                        for ex in dataset:
                            question = ex["passage"] + " " + ex["question"]
                            input_values = "drop annotated question: " + question
                            target_values = ex["calculation"]
                            yield (
                                input_values,
                                target_values,
                                np.array([1] * len(target_values), dtype=np.int32),
                            )
                return gen

            legacy = legacy_create_annotated_drop_inputs(tmp_dir)()
            new = drop_module.CreateAnnotatedDropInputs(
                dataset_path=tmp_dir, train=True, single_file=True
            )()

            legacy_example = next(legacy)
            new_example = next(new)
            self.assertEqual(legacy_example[0], new_example[0])
            self.assertEqual(legacy_example[1], new_example[1])
            np.testing.assert_array_equal(legacy_example[2], new_example[2])


if __name__ == "__main__":
    absltest.main()
