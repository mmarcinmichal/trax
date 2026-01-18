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

"""Tests for Serial C4 preprocessing."""

import itertools

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from absl.testing import absltest

from tests.data.utils import _c4_dataset, _spm_path
from trax import data


class InputsC4Test(absltest.TestCase):
    def test_c4_preprocess_char_filters(self):
        raw_examples = list(itertools.islice(tfds.as_numpy(_c4_dataset()), 50))
        self.assertGreater(len(raw_examples), 0)

        processed = list(
            data.C4Preprocess(max_target_length=50)(
                (example for example in raw_examples)
            )
        )
        self.assertLessEqual(len(processed), len(raw_examples))

        for inputs_arr, targets_arr in processed:
            self.assertIsInstance(inputs_arr, np.ndarray)
            self.assertEqual(inputs_arr.dtype, np.int64)
            np.testing.assert_array_equal(inputs_arr, targets_arr)
            self.assertLessEqual(len(inputs_arr), 50)

    def test_c4_preprocess_spc_shorter_or_equal(self):
        raw_examples = list(itertools.islice(tfds.as_numpy(_c4_dataset()), 10))
        self.assertGreater(len(raw_examples), 0)

        char_results = list(
            data.C4Preprocess(max_target_length=-1)(
                (example for example in raw_examples)
            )
        )
        spc_results = list(
            data.C4Preprocess(
                max_target_length=-1, tokenization="spc", spm_path=_spm_path()
            )((example for example in raw_examples))
        )

        self.assertLen(char_results, len(spc_results))
        for (char_inputs, char_targets), (spc_inputs, spc_targets) in zip(
            char_results, spc_results
        ):
            self.assertEqual(char_inputs.dtype, np.int64)
            self.assertEqual(spc_inputs.dtype, np.int64)
            np.testing.assert_array_equal(char_inputs, char_targets)
            np.testing.assert_array_equal(spc_inputs, spc_targets)
            self.assertLessEqual(len(spc_inputs), len(char_inputs))


if __name__ == "__main__":
    tf.test.main()
