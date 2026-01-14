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

"""Tests for WMT preprocessing compatibility between TF and Serial pipelines."""

import os

import numpy as np
import tensorflow as tf

from absl.testing import absltest, parameterized

from trax.data.encoder import encoder
from trax.data.preprocessing import inputs as serial_inputs
from trax.data.preprocessing.tf import wmt as tf_wmt


class WMTPreprocessCompatibilityTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        pkg_dir, _ = os.path.split(__file__)
        vocab_dir = os.path.normpath(
            os.path.join(pkg_dir, "../../../resources/data/vocabs/test")
        )
        vocab_path = os.path.join(
            vocab_dir, "vocab.translate_ende_wmt32k.32768.subwords"
        )
        self.tokenizer = encoder.SubwordTextEncoder(filename=vocab_path)

    @staticmethod
    def _make_tf_dataset(inputs, targets):
        features = {"en": tf.constant(inputs), "de": tf.constant(targets)}
        return tf.data.Dataset.from_tensor_slices((features, tf.constant(targets)))

    @staticmethod
    def _make_serial_stream(inputs, targets):
        def _stream():
            for text_input, text_target in zip(inputs, targets):
                yield {"en": text_input, "de": text_target}

        return _stream

    def test_wmt_preprocess_matches_serial(self):
        inputs = ["Hello", "Goodbye"]
        targets = ["Hallo", "Auf Wiedersehen"]
        tf_dataset = tf_wmt.wmt_preprocess(
            self._make_tf_dataset(inputs, targets),
            training=True,
            max_length=50,
            max_eval_length=50,
            tokenizer=self.tokenizer,
        )
        tf_results = list(tf_dataset.as_numpy_iterator())

        serial_stream = serial_inputs.WMTPreprocess(
            tokenizer=self.tokenizer,
            max_length=50,
            max_eval_length=50,
            training=True,
        )(self._make_serial_stream(inputs, targets)())
        serial_results = list(serial_stream)

        self.assertLen(tf_results, len(serial_results))
        for (tf_features, tf_targets), (serial_inputs_arr, serial_targets_arr) in zip(
            tf_results, serial_results
        ):
            np.testing.assert_array_equal(tf_features["inputs"], serial_inputs_arr)
            np.testing.assert_array_equal(tf_targets, serial_targets_arr)

    @parameterized.named_parameters(
        ("train", True),
        ("eval", False),
    )
    def test_wmt_preprocess_filters_consistently(self, training):
        inputs = ["short", "toolong"]
        targets = ["ok", "alsoverylong"]
        tf_dataset = tf_wmt.wmt_preprocess(
            self._make_tf_dataset(inputs, targets),
            training=training,
            max_length=5,
            max_eval_length=5,
            tokenizer=self.tokenizer,
        )
        tf_results = list(tf_dataset.as_numpy_iterator())

        serial_stream = serial_inputs.WMTPreprocess(
            tokenizer=self.tokenizer,
            max_length=5,
            max_eval_length=5,
            training=training,
        )(self._make_serial_stream(inputs, targets)())
        serial_results = list(serial_stream)

        self.assertLen(tf_results, len(serial_results))
        for (tf_features, tf_targets), (serial_inputs_arr, serial_targets_arr) in zip(
            tf_results, serial_results
        ):
            np.testing.assert_array_equal(tf_features["inputs"], serial_inputs_arr)
            np.testing.assert_array_equal(tf_targets, serial_targets_arr)

    def test_wmt_concat_preprocess_matches_serial(self):
        inputs = ["Hello", "Hi"]
        targets = ["Hallo", "Czesc"]
        tf_dataset = tf_wmt.wmt_concat_preprocess(
            self._make_tf_dataset(inputs, targets),
            training=True,
            max_length=50,
            max_eval_length=50,
            tokenizer=self.tokenizer,
        )
        tf_results = list(tf_dataset.as_numpy_iterator())

        serial_stream = serial_inputs.WMTConcatPreprocess(
            tokenizer=self.tokenizer,
            max_length=50,
            max_eval_length=50,
            training=True,
        )(self._make_serial_stream(inputs, targets)())
        serial_results = list(serial_stream)

        self.assertLen(tf_results, len(serial_results))
        for (tf_features, tf_targets), (serial_concat, serial_output, serial_mask) in zip(
            tf_results, serial_results
        ):
            np.testing.assert_array_equal(tf_features["inputs"], serial_concat)
            np.testing.assert_array_equal(tf_targets, serial_output)
            np.testing.assert_array_equal(tf_features["mask"], serial_mask)


if __name__ == "__main__":
    absltest.main()
