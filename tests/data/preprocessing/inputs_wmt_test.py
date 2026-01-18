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

"""Tests for WMT preprocessing serial pipelines."""

import os

import numpy as np

from absl.testing import absltest, parameterized

from trax.data.encoder import encoder
from trax.data.preprocessing import inputs as serial_inputs


class WMTPreprocessTest(parameterized.TestCase):
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
    def _make_serial_stream(inputs, targets):
        def _stream():
            for text_input, text_target in zip(inputs, targets):
                yield {"en": text_input, "de": text_target}

        return _stream

    def test_wmt_preprocess_tokenizes_inputs_targets(self):
        inputs = ["Hello", "Goodbye"]
        targets = ["Hallo", "Auf Wiedersehen"]
        serial_stream = serial_inputs.WMTPreprocess(
            tokenizer=self.tokenizer,
            max_length=50,
            max_eval_length=50,
            training=True,
        )(self._make_serial_stream(inputs, targets)())
        serial_results = list(serial_stream)

        self.assertLen(serial_results, len(inputs))
        for (serial_inputs_arr, serial_targets_arr), text_input, text_target in zip(
            serial_results, inputs, targets
        ):
            np.testing.assert_array_equal(
                serial_inputs_arr,
                np.array(self.tokenizer.encode(text_input), dtype=np.int64),
            )
            np.testing.assert_array_equal(
                serial_targets_arr,
                np.array(self.tokenizer.encode(text_target), dtype=np.int64),
            )

    @parameterized.named_parameters(
        ("train", True),
        ("eval", False),
    )
    def test_wmt_preprocess_filters_by_length(self, training):
        inputs = ["short", "toolong"]
        targets = ["ok", "alsoverylong"]
        serial_stream = serial_inputs.WMTPreprocess(
            tokenizer=self.tokenizer,
            max_length=5,
            max_eval_length=5,
            training=training,
        )(self._make_serial_stream(inputs, targets)())
        serial_results = list(serial_stream)

        self.assertLen(serial_results, 1)
        serial_inputs_arr, serial_targets_arr = serial_results[0]
        np.testing.assert_array_equal(
            serial_inputs_arr,
            np.array(self.tokenizer.encode(inputs[0]), dtype=np.int64),
        )
        np.testing.assert_array_equal(
            serial_targets_arr,
            np.array(self.tokenizer.encode(targets[0]), dtype=np.int64),
        )

    def test_wmt_concat_preprocess_builds_mask(self):
        inputs = ["Hello", "Hi"]
        targets = ["Hallo", "Czesc"]
        serial_stream = serial_inputs.WMTConcatPreprocess(
            tokenizer=self.tokenizer,
            max_length=50,
            max_eval_length=50,
            training=True,
        )(self._make_serial_stream(inputs, targets)())
        serial_results = list(serial_stream)

        self.assertLen(serial_results, len(inputs))
        for (serial_concat, serial_output, serial_mask), text_input, text_target in zip(
            serial_results, inputs, targets
        ):
            inputs_tokens = np.array(
                self.tokenizer.encode(text_input), dtype=np.int64
            )
            targets_tokens = np.array(
                self.tokenizer.encode(text_target), dtype=np.int64
            )
            pad = np.zeros_like(inputs_tokens[:1])
            expected_concat = np.concatenate(
                [inputs_tokens, pad, targets_tokens], axis=0
            )
            expected_mask = np.concatenate(
                [
                    np.zeros_like(inputs_tokens),
                    pad,
                    np.ones_like(targets_tokens),
                ],
                axis=0,
            )

            np.testing.assert_array_equal(serial_concat, expected_concat)
            np.testing.assert_array_equal(serial_output, expected_concat)
            np.testing.assert_array_equal(serial_mask, expected_mask)


if __name__ == "__main__":
    absltest.main()
