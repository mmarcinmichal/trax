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

"""Tests for BERT preprocessing modules."""

import gin
import numpy as np
import tensorflow as tf

from tests.data.utils import TEST_CORPUS
from trax.data.preprocessing.inputs import NextSentencePrediction
from trax.data.preprocessing.inputs import batcher  # noqa: F401
from trax.data.preprocessing.modules import bert as modules_bert


class InputsBertTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        gin.clear_config()

    def test_create_bert_inputs(self):
        inputs_sentences_1 = [np.array([100, 150, 200])]
        inputs_sentences_2 = [np.array([300, 500])]
        labels = [np.array(1)]

        create_inputs_1 = modules_bert.CreateBertInputs(False)
        create_inputs_2 = modules_bert.CreateBertInputs(True)
        for res in create_inputs_1(zip(inputs_sentences_1, labels)):
            values, segment_embs, _, label, weight = res
            self.assertAllEqual(values, np.array([101, 100, 150, 200, 102]))
            self.assertAllEqual(segment_embs, np.zeros(5))
            self.assertEqual(label, np.int64(1))
            self.assertEqual(weight, np.int64(1))

        for res in create_inputs_2(zip(inputs_sentences_1, inputs_sentences_2, labels)):
            values, segment_embs, _, label, weight = res
            self.assertAllEqual(
                values, np.array([101, 100, 150, 200, 102, 300, 500, 102])
            )
            exp_segment = np.concatenate((np.zeros(5), np.ones(3)))
            self.assertAllEqual(segment_embs, exp_segment)
            self.assertEqual(label, np.int64(1))
            self.assertEqual(weight, np.int64(1))

    def test_bert_next_sentence_prediction_inputs(self):
        tf.random.set_seed(0)
        exp_sent1 = "The woman who died after falling from"
        exp_sent2 = "The woman who died after falling from"
        sent1, sent2, label = next(
            modules_bert.BertNextSentencePredictionInputs(
                "c4/en:2.3.0", data_dir=TEST_CORPUS, train=False, shuffle_size=1
            )()
        )
        print(sent1, sent2, label)

        self.assertIn(exp_sent1, sent1, "exp_sent1 powinien być częścią sent1")
        self.assertIn(exp_sent2, sent1, "exp_sent1 powinien być częścią sent1")
        self.assertIsInstance(label, (bool, np.bool_))

    def test_mask_random_tokens(self):
        """Test only standard tokens.

        This test deals with sentences composed of two parts: [100 CLS tokens, 100
        chosen standard tokens]. CLS is the token that is added at the beginning of
        the sentence and there is only one token in standard scenario. It is never
        masked because it is not a part of the sentence.
        This tests whether mask_random_tokens will:
          - mask only standard tokens
          - mask expected number of tokens (15 percent candidates for masking)
        """
        cls_token = 101
        mask_token = 103
        example_standard_token = 1001
        test_case_row = np.array([cls_token] * 100 + [example_standard_token] * 100)
        test_case = [(test_case_row.copy(),)]

        np.random.seed(0)
        out, original_tokens, token_weights = next(
            modules_bert.mask_random_tokens(test_case)
        )
        # test whether original tokens are unchanged
        self.assertAllEqual(test_case_row, original_tokens)

        self.assertEqual(1, token_weights.sum())
        self.assertEqual(
            15, (token_weights > 0).sum()
        )  # we should have 15 candidates for masking

        # 101 is a special token, so only 1001 should be masked
        self.assertAllEqual(out[:100], test_case_row[:100])

        # Each candidate has 0.8 probability to be masked while others have 0, so
        # no more than 15 tokens with MASK
        self.assertLessEqual((out == mask_token).sum(), 15)

    def test_next_sentence_prediction_parity_with_tf(self):
        examples = [
            {"text": "First example sentence. Second example sentence."},
            {"text": "Alpha sentence. Beta sentence."},
        ]

        outputs = list(
            NextSentencePrediction(text_key="text", buffer_size=1, seed=0)(
                (example for example in examples)
            )
        )

        self.assertLen(outputs, len(examples))
        for sent1, sent2, label in outputs:
            self.assertTrue(sent1)
            self.assertTrue(sent2)
            self.assertIsInstance(label, (bool, np.bool_))


if __name__ == "__main__":
    tf.test.main()
