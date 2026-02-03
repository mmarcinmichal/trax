# coding=utf-8
# Copyright 2026 The Trax Authors.
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

"""Tests for ModernBERT preprocessing components."""

import numpy as np
import tensorflow as tf

from absl.testing import absltest

from trax import data


class ModernBertPreprocessTest(absltest.TestCase):
    def test_chunk_fixed_length(self):
        stream = iter(
            [
                {"input_ids": np.asarray([1, 2], dtype=np.int32)},
                {"input_ids": np.asarray([3, 4, 5], dtype=np.int32)},
            ]
        )
        pipeline = data.Serial(
            data.AppendDocBoundaryTokens(eos_tokens=[9]),
            data.ChunkFixedLength(max_seq_len=4, no_wrap=False),
        )
        outputs = list(pipeline(stream))
        self.assertLen(outputs, 1)
        np.testing.assert_array_equal(
            outputs[0]["input_ids"], np.asarray([1, 2, 9, 3], dtype=np.int32)
        )

    def test_chunk_fixed_length_no_wrap(self):
        stream = iter(
            [
                {"input_ids": np.asarray([1, 2, 3, 4, 5], dtype=np.int32)},
                {"input_ids": np.asarray([6, 7, 8], dtype=np.int32)},
            ]
        )
        pipeline = data.Serial(
            data.AppendDocBoundaryTokens(eos_tokens=[9]),
            data.ChunkFixedLength(max_seq_len=4, no_wrap=True),
        )
        outputs = list(pipeline(stream))
        self.assertLen(outputs, 2)
        np.testing.assert_array_equal(outputs[0]["input_ids"], [1, 2, 3, 4])
        np.testing.assert_array_equal(outputs[1]["input_ids"], [6, 7, 8, 9])

    def test_batch_dict_padding(self):
        stream = iter(
            [
                {"input_ids": np.asarray([1, 2, 3], dtype=np.int32)},
                {"input_ids": np.asarray([4, 5], dtype=np.int32)},
            ]
        )
        batcher = data.BatchDict(
            batch_size=2, keys=("input_ids",), pad_to=4, pad_value={"input_ids": 0}
        )
        outputs = list(batcher(stream))
        self.assertLen(outputs, 1)
        batch = outputs[0]["input_ids"]
        self.assertEqual(batch.shape, (2, 4))
        np.testing.assert_array_equal(batch[0], [1, 2, 3, 0])
        np.testing.assert_array_equal(batch[1], [4, 5, 0, 0])

    def test_sequence_packer_fixed_length(self):
        batch = np.asarray(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            dtype=np.int32,
        )
        stream = iter([{"input_ids": batch}])
        packer = data.ModernBertSequencePacker(
            src_batch_size=4,
            src_max_seq_len=4,
            micro_batch_size=2,
            pad_token_id=0,
            mask_token_id=99,
            suppress_masking=True,
            pad_cu_seqlens_to=3,
        )
        outputs = list(packer(stream))
        self.assertLen(outputs, 1)
        out = outputs[0]
        packed = out["input_ids"]
        self.assertEqual(packed.shape, (2, 8))
        np.testing.assert_array_equal(packed[0], [1, 2, 3, 4, 5, 6, 7, 8])
        np.testing.assert_array_equal(packed[1], [9, 10, 11, 12, 13, 14, 15, 16])
        np.testing.assert_array_equal(out["cu_seqlens"], [[0, 4, 8], [0, 4, 8]])
        np.testing.assert_array_equal(out["max_seqlen"], [4, 4])
        self.assertIsNone(out["labels"])
        self.assertNotIn("attention_mask", out)

    def test_add_position_ids_and_tuple(self):
        stream = iter(
            [
                {
                    "input_ids": np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
                    "labels": np.asarray([[0, 0, 0], [0, 0, 0]], dtype=np.int32),
                    "attention_mask": np.ones((2, 3), dtype=np.int64),
                    "cu_seqlens": np.asarray([[0, 3], [0, 3]], dtype=np.int32),
                    "max_seqlen": np.asarray([3, 3], dtype=np.int32),
                }
            ]
        )
        pipeline = data.Serial(
            data.AddPositionIds(),
            data.ToModelTuple(
                inputs_keys=(
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "cu_seqlens",
                    "max_seqlen",
                )
            ),
        )
        outputs = list(pipeline(stream))
        self.assertLen(outputs, 1)
        inputs, labels = outputs[0]
        self.assertEqual(len(inputs), 5)
        self.assertEqual(labels.shape, (2, 3))
        position_ids = inputs[2]
        np.testing.assert_array_equal(position_ids[0], [0, 1, 2])
        np.testing.assert_array_equal(position_ids[1], [0, 1, 2])

    def test_mlm_masking_random_range(self):
        batch = np.asarray([[1, 2, 3, 4]], dtype=np.int32)
        stream = iter([{"input_ids": batch}])
        packer = data.ModernBertSequencePacker(
            src_batch_size=1,
            src_max_seq_len=4,
            micro_batch_size=1,
            pad_token_id=0,
            mask_token_id=99,
            mask_prob=1.0,
            seed=0,
            suppress_masking=False,
            pad_cu_seqlens_to=2,
        )
        out = next(iter(packer(stream)))
        masked = out["input_ids"]
        self.assertTrue(np.all(masked >= 0))
        self.assertTrue(np.all(masked <= np.max(masked)))
        labels = out["labels"]
        self.assertEqual(labels.shape, masked.shape)

    def test_batch_dict_list(self):
        stream = iter(
            [
                {"input_ids": np.asarray([1, 2], dtype=np.int32)},
                {"input_ids": np.asarray([3, 4, 5], dtype=np.int32)},
                {"input_ids": np.asarray([6], dtype=np.int32)},
            ]
        )
        outputs = list(data.BatchDictList(batch_size=2, drop_last=False)(stream))
        self.assertLen(outputs, 2)
        self.assertIsInstance(outputs[0], list)
        self.assertLen(outputs[0], 2)
        self.assertLen(outputs[1], 1)

    def test_end_to_end_sample_text(self):
        text = (
            "Wikipedia is a free online encyclopedia that is written and maintained "
            "by a community of volunteers."
        )
        stream = iter([{"text": text}])

        def _encode(text_value):
            return [ord(c) % 97 for c in text_value if c.isascii()]

        pipeline = data.Serial(
            data.SelectTextField(field="text", output_key="text"),
            data.TokenizerEncode(encode_fn=_encode, input_key="text", output_key="input_ids"),
            data.AppendDocBoundaryTokens(eos_tokens=[99]),
            data.ChunkFixedLength(max_seq_len=16, no_wrap=False),
            data.BatchDictList(batch_size=4, drop_last=False),
            data.ModernBertSequencePacker(
                src_batch_size=4,
                src_max_seq_len=16,
                micro_batch_size=2,
                pad_token_id=0,
                mask_token_id=99,
                suppress_masking=True,
                pad_cu_seqlens_to=3,
            ),
            data.AddPositionIds(),
            data.ToModelTuple(
                inputs_keys=(
                    "input_ids",
                    "position_ids",
                    "cu_seqlens",
                    "max_seqlen",
                )
            ),
        )
        outputs = list(pipeline(stream))
        self.assertGreater(len(outputs), 0)
        inputs, labels = outputs[0]
        self.assertIsNone(labels)
        self.assertEqual(len(inputs), 4)
        self.assertEqual(inputs[0].shape[-1], 32)
        self.assertEqual(inputs[1].shape, inputs[0].shape)


if __name__ == "__main__":
    tf.test.main()
