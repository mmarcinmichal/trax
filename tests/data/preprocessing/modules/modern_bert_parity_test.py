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

"""Parity tests against ModernBERT sequence packer."""

import importlib.util
import os

import numpy as np
import tensorflow as tf

from absl.testing import absltest

from trax import data


def _load_modernbert_sequence_packer():
    root = "/mnt/d/Projects/trax/resources/ModernBERT-main/src/sequence_packer.py"
    if not os.path.exists(root):
        return None
    spec = importlib.util.spec_from_file_location("modernbert_sequence_packer", root)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError:
        return None
    return module


class ModernBertParityTest(absltest.TestCase):
    def test_sequence_packer_parity(self):
        module = _load_modernbert_sequence_packer()
        if module is None:
            self.skipTest("ModernBERT sequence_packer.py not found.")

        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not available.")

        # Build a deterministic batch of unpadded sequences.
        seqs = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5]},
            {"input_ids": [6, 7, 8, 9]},
            {"input_ids": [10]},
        ]
        src_iterable = [seqs]

        mb_packer = module.GreedyBestFitSequencePacker.from_composer(
            src_iterable=src_iterable,
            batch_size=4,
            micro_batch_size=2,
            max_seq_len=4,
            buffer_size=10,
            pad_token_id=0,
            mask_token_id=99,
            suppress_masking=True,
        )
        mb_out = next(iter(mb_packer))
        mb_packed = mb_out["input_ids"].cpu().numpy()
        mb_cu = [x.cpu().numpy() for x in mb_out["cu_seqlens"]]

        pad_len = max(len(x) for x in mb_cu)
        mb_cu_padded = np.full((len(mb_cu), pad_len), -1, dtype=np.int32)
        for i, entry in enumerate(mb_cu):
            mb_cu_padded[i, : len(entry)] = entry

        trax_stream = iter([seqs])
        trax_packer = data.ModernBertSequencePacker(
            src_batch_size=4,
            src_max_seq_len=4,
            micro_batch_size=2,
            pad_token_id=0,
            mask_token_id=99,
            suppress_masking=True,
            pad_cu_seqlens_to=pad_len,
        )
        trax_out = next(iter(trax_packer(trax_stream)))
        trax_packed = trax_out["input_ids"]
        trax_cu = trax_out["cu_seqlens"]

        np.testing.assert_array_equal(trax_packed, mb_packed)
        np.testing.assert_array_equal(trax_cu, mb_cu_padded)


if __name__ == "__main__":
    tf.test.main()
