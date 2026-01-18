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

"""C4-specific preprocessing pipelines."""

import gin
import numpy as np

from trax.data.encoder.encoder import SentencePieceEncoder
from trax.data.preprocessing import inputs as preprocessing_inputs


def _text_to_str(text):
    if isinstance(text, np.ndarray):
        if text.shape == ():
            text = text.item()
        else:
            return text
    if isinstance(text, np.bytes_):
        text = text.tobytes()
    if isinstance(text, bytes):
        return preprocessing_inputs.to_unicode(text)
    if isinstance(text, str):
        return text
    return str(text)


@gin.configurable(module="trax.data")
def C4Tokenize(tokenization=None, spm_path=None):  # pylint: disable=invalid-name
    """Tokenizes C4 examples into (inputs, targets) token arrays."""
    if tokenization == "spc":
        if not spm_path:
            raise ValueError(
                "A valid SentencePiece model path (`spm_path`) must be provided."
            )
        tokenizer = SentencePieceEncoder(spm_path)

        def encode(text):
            return np.array(tokenizer.encode(text), dtype=np.int64)

    else:

        def encode(text):
            return np.array([ord(ch) for ch in text], dtype=np.int64)

    def _tokenize(stream):
        for example in stream:
            if isinstance(example, dict):
                text = example.get("text", example.get("inputs", example.get("targets")))
            elif isinstance(example, (list, tuple)) and example:
                text = example[0]
            else:
                text = example
            text = _text_to_str(text)
            if isinstance(text, np.ndarray):
                tokens = np.asarray(text, dtype=np.int64)
            else:
                tokens = encode(text)
            yield tokens, tokens

    return _tokenize


@gin.configurable(module="trax.data")
def C4Preprocess(  # pylint: disable=invalid-name
    max_target_length=-1, tokenization=None, spm_path=None
):
    """Returns a Serial pipeline for C4 preprocessing."""
    steps = [C4Tokenize(tokenization=tokenization, spm_path=spm_path)]
    if max_target_length and max_target_length > 0:
        steps.append(
            preprocessing_inputs.FilterByLength(
                max_length=max_target_length, length_keys=[0]
            )
        )
    return preprocessing_inputs.Serial(*steps)
