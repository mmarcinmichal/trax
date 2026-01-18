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

"""WMT-specific preprocessing pipelines."""

import gin
import numpy as np

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


def _text_length(text):
    if isinstance(text, np.ndarray):
        if text.shape == ():
            text = text.item()
        else:
            return text.shape[0]
    if isinstance(text, np.bytes_):
        text = text.tobytes()
    if isinstance(text, (bytes, str)):
        return len(text)
    return len(text)


@gin.configurable(module="trax.data")
def WMTEnsureInputsTargets():  # pylint: disable=invalid-name
    """Normalizes WMT examples to dictionaries with `inputs` and `targets`."""

    def _ensure(stream):
        for example in stream:
            if isinstance(example, (list, tuple)) and len(example) == 2:
                features, targets = example
                if isinstance(features, dict):
                    normalized = dict(features)
                    fallback_target = targets
                else:
                    normalized = {"inputs": features}
                    fallback_target = targets
                if "translation" in normalized and isinstance(
                    normalized["translation"], dict
                ):
                    translation = normalized["translation"]
                    if "inputs" not in normalized and "en" in translation:
                        normalized["inputs"] = translation["en"]
                    if "targets" not in normalized and "de" in translation:
                        normalized["targets"] = translation["de"]
                if "inputs" not in normalized:
                    if "en" in normalized and "de" in normalized:
                        normalized["inputs"] = normalized["en"]
                    elif "targets" in normalized:
                        normalized["inputs"] = normalized["targets"]
                if "targets" not in normalized:
                    if "de" in normalized:
                        normalized["targets"] = normalized["de"]
                    else:
                        normalized["targets"] = fallback_target
                if "inputs" not in normalized or "targets" not in normalized:
                    raise ValueError(
                        "WMT example missing inputs/targets after normalization: "
                        f"{normalized}"
                    )
                yield normalized
            elif isinstance(example, dict):
                normalized = dict(example)
                if "translation" in normalized and isinstance(
                    normalized["translation"], dict
                ):
                    translation = normalized["translation"]
                    if "inputs" not in normalized and "en" in translation:
                        normalized["inputs"] = translation["en"]
                    if "targets" not in normalized and "de" in translation:
                        normalized["targets"] = translation["de"]
                if "inputs" not in normalized:
                    if "en" in normalized and "de" in normalized:
                        normalized["inputs"] = normalized["en"]
                    elif "targets" in normalized:
                        normalized["inputs"] = normalized["targets"]
                if "targets" not in normalized:
                    if "de" in normalized:
                        normalized["targets"] = normalized["de"]
                if "inputs" not in normalized or "targets" not in normalized:
                    raise ValueError(
                        "WMT example missing inputs/targets after normalization: "
                        f"{normalized}"
                    )
                yield normalized
            else:
                raise ValueError(f"Unsupported WMT example type: {type(example)}")

    return _ensure


@gin.configurable(module="trax.data")
def WMTFilterByLength(  # pylint: disable=invalid-name
    max_length=-1, max_eval_length=-1, training=True
):
    """Filters WMT examples by the string lengths of inputs/targets."""

    def _filter(stream):
        max_allowed = max_length if training else max_eval_length
        if max_allowed <= 0:
            for example in stream:
                yield example
            return
        for example in stream:
            max_text_len = max(
                _text_length(example["inputs"]),
                _text_length(example["targets"]),
            )
            if max_text_len < max_allowed + 1:
                yield example

    return _filter


@gin.configurable(module="trax.data")
def WMTTokenize(  # pylint: disable=invalid-name
    tokenizer=gin.REQUIRED, keys=("inputs", "targets")
):
    """Tokenizes WMT string fields using a provided SubwordTextEncoder."""

    def _tokenize(stream):
        for example in stream:
            if not isinstance(example, dict):
                raise ValueError(
                    "WMTTokenize expects dict examples, got: " f"{type(example)}"
                )
            tokenized = dict(example)
            for key in keys:
                if key not in tokenized:
                    continue
                text = _text_to_str(tokenized[key])
                if isinstance(text, np.ndarray):
                    tokenized[key] = text
                else:
                    tokenized[key] = np.array(tokenizer.encode(text), dtype=np.int64)
            yield tokenized

    return _tokenize


@gin.configurable(module="trax.data")
def WMTToInputsTargetsTuple():  # pylint: disable=invalid-name
    """Converts WMT dict examples to (inputs, targets) tuples."""

    def _to_tuple(stream):
        for example in stream:
            if isinstance(example, dict):
                yield example["inputs"], example["targets"]
            elif isinstance(example, (list, tuple)) and len(example) == 2:
                yield tuple(example)
            else:
                raise ValueError(f"Unsupported WMT example type: {type(example)}")

    return _to_tuple


@gin.configurable(module="trax.data")
def WMTConcatInputsTargets():  # pylint: disable=invalid-name
    """Concatenates inputs/targets with a separator pad and builds a mask."""

    def _concat(stream):
        for example in stream:
            if isinstance(example, dict):
                inputs = example["inputs"]
                targets = example["targets"]
            elif isinstance(example, (list, tuple)) and len(example) >= 2:
                inputs, targets = example[0], example[1]
            else:
                raise ValueError(f"Unsupported WMT example type: {type(example)}")
            inputs = np.asarray(inputs)
            targets = np.asarray(targets)
            pad = np.zeros_like(inputs[:1])
            concat = np.concatenate([inputs, pad, targets], axis=0)
            mask = np.concatenate(
                [np.zeros_like(inputs), pad, np.ones_like(targets)], axis=0
            )
            yield concat, concat, mask

    return _concat


@gin.configurable(module="trax.data")
def WMTPreprocess(  # pylint: disable=invalid-name
    tokenizer=gin.REQUIRED,
    max_length=-1,
    max_eval_length=-1,
    training=True,
):
    """Returns a Serial pipeline for WMT preprocessing."""
    return preprocessing_inputs.Serial(
        WMTEnsureInputsTargets(),
        WMTFilterByLength(
            max_length=max_length,
            max_eval_length=max_eval_length,
            training=training,
        ),
        WMTTokenize(tokenizer=tokenizer),
        WMTToInputsTargetsTuple(),
    )


@gin.configurable(module="trax.data")
def WMTConcatPreprocess(  # pylint: disable=invalid-name
    tokenizer=gin.REQUIRED,
    max_length=-1,
    max_eval_length=-1,
    training=True,
):
    """Returns a Serial pipeline for WMT preprocessing with concatenation."""
    return preprocessing_inputs.Serial(
        WMTPreprocess(
            tokenizer=tokenizer,
            max_length=max_length,
            max_eval_length=max_eval_length,
            training=training,
        ),
        WMTConcatInputsTargets(),
    )
