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

"""ImageNet-specific preprocessing helpers."""

import gin
import numpy as np


@gin.configurable(module="trax.data")
def DownsampledImagenetFlatten(  # pylint: disable=invalid-name
    image_key="image",
):
    """Flattens ImageNet image tensors into 1D int64 vectors."""

    def _flatten(generator):
        for example in generator:
            if not isinstance(example, dict):
                raise ValueError("DownsampledImagenetFlatten expects dict examples.")
            img = np.asarray(example[image_key])
            flat = img.reshape(-1).astype(np.int64)
            updated = dict(example)
            updated[image_key] = flat
            yield updated

    return _flatten
