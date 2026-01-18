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

"""Functions and classes for obtaining and preprocesing data.

The ``trax.data`` module presents a flattened (no subpackages) public API.
(Many of the functions and class initilizers in the API are also accessible for
gin configuration.) To use as a client, import ``trax.data`` and access
functions using ``data.foo`` qualified names; for example::

   from trax import data
   ...
   training_inputs = data.Serial(
     ...
     data.Tokenize(),
     data.Shuffle(),
     ...
  )

"""

from trax.data.preprocessing import inputs as inputs  # noqa: F401
from trax.data.preprocessing.modules import *  # pylint: disable=wildcard-import

# Re-export commonly-used preprocessing utilities at the module level to keep
# the historical ``trax.data.*`` namespace working (e.g. gin configs that refer
# to ``trax.data.inputs.Serial`` or ``trax.data.Batch``).
from trax.data.preprocessing.inputs import *  # pylint: disable=wildcard-import

__all__ = ["inputs"]
__all__ += [
    name
    for name in dir(inputs)
    if not name.startswith("_") and name not in __all__
]
__all__ += [
    name
    for name in dir()
    if not name.startswith("_") and name not in __all__
]






