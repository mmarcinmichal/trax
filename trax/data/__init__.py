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








