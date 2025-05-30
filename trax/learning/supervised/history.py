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

"""Trax history."""
import collections
import copy

import six

from absl import logging


class History:
    """History of metrics.

    History contains the metrics recorded during training and evaluation.
    Save data with history.append and get a sequence of data by calling
    history.get.

    For example:
    history.append('train', 'metrics/accuracy', 1, 0.04)
    history.append('train', 'metrics/accuracy', 1000, 0.31)
    history.get('train', 'metrics/accuracy')
    # returns [(1, 0.04), (1000, 0.31)]
    """

    def __init__(self):
        # Structure is
        # values = {
        #   'mode1': {
        #     'metric1': [val1, val2],
        #     ...
        #   },
        #   'mode2': ...
        # }
        self._values = {}

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `History` from a Python dictionary of parameters."""
        history = History()
        for key, value in six.iteritems(json_object):
            history.__dict__[key] = value
        return history

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def append(self, mode, metric, step, value):
        """Append (step, value) pair to history for the given mode and metric."""
        if mode not in self._values:
            self._values[mode] = collections.defaultdict(list)
        self._values[mode][metric].append((step, value))

    def get(self, mode, metric):
        """Get the history for the given metric and mode."""
        if mode not in self._values:
            logging.info("Metric %s not found for mode %s", metric, mode)
            return []
        return list(self._values[mode][metric])

    @property
    def modes(self):
        """Current tracked modes."""
        return sorted(list(self._values.keys()))

    def metrics_for_mode(self, mode):
        """Metrics available for a given mode."""
        if mode not in self._values:
            logging.info("Mode %s not found", mode)
            return []
        return sorted(list(self._values[mode].keys()))

    def __str__(self):
        return str(self._values)
