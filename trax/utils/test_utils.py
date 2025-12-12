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

"""A few utilities for tests."""

import sys

from absl import flags


# pytest doesn't run the test as a main, so it doesn't parse the flags
# so if flags are required in tests, this will ensure that flags are manually
# parsed and the desired flag exists.
def ensure_flag(flag_str):
    try:
        getattr(flags.FLAGS, flag_str)
    except flags.UnparsedFlagAccessError:
        # Manually parse flags; ignore unknown pytest/IDE flags.
        try:
            flags.FLAGS(sys.argv)
        except flags.Error:
            flags.FLAGS([sys.argv[0]], known_only=True)
    except AttributeError:
        # Flag not defined – define a placeholder so tests can proceed.
        flags.DEFINE_string(flag_str, "", f"Auto-added test flag: --{flag_str}")
    finally:
        assert getattr(flags.FLAGS, flag_str)
