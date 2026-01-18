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

"""Video-specific preprocessing helpers."""

import gin


@gin.configurable(module="trax.data")
def bair_robot_pushing_hparams(  # pylint: disable=invalid-name
    hparams=None, video_num_input_frames=1, video_num_target_frames=15
):
    """Configures BAIR robot pushing frame counts."""
    if hparams is not None:
        hparams.video_num_input_frames = video_num_input_frames
        hparams.video_num_target_frames = video_num_target_frames
        return None
    return video_num_input_frames, video_num_target_frames
