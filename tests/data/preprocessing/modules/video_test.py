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

"""Tests for video preprocessing modules."""

import tensorflow as tf

from trax.data.preprocessing.modules import video as modules_video


class _DummyHParams:
    def __init__(self):
        self.video_num_input_frames = None
        self.video_num_target_frames = None


class VideoPreprocessingTest(tf.test.TestCase):
    def test_bair_robot_pushing_hparams_updates_hparams(self):
        input_frames = 2
        target_frames = 3
        hparams = _DummyHParams()

        result = modules_video.bair_robot_pushing_hparams(
            hparams=hparams,
            video_num_input_frames=input_frames,
            video_num_target_frames=target_frames,
        )

        self.assertIsNone(result)
        self.assertEqual(hparams.video_num_input_frames, input_frames)
        self.assertEqual(hparams.video_num_target_frames, target_frames)

    def test_bair_robot_pushing_hparams_returns_tuple(self):
        result = modules_video.bair_robot_pushing_hparams(
            hparams=None, video_num_input_frames=4, video_num_target_frames=5
        )
        self.assertEqual(result, (4, 5))


if __name__ == "__main__":
    tf.test.main()
