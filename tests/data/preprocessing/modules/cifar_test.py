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

"""Tests for CIFAR preprocessing modules."""

from unittest import mock

import numpy as np
import tensorflow as tf

from trax.data.preprocessing.modules import cifar as modules_cifar


class CifarPreprocessingTest(tf.test.TestCase):
    def _build_dataset(self):
        images = np.arange(2 * 32 * 32 * 3, dtype=np.uint8).reshape(2, 32, 32, 3)
        labels = np.arange(2, dtype=np.int64)
        dataset = tf.data.Dataset.from_tensor_slices(({"image": images}, labels))
        return images, labels, dataset

    def _run_serial(self, pipeline, images, labels):
        tf.random.set_seed(0)
        generator = (({"image": image}, label) for image, label in zip(images, labels))
        return list(pipeline(generator))

    def test_cifar10_no_augmentation(self):
        images, labels, _ = self._build_dataset()

        serial_outputs = self._run_serial(
            modules_cifar.Cifar10NoAugmentation(), images, labels
        )

        self.assertLen(serial_outputs, len(labels))
        for (image, label), expected_label in zip(serial_outputs, labels):
            self.assertEqual(image.shape, (32, 32, 3))
            self.assertEqual(label, expected_label)

    def test_cifar10_augmentation(self):
        images, labels, _ = self._build_dataset()

        def deterministic_crop(image, size):
            height, width, _ = size
            offset_h = (tf.shape(image)[0] - height) // 2
            offset_w = (tf.shape(image)[1] - width) // 2
            return tf.image.crop_to_bounding_box(
                image, offset_h, offset_w, height, width
            )

        with mock.patch("tensorflow.image.random_crop", side_effect=deterministic_crop):
            with mock.patch(
                "tensorflow.image.random_flip_left_right",
                side_effect=lambda image: image,
            ):
                serial_outputs = self._run_serial(
                    modules_cifar.Cifar10Augmentation(training=True), images, labels
                )

        self.assertLen(serial_outputs, len(labels))
        for (image, label), expected_label in zip(serial_outputs, labels):
            self.assertEqual(image.shape, (32, 32, 3))
            self.assertEqual(label, expected_label)

    def test_cifar10_augmentation_flatten(self):
        images, labels, _ = self._build_dataset()

        def deterministic_crop(image, size):
            height, width, _ = size
            offset_h = (tf.shape(image)[0] - height) // 2
            offset_w = (tf.shape(image)[1] - width) // 2
            return tf.image.crop_to_bounding_box(
                image, offset_h, offset_w, height, width
            )

        with mock.patch("tensorflow.image.random_crop", side_effect=deterministic_crop):
            with mock.patch(
                "tensorflow.image.random_flip_left_right",
                side_effect=lambda image: image,
            ):
                serial_outputs = self._run_serial(
                    modules_cifar.Cifar10AugmentationFlatten(training=True),
                    images,
                    labels,
                )

        self.assertLen(serial_outputs, len(labels))
        for flat, targets, mask in serial_outputs:
            self.assertEqual(flat.shape, (32 * 32 * 3 + 1,))
            self.assertEqual(targets.shape, flat.shape)
            self.assertEqual(mask.shape, flat.shape)


if __name__ == "__main__":
    tf.test.main()
