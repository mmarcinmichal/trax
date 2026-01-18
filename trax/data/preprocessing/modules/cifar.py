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

"""CIFAR-specific preprocessing pipelines."""

import gin
import numpy as np
import tensorflow as tf


def _extract_cifar_example(example):
    if isinstance(example, dict):
        image = example.get("image", example.get("inputs"))
        label = example.get("label", example.get("targets"))
    elif isinstance(example, (list, tuple)) and len(example) == 2:
        features, label = example
        if isinstance(features, dict):
            image = features.get("image", features.get("inputs"))
        else:
            image = features
    else:
        raise ValueError(f"Unsupported CIFAR example type: {type(example)}")
    if image is None or label is None:
        raise ValueError(f"CIFAR example missing image/label: {example}")
    return image, label


def _cifar_augment_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    return image.numpy()


@gin.configurable(module="trax.data")
def Cifar10NoAugmentation():  # pylint: disable=invalid-name
    """Serial-friendly preprocessing for CIFAR-10 without augmentation."""

    def _process(stream):
        for example in stream:
            image, label = _extract_cifar_example(example)
            image = np.asarray(image, dtype=np.float32) / 255.0
            yield image, np.asarray(label)

    return _process


@gin.configurable(module="trax.data")
def Cifar10Augmentation(training=True):  # pylint: disable=invalid-name
    """Serial-friendly preprocessing for CIFAR-10 with augmentation."""

    def _process(stream):
        for example in stream:
            image, label = _extract_cifar_example(example)
            if training:
                image = _cifar_augment_image(image)
            image = np.asarray(image, dtype=np.float32) / 255.0
            yield image, np.asarray(label)

    return _process


@gin.configurable(module="trax.data")
def Cifar10AugmentationFlatten(  # pylint: disable=invalid-name
    training=True, predict_image_train_weight=0.01
):
    """Serial-friendly preprocessing that flattens CIFAR-10 and appends targets."""

    def _process(stream):
        for example in stream:
            image, label = _extract_cifar_example(example)
            if training:
                image = _cifar_augment_image(image)
            flat = np.asarray(image).reshape(-1).astype(np.int64)
            tgt = np.asarray(label).reshape(1).astype(np.int64)
            flat_with_target = np.concatenate([flat, tgt], axis=0)
            predict_image_weight = predict_image_train_weight if training else 0.0
            mask_begin = np.ones_like(flat, dtype=np.float32) * predict_image_weight
            mask_end = np.ones_like(tgt, dtype=np.float32)
            mask = np.concatenate([mask_begin, mask_end], axis=0)
            yield flat_with_target, flat_with_target, mask

    return _process


@gin.configurable(module="trax.data")
def Cifar10FlattenNoAugmentation(training=True):  # pylint: disable=invalid-name
    """Serial-friendly preprocessing that flattens CIFAR-10 without augmentation."""
    del training

    def _process(stream):
        for example in stream:
            image, _ = _extract_cifar_example(example)
            flat = np.asarray(image).reshape(-1).astype(np.int64)
            yield flat, flat

    return _process
