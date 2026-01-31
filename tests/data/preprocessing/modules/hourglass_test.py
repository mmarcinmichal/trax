import numpy as np
import tensorflow as tf

from trax import fastmath
from trax import data as trax_data


def _dataset_stream(dataset):
    def gen(_=None):
        for example in fastmath.dataset_as_numpy(dataset):
            yield example

    return gen


def test_hourglass_cifar10_pipeline_outputs_flattened_pairs():
    images = np.arange(2 * 2 * 3 * 2, dtype=np.uint8).reshape(2, 2, 2, 3)
    labels = np.array([1, 2], dtype=np.int64)
    train_ds = tf.data.Dataset.from_tensor_slices({"image": images, "label": labels})

    train_stream = trax_data.Cifar10FlattenNoAugmentation()(
        _dataset_stream(train_ds)(None)
    )
    example = next(train_stream)
    expected = images.reshape(2, -1).astype(np.int64)
    np.testing.assert_array_equal(example[0], expected[0])
    np.testing.assert_array_equal(example[1], expected[0])


def test_hourglass_imagenet_pipeline_outputs_flattened_pairs():
    images = np.arange(2 * 2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 2, 3)
    train_ds = tf.data.Dataset.from_tensor_slices({"image": images})

    train_stream = trax_data.DictToTuple(keys=("image", "image"))(
        trax_data.DownsampledImagenetFlatten()(_dataset_stream(train_ds)(None))
    )
    example = next(train_stream)
    expected = images.reshape(2, -1).astype(np.int64)
    np.testing.assert_array_equal(example[0], expected[0])
    np.testing.assert_array_equal(example[1], expected[0])


def test_hourglass_enwik8_pipeline_outputs_targets_pairs():
    targets = np.arange(6, dtype=np.int64).reshape(2, 3)
    train_ds = tf.data.Dataset.from_tensor_slices({"targets": targets})

    train_stream = trax_data.DictToTuple(keys=("targets", "targets"))(
        _dataset_stream(train_ds)(None)
    )
    example = next(train_stream)
    np.testing.assert_array_equal(example[0], targets[0])
    np.testing.assert_array_equal(example[1], targets[0])
