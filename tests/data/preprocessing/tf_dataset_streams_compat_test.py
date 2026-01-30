import numpy as np
import tensorflow as tf

from trax import fastmath
from trax import data as trax_data


def _dataset_stream(dataset):
    def gen(_=None):
        for example in fastmath.dataset_as_numpy(dataset):
            yield example

    return gen


def test_cifar10_flatten_no_aug_outputs_flattened_pairs():
    images = np.arange(2 * 2 * 3 * 2, dtype=np.uint8).reshape(2, 2, 2, 3)
    labels = np.array([1, 2], dtype=np.int64)
    train_ds = tf.data.Dataset.from_tensor_slices({"image": images, "label": labels})

    stream = trax_data.Cifar10FlattenNoAugmentation()(_dataset_stream(train_ds)(None))
    example = next(stream)
    expected = images.reshape(2, -1).astype(np.int64)
    np.testing.assert_array_equal(example[0], expected[0])
    np.testing.assert_array_equal(example[1], expected[0])


def test_cifar10_augmentation_flatten_outputs_masked_triplets():
    images = np.arange(2 * 2 * 3 * 2, dtype=np.uint8).reshape(2, 2, 2, 3)
    labels = np.array([1, 2], dtype=np.int64)
    train_ds = tf.data.Dataset.from_tensor_slices({"image": images, "label": labels})

    preprocess = trax_data.Cifar10AugmentationFlatten(training=False)
    stream = preprocess(_dataset_stream(train_ds)(None))
    flat, target, mask = next(stream)

    expected_flat = images[0].reshape(-1).astype(np.int64)
    expected_target = np.concatenate(
        [expected_flat, np.array([labels[0]], dtype=np.int64)], axis=0
    )
    expected_mask = np.concatenate(
        [
            np.zeros_like(expected_flat, dtype=np.float32),
            np.ones(1, dtype=np.float32),
        ],
        axis=0,
    )
    np.testing.assert_array_equal(flat, expected_target)
    np.testing.assert_array_equal(target, expected_target)
    np.testing.assert_array_equal(mask, expected_mask)


def test_imagenet_flatten_outputs_tuple_pairs():
    images = np.arange(2 * 2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 2, 3)
    train_ds = tf.data.Dataset.from_tensor_slices({"image": images})

    stream = trax_data.DictToTuple(keys=("image", "image"))(
        trax_data.DownsampledImagenetFlatten()(_dataset_stream(train_ds)(None))
    )
    example = next(stream)
    expected = images.reshape(2, -1).astype(np.int64)
    np.testing.assert_array_equal(example[0], expected[0])
    np.testing.assert_array_equal(example[1], expected[0])


def test_enwik8_outputs_tuple_pairs():
    targets = np.arange(6, dtype=np.int64).reshape(2, 3)
    train_ds = tf.data.Dataset.from_tensor_slices({"targets": targets})

    stream = trax_data.DictToTuple(keys=("targets", "targets"))(
        _dataset_stream(train_ds)(None)
    )
    example = next(stream)
    np.testing.assert_array_equal(example[0], targets[0])
    np.testing.assert_array_equal(example[1], targets[0])


def test_bair_robot_pushing_outputs_targets_mask():
    inputs = np.arange(2 * 2, dtype=np.int32).reshape(2, 2)
    targets = np.arange(2 * 2, dtype=np.int32).reshape(2, 2) + 10
    train_ds = tf.data.Dataset.from_tensor_slices({"inputs": inputs, "targets": targets})

    preprocess = trax_data.BairRobotPushingPreprocess()
    stream = trax_data.DictToTuple(keys=("targets", "targets", "mask"))(
        preprocess(_dataset_stream(train_ds)(None))
    )
    flat, targets_out, mask = next(stream)

    expected_concat = np.concatenate([inputs[0], targets[0]], axis=0).reshape(-1)
    expected_mask = np.concatenate(
        [np.zeros_like(inputs[0]), np.ones_like(targets[0])], axis=0
    ).reshape(-1)
    np.testing.assert_array_equal(flat, expected_concat)
    np.testing.assert_array_equal(targets_out, expected_concat)
    np.testing.assert_array_equal(mask, expected_mask.astype(np.float32))


def test_lm1b_filter_by_length_passes_examples():
    inputs = np.arange(4, dtype=np.int64).reshape(2, 2)
    targets = np.arange(4, dtype=np.int64).reshape(2, 2) + 5
    train_ds = tf.data.Dataset.from_tensor_slices((inputs, targets))

    preprocess = trax_data.LM1BFilterByLength(
        max_target_length=4, max_eval_target_length=4, training=True
    )
    stream = preprocess(_dataset_stream(train_ds)(None))
    example = next(stream)
    np.testing.assert_array_equal(example[0], inputs[0])
    np.testing.assert_array_equal(example[1], targets[0])


def test_squeeze_targets_reduces_last_axis():
    inputs = np.arange(4, dtype=np.int64).reshape(2, 2)
    targets = np.arange(2, dtype=np.int64).reshape(2, 1)
    train_ds = tf.data.Dataset.from_tensor_slices((inputs, targets))

    stream = trax_data.SqueezeTargets()(_dataset_stream(train_ds)(None))
    features, squeezed = next(stream)
    np.testing.assert_array_equal(features, inputs[0])
    np.testing.assert_array_equal(squeezed, targets[0].reshape(-1))


def test_mnist_dict_to_tuple_outputs():
    images = np.arange(4, dtype=np.int64).reshape(2, 2)
    labels = np.array([1, 2], dtype=np.int64)
    train_ds = tf.data.Dataset.from_tensor_slices({"image": images, "label": labels})

    stream = trax_data.DictToTuple(keys=("image", "label"))(
        _dataset_stream(train_ds)(None)
    )
    example = next(stream)
    np.testing.assert_array_equal(example[0], images[0])
    np.testing.assert_array_equal(example[1], labels[0])
