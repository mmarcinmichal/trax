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

"""Tests for trax.data.tf.datasets."""
from unittest import mock

import gin
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tests.data.utils import TEST_CORPUS  # relative import
from trax.data.loader.tf import base as ds
from trax.data.preprocessing import inputs
from trax.data.preprocessing.inputs import batcher  # noqa: F401
from trax.data.preprocessing.tf import inputs as tf_inputs


class TFDatasetTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        gin.clear_config()

    def test_TFDS_single_host_with_eval_holdout(self):
        train_ds_gen = ds.TFDS(
            "c4/en:2.3.0",
            data_dir=TEST_CORPUS,
            train=True,
            host_id=0,
            keys=("text",),
            n_hosts=1,
            eval_holdout_size=0.1,
        )

        result = None

        try:
            # Just ensure that this doesn't crash.
            for d in train_ds_gen():
                break

            result = True
        except Exception as e:
            self.fail(
                f"test_TFDS_single_host_with_eval_holdout() raised an unexpected exception: {e}"
            )

        self.assertIsNotNone(
            result,
            "test_TFDS_single_host_with_eval_holdout() returned None unexpectedly",
        )

        valid_ds_gen = ds.TFDS(
            "c4/en:2.3.0",
            data_dir=TEST_CORPUS,
            train=False,
            host_id=0,
            keys=("text",),
            n_hosts=1,
            eval_holdout_size=0.1,
        )

        result = None

        try:
            # Just ensure that this doesn't crash.
            for d in valid_ds_gen():
                break

            result = True
        except Exception as e:
            self.fail(
                f"test_TFDS_single_host_with_eval_holdout() raised an unexpected exception: {e}"
            )

        self.assertIsNotNone(
            result,
            "test_TFDS_single_host_with_eval_holdout() returned None unexpectedly",
        )

    def test_TFDS_single_host_with_eval_holdout_no_valid_split(self):
        train_ds_gen = ds.TFDS(
            "para_crawl/ende",
            data_dir=TEST_CORPUS,
            train=True,
            host_id=0,
            keys=("en", "de"),
            n_hosts=1,
            eval_holdout_size=0.1,
        )

        result = None

        try:
            # Just ensure that this doesn't crash.
            for d in train_ds_gen():
                break

            result = True
        except Exception as e:
            self.fail(
                f"test_TFDS_single_host_with_eval_holdout() raised an unexpected exception: {e}"
            )

        self.assertIsNotNone(
            result,
            "test_TFDS_single_host_with_eval_holdout() returned None unexpectedly",
        )

        # para_crawl doesn't have a validation set, see that this still doesn't
        # crash because of eval_holdout_set.
        valid_ds_gen = ds.TFDS(
            "para_crawl/ende",
            data_dir=TEST_CORPUS,
            train=False,
            host_id=0,
            keys=("en", "de"),
            n_hosts=1,
            eval_holdout_size=0.1,
        )

        result = None

        try:
            # Just ensure that this doesn't crash.
            for d in valid_ds_gen():
                break
            result = True
        except Exception as e:
            self.fail(
                f"test_TFDS_single_host_with_eval_holdout() raised an unexpected exception: {e}"
            )

        self.assertIsNotNone(
            result,
            "test_TFDS_single_host_with_eval_holdout() returned None unexpectedly",
        )

    def test_TFDS_mnli_split_is_eval(self):
        with mock.patch("tensorflow_datasets.load") as tfds_load:
            with mock.patch(
                "trax.data.loader.tf.interface.download_and_prepare",
                lambda _, data_dir: data_dir,
            ):
                _ = ds.TFDS("glue/mnli", keys=("premise", "hypothesis"), train=False)
            call_kwargs = tfds_load.call_args[1]
            self.assertEqual(call_kwargs["split"], "validation_matched")

    def test_TFDS_mnli_split_is_alt_eval(self):
        with mock.patch("tensorflow_datasets.load") as tfds_load:
            with mock.patch(
                "trax.data.loader.tf.interface.download_and_prepare",
                lambda _, data_dir: data_dir,
            ):
                _ = ds.TFDS(
                    "glue/mnli",
                    keys=("premise", "hypothesis"),
                    train=False,
                    use_alt_eval=True,
                )
            call_kwargs = tfds_load.call_args[1]
            self.assertEqual(call_kwargs["split"], "validation_mismatched")

    def test_data_streams_returns_dataset_streams(self):
        streams = ds.data_streams("c4/en:2.3.0", data_dir=TEST_CORPUS, download=False)
        self.assertIsNotNone(streams.train)
        self.assertIsNotNone(streams.eval)
        # Some datasets (like C4) are unsupervised and do not expose keys.
        self.assertIn(streams.supervised_keys, (None, (["inputs"], ["targets"])))

    def test_tf_dataset_streams_seeded_shuffle(self):
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "inputs": np.arange(5, dtype=np.int64),
                "targets": np.arange(5, dtype=np.int64),
            }
        )
        datasets = (dataset, dataset, (["inputs"], ["targets"]))

        train_stream1, _ = tf_inputs.tf_dataset_streams(
            datasets=datasets,
            shuffle_buffer_size=5,
            seed=123,
        )
        train_stream2, _ = tf_inputs.tf_dataset_streams(
            datasets=datasets,
            shuffle_buffer_size=5,
            seed=123,
        )

        first1 = next(train_stream1())
        first2 = next(train_stream2())
        np.testing.assert_array_equal(first1[0], first2[0])
        np.testing.assert_array_equal(first1[1], first2[1])


if __name__ == "__main__":
    tf.test.main()
