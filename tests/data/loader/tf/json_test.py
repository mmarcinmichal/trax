# coding=utf-8
"""Tests for the raw_json local loader integration via TFDS() wrapper."""
import gin
import tensorflow as tf

from tests.data.utils import TEST_CORPUS  # existing test helper
from trax.data.loader.tf import base as ds


class TFRawJsonTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        gin.clear_config()

    def test_raw_json_train_loads(self):
        """Ensure TFDS('raw_json:...') returns a working train generator."""
        train_ds_gen = ds.TFDS(
            "raw_json:testdataset",
            data_dir=TEST_CORPUS,
            train=True,
            host_id=0,
            keys=("inputs",),
            n_hosts=1,
        )

        # iterate a bit to ensure it doesn't crash and returns examples
        seen = False
        try:
            for example in train_ds_gen():
                seen = True
                break
        except Exception as e:
            self.fail(f"test_raw_json_train_loads() raised unexpected exception: {e}")

        self.assertTrue(seen, "Train generator did not yield any examples.")

    def test_raw_json_eval_loads(self):
        """Ensure TFDS('raw_json:...') returns a working eval generator."""
        eval_ds_gen = ds.TFDS(
            "raw_json:testdataset",
            data_dir=TEST_CORPUS,
            train=False,
            host_id=0,
            keys=("inputs", "targets"),
            n_hosts=1,
        )

        seen = False
        try:
            for example in eval_ds_gen():
                seen = True
                break
        except Exception as e:
            self.fail(f"test_raw_json_eval_loads() raised unexpected exception: {e}")

        self.assertTrue(seen, "Eval generator did not yield any examples.")
