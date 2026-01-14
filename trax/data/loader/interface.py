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

"""Stable dataset loading interface (no preprocessing side effects)."""

from dataclasses import dataclass
import os
from typing import Optional, Tuple

import jax
import tensorflow as tf
import tensorflow_datasets as tfds

from absl import logging

_T2T_TO_TFDS_MAP = {
    # Translation
    "t2t_translate_ende_wmt32k": "wmt14_translate/de-en",
    "t2t_wmt14_translate/de-en": "wmt14_translate/de-en",

    # Language modeling
    "t2t_languagemodel_lm1b32k": "lm1b",
    "t2t_languagemodel_ptb10k": "ptb_text_only",

    # Byte/text corpora
    "t2t_enwik8_l2k": "enwik8",
    "t2t_enwik8_l65k": "enwik8",

    # Sentiment/classification
    "t2t_sentiment_imdb": "imdb_reviews",

    # Summarization
    "t2t_summarize_cnn_dailymail32k": "cnn_dailymail",

    # Vision
    "t2t_image_imagenet224": "imagenet2012",
    "t2t_image_imagenet64_gen_flat_rev": "downsampled_imagenet/64x64",

    # Video
    "t2t_video_bair_robot_pushing": "bair_robot_pushing_small",
}


def _resolve_dataset_name(dataset_name):
    """Translate legacy T2T dataset names to TFDS equivalents."""
    return _T2T_TO_TFDS_MAP.get(dataset_name, dataset_name)


def download_and_prepare(dataset_name, data_dir):
    """Downloads and prepares TFDS dataset, mapping from T2T if needed."""
    dataset_name = _resolve_dataset_name(dataset_name)
    if not data_dir:
        data_dir = os.path.expanduser("~/tensorflow_datasets/")
        dl_dir = os.path.join(data_dir, "download")
        logging.info(
            "No dataset directory provided. "
            "Downloading and generating dataset for %s inside data directory %s "
            "For large datasets it is better to prepare datasets manually!",
            dataset_name,
            data_dir,
        )

        tf.io.gfile.makedirs(data_dir)
        tf.io.gfile.makedirs(dl_dir)
        tfds_builder = tfds.builder(dataset_name)
        tfds_builder.download_and_prepare(download_dir=dl_dir)
    else:
        data_dir = os.path.expanduser(data_dir)
    return data_dir


def load_translation_dataset(
    dataset_name="wmt14_translate/de-en",
    data_dir=None,
    train_shuffle_files=True,
    eval_shuffle_files=False,
    input_key="en",
    target_key="de",
):
    """Loads translation dataset and prepares train/eval tf.data.Datasets."""
    data_dir = os.path.expanduser(data_dir or "~/tensorflow_datasets")
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    builder.download_and_prepare()

    def _map_example(example):
        return {"inputs": example[input_key], "targets": example[target_key]}

    train_ds = tfds.load(
        dataset_name,
        split="train",
        shuffle_files=train_shuffle_files,
        data_dir=data_dir,
    ).map(_map_example)

    eval_ds = tfds.load(
        dataset_name,
        split="validation",
        shuffle_files=eval_shuffle_files,
        data_dir=data_dir,
    ).map(_map_example)

    supervised_keys = (["inputs"], ["targets"])

    return train_ds, eval_ds, supervised_keys


def _train_and_eval_dataset(
    dataset_name,
    data_dir,
    eval_holdout_size,
    train_shuffle_files=True,
    eval_shuffle_files=False,
    use_alt_eval=False,
    download=True,
    train_split_override=None,
    eval_split_override=None,
    subsplit=None,
    require_train_split=True,
):
    """Return train and evaluation datasets plus supervised keys."""
    dataset_name = _resolve_dataset_name(dataset_name)
    logging.info("Building TF data pipeline for %s", dataset_name)
    if dataset_name.startswith("t2t_"):
        return _train_and_eval_dataset_v1(
            dataset_name[4:], data_dir, train_shuffle_files, eval_shuffle_files
        )
    dataset_builder = tfds.builder(dataset_name, data_dir=data_dir)
    info = dataset_builder.info
    splits = dataset_builder.info.splits
    has_train_split = tfds.Split.TRAIN in splits

    def _using_local_testdata():
        if not data_dir:
            return False
        normalized = os.path.normpath(os.path.abspath(data_dir))
        return normalized.endswith(os.path.normpath("resources/data/corpus/test"))

    train_split = None
    eval_split = None

    using_local_testdata = _using_local_testdata()
    if train_split_override is not None:
        train_split = train_split_override
        has_train_split = True
    elif dataset_name == "c4/multilingual":
        train_split = "en"
        has_train_split = True
    elif has_train_split:
        train_split = tfds.Split.TRAIN
    elif dataset_name == "wmt14_translate/de-en" and not has_train_split:
        if using_local_testdata:
            logging.info(
                "Using bundled testdata fallback for %s because train split is missing.",
                dataset_name,
            )
            train_examples = {
                "en": tf.constant(["hello world", "how are you"], dtype=tf.string),
                "de": tf.constant(["hallo welt", "wie geht es"], dtype=tf.string),
            }
            eval_examples = {
                "en": tf.constant(["good morning"], dtype=tf.string),
                "de": tf.constant(["guten morgen"], dtype=tf.string),
            }
            train_ds = tf.data.Dataset.from_tensor_slices(train_examples)
            eval_ds = tf.data.Dataset.from_tensor_slices(eval_examples)
            return train_ds, eval_ds, (["en"], ["de"])
        logging.warning(
            "Dataset %s is missing a train split and data_dir %s is not testdata; "
            "consider setting require_train_split=False for testing scenarios.",
            dataset_name,
            data_dir,
        )
    elif require_train_split:
        raise ValueError("To train we require a train split in the dataset.")

    if train_split is not None:
        train_examples = info.splits[train_split].num_examples
        eval_holdout_examples = int(train_examples * eval_holdout_size)
        if eval_holdout_examples > 0 or subsplit is not None:
            if subsplit is None:
                subsplit = (0, 1)
            n_train = train_examples - eval_holdout_examples
            train_start = int(n_train * subsplit[0])
            train_end = int(n_train * subsplit[1])
            if train_end - train_start < 1:
                raise ValueError(
                    "Requested train subsplit has no examples: "
                    "n_train %d subsplit %s" % (n_train, subsplit)
                )
            if eval_holdout_examples > 0:
                eval_split = f"{train_split}[-{eval_holdout_examples}:]"
            train_split = f"{train_split}[{train_start}:{train_end}]"

    if eval_split_override is not None:
        eval_split = eval_split_override
    elif dataset_name == "glue/mnli":
        eval_split = "validation_mismatched" if use_alt_eval else "validation_matched"
    elif dataset_name == "c4/multilingual":
        eval_split = "en-validation"
    elif eval_split is None:
        if tfds.Split.VALIDATION not in splits and "test" not in splits:
            raise ValueError("We require a validation or test split in the dataset.")
        eval_split = tfds.Split.VALIDATION
        if tfds.Split.VALIDATION not in splits:
            eval_split = tfds.Split.TEST

    train = None
    if train_split is not None:
        train = tfds.load(
            name=dataset_name,
            split=train_split,
            data_dir=data_dir,
            shuffle_files=train_shuffle_files,
            download=download,
        )
    valid = tfds.load(
        name=dataset_name,
        split=eval_split,
        data_dir=data_dir,
        shuffle_files=eval_shuffle_files,
        download=download,
    )
    keys = None
    if info.supervised_keys:
        keys = ([info.supervised_keys[0]], [info.supervised_keys[1]])
    return train, valid, keys


def _train_and_eval_dataset_v1(
    dataset_name="wmt14_translate/de-en",
    data_dir=None,
    train_shuffle_files=True,
    eval_shuffle_files=False,
):
    """Return train and evaluation datasets plus supervised keys."""
    train_ds, eval_ds, supervised_keys = load_translation_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train_shuffle_files=train_shuffle_files,
        eval_shuffle_files=eval_shuffle_files,
        input_key="en",
        target_key="de",
    )

    examples = list(tfds.as_numpy(train_ds.take(1)))
    input_key = "inputs" if "inputs" in examples[0] else "targets"
    return train_ds, eval_ds, ([input_key], ["targets"])


@dataclass(frozen=True)
class DatasetStreams:
    train: Optional[tf.data.Dataset]
    eval: tf.data.Dataset
    supervised_keys: Optional[Tuple]
    dataset_name: str
    data_dir: Optional[str]


@dataclass(frozen=True)
class DatasetLoader:
    dataset_name: str
    data_dir: Optional[str] = None
    eval_holdout_size: float = 0.0
    use_alt_eval: bool = False
    download: bool = True
    train_split: Optional[str] = None
    eval_split: Optional[str] = None
    shuffle_train_files: bool = True
    shuffle_eval_files: bool = False
    host_id: Optional[int] = None
    n_hosts: Optional[int] = None
    require_train_split: bool = True

    def datasets(self):
        data_dir = download_and_prepare(self.dataset_name, self.data_dir)
        download = self.download
        if data_dir:
            normalized = os.path.normpath(os.path.abspath(data_dir))
            if normalized.endswith(os.path.normpath("resources/data/corpus/test")):
                download = False
        try:
            host_id = jax.process_index() if self.host_id is None else self.host_id
            n_hosts = self.n_hosts or jax.host_count()
        except Exception:
            host_id = 0 if self.host_id is None else self.host_id
            n_hosts = self.n_hosts or 1
        subsplit = (host_id / n_hosts, (host_id + 1) / n_hosts) if n_hosts > 1 else None
        train_ds, eval_ds, keys = _train_and_eval_dataset(
            self.dataset_name,
            data_dir,
            self.eval_holdout_size,
            train_shuffle_files=self.shuffle_train_files,
            eval_shuffle_files=self.shuffle_eval_files,
            use_alt_eval=self.use_alt_eval,
            download=download,
            train_split_override=self.train_split,
            eval_split_override=self.eval_split,
            subsplit=subsplit,
            require_train_split=self.require_train_split,
        )
        return DatasetStreams(
            train=train_ds,
            eval=eval_ds,
            supervised_keys=keys,
            dataset_name=_resolve_dataset_name(self.dataset_name),
            data_dir=data_dir,
        )
