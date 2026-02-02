from enum import Enum
from typing import Tuple, Union

import datasets
import numpy as np

from trax.utils import logging as trax_logging
from datasets import DatasetDict
from sklearn.datasets import (
    _twenty_newsgroups,
    fetch_20newsgroups,
    load_digits,
    load_iris,
)

trax_logging.set_verbosity(trax_logging.INFO)


class RawDataset(Enum):
    """Supported datasets."""

    IRIS = "iris"
    DIGITS = "digits"
    MNIST = "mnist"
    IMDB = "imdb"
    NG = "20NG"


class Splits(Enum):
    """Supported datasets."""

    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


def load_20ng(split: str = Splits.TRAIN.value) -> Tuple[np.ndarray, np.ndarray]:
    # Mirror for broken upstream 20NG archive URL.
    ARCHIVE_URL_OVERRIDE = (
        "https://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz"
    )

    def _override_20ng_archive_url():
        archive = _twenty_newsgroups.ARCHIVE
        if getattr(archive, "url", None) == ARCHIVE_URL_OVERRIDE:
            return
        if hasattr(archive, "_replace"):
            _twenty_newsgroups.ARCHIVE = archive._replace(url=ARCHIVE_URL_OVERRIDE)
        else:
            _twenty_newsgroups.ARCHIVE = type(archive)(
                filename=archive.filename,
                url=ARCHIVE_URL_OVERRIDE,
                checksum=archive.checksum,
            )

    _override_20ng_archive_url()

    dataset = fetch_20newsgroups(subset=split, shuffle=False)

    ds_dict = DatasetDict(text=dataset["data"], label=dataset["target"])

    texts = np.array(ds_dict["text"], dtype=object)
    labels = np.array(ds_dict["label"], dtype=np.int64)
    return texts, labels


def load_mnist(split: str = Splits.TRAIN.value) -> Tuple[np.ndarray, np.ndarray]:
    # Load the MNIST dataset using Hugging Face Datasets
    # Use 'mnist' for the standard MNIST dataset
    dataset = datasets.load_dataset("mnist", split=split)

    # Pre-allocate arrays with the correct shape
    num_examples = len(dataset)
    X = np.zeros((num_examples, 784), dtype=np.float32)
    y = np.zeros(num_examples, dtype=np.int64)

    # Process each example in the dataset
    i = 0
    for image, label in zip(dataset["image"], dataset["label"]):
        # Flatten image from (28, 28) to (784,) and normalize
        X[i] = np.array(image).reshape(-1).astype(np.float32) / 255.0
        y[i] = label
        i += 1

    return X, y


def load_imdb(split: str = Splits.TRAIN.value) -> Tuple[np.ndarray, np.ndarray]:
    """Load the IMDB sentiment dataset as text and labels."""
    dataset = datasets.load_dataset("imdb", split=split)
    texts = np.array(dataset["text"], dtype=object)
    labels = np.array(dataset["label"], dtype=np.int64)
    return texts, labels


def load_dataset(
    dataset_name: str = RawDataset.IRIS.value,
    split: str = Splits.TRAIN.value,
) -> Union[Tuple[np.ndarray, np.ndarray]]:
    """
    Load a dataset by name and split.

    Args:
        dataset_name: Name of the dataset to load.
        split: Which split to load ('train', 'test', or 'validation')

    Returns:
        For sklearn datasets: Tuple of (data, labels) arrays.
        For TensorFlow datasets: A TensorFlow dataset object.
    """
    if dataset_name == RawDataset.IRIS.value:
        dataset = load_iris()
        data, labels = dataset.data, dataset.target
        # For sklearn datasets, we'll simulate train/test split
        if split == "test":
            # Use last 20% as test
            test_size = len(data) // 5
            return data[-test_size:], labels[-test_size:]
        else:
            # Use first 80% as train
            train_size = len(data) - (len(data) // 5)
            return data[:train_size], labels[:train_size]

    elif dataset_name == RawDataset.DIGITS.value:
        dataset = load_digits()
        data, labels = dataset.data, dataset.target
        # For sklearn datasets, we'll simulate train/test split
        if split == "test":
            # Use last 20% as test
            test_size = len(data) // 5
            return data[-test_size:], labels[-test_size:]
        else:
            # Use first 80% as train
            train_size = len(data) - (len(data) // 5)
            return data[:train_size], labels[:train_size]

    elif dataset_name == RawDataset.MNIST.value:
        x, y = load_mnist(split=split)
        return x, y
    elif dataset_name == RawDataset.IMDB.value:
        x, y = load_imdb(split=split)
        return x, y
    elif dataset_name == RawDataset.NG.value:
        x, y = load_20ng(split=split)
        return x, y
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


if __name__ == "__main__":
    trax_logging.info("Loading data module")