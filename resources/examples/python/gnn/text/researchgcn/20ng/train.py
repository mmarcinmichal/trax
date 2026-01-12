import re

from collections import Counter

import datasets
import numpy as np

import trax.fastmath as fastmath

from resources.examples.python.base import (
    DeviceType,
    crete_graph_batch_generator,
    evaluate_model,
    initialize_model,
    train_model,
)
from trax import layers as tl
from trax import optimizers
from trax.fastmath import numpy as jnp
from trax.models import gnn
from learning.training.engines import jax as trainers

MAX_LEN = 2000
VOCAB_SIZE = 200_000
WINDOW_SIZE = 5


def clean(text):
    t = text.lower()
    t = re.sub(r"\S+@\S+", " ", t)  # emails
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[_A-Za-z]:/[^ \n]+", " ", t)  # paths/urls
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_vocab(texts, min_freq=5):
    counter = Counter()
    for t in texts:
        counter.update(clean(t).split()[:MAX_LEN])
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, c in counter.most_common():
        if c < min_freq or len(vocab) >= VOCAB_SIZE:
            break
        vocab[w] = len(vocab)
    return vocab


def encode(text, vocab):
    tokens = [vocab.get(w, 1) for w in text.lower().split()[:MAX_LEN]]
    if len(tokens) < MAX_LEN:
        tokens += [0] * (MAX_LEN - len(tokens))
    return np.array(tokens)


def window_adjacency(length=MAX_LEN, window=WINDOW_SIZE):
    """Create adjacency connecting tokens within a sliding window."""
    adj = np.zeros((length, length), dtype=np.float32)
    for i in range(length):
        left, right = max(0, i - window), min(length, i + window + 1)
        for j in range(left, right):
            if i != j:
                adj[i, j] = 1.0
    np.fill_diagonal(adj, 1.0)
    return adj


def load_data():
    train_ds = datasets.load_dataset("SetFit/20_newsgroups", split="train")
    test_ds = datasets.load_dataset("SetFit/20_newsgroups", split="test")
    # train_ds = datasets.load_dataset("imdb", split="train[:2000]")
    # test_ds = datasets.load_dataset("imdb", split="test[:1000]")

    vocab = build_vocab(train_ds["text"])
    x_train = np.stack([encode(t, vocab) for t in train_ds["text"]])
    y_train = np.array(train_ds["label"], dtype=np.int64)
    x_test = np.stack([encode(t, vocab) for t in test_ds["text"]])
    y_test = np.array(test_ds["label"], dtype=np.int64)

    adj = window_adjacency()
    a_train = np.broadcast_to(adj, (x_train.shape[0], MAX_LEN, MAX_LEN))
    a_test = np.broadcast_to(adj, (x_test.shape[0], MAX_LEN, MAX_LEN))

    return (x_train, a_train, y_train), (x_test, a_test, y_test), len(vocab)


def attention_pool():
    """Compute weighted average of node embeddings."""
    return tl.Serial(
        tl.Branch(
            None,
            tl.Serial(
                tl.Dense(1),
                tl.Flatten(n_axes_to_keep=2),
                tl.Softmax(),
            ),
        ),
        tl.Fn(
            "AttnPool",
            lambda x, w: jnp.sum(x * w, axis=1),
        ),
    )


def build_model(vocab_size):
    return tl.Serial(
        tl.Parallel(tl.Embedding(vocab_size, 512), None),
        gnn.GraphAttentionNet(hidden_sizes=(512, 64, 32), num_heads=16),
        tl.Select([0]),
        attention_pool(),
        tl.Dense(20),
        tl.Select([0, 2, 3]),
    )


def main():
    DEFAULT_BATCH_SIZE = 8
    STEPS_NUMBER = 14_000

    (x_train, a_train, y_train), (x_test, a_test, y_test), vocab_size = load_data()
    batch_gen = crete_graph_batch_generator(
        x_train, a_train, y_train, batch_size=DEFAULT_BATCH_SIZE
    )
    example_batch = next(batch_gen)

    model_with_loss = tl.Serial(
        build_model(vocab_size), tl.CrossEntropyLossWithLogSoftmax()
    )
    initialize_model(model_with_loss, example_batch)

    optimizer = optimizers.Adam(0.00001)
    trainer = trainers.TrainingEngine(model_with_loss, optimizer)

    base_rng = fastmath.random.get_prng(0)
    train_model(
        trainer,
        batch_gen,
        num_steps=STEPS_NUMBER,
        base_rng=base_rng,
        device_type=DeviceType.GPU.value,
    )

    test_batch_gen = crete_graph_batch_generator(
        x_test, a_test, y_test, batch_size=DEFAULT_BATCH_SIZE
    )

    # Evaluate model on a test set
    test_results = evaluate_model(
        trainer=trainer,
        batch_gen=test_batch_gen,
        device_type=DeviceType.CPU.value,
        num_batches=50,
    )

    print(f"Final test accuracy: {test_results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
