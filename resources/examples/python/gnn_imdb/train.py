from collections import Counter

import datasets
import numpy as np

import trax.fastmath as fastmath

from resources.examples.python.base import (
    DeviceType,
    initialize_model,
    train_model,
)
from trax import layers as tl
from trax import optimizers
from trax.models import gnn
from trax.trainers import jax as trainers

MAX_LEN = 40
VOCAB_SIZE = 8000


def build_vocab(texts):
    counter = Counter()
    for t in texts:
        counter.update(t.lower().split()[:MAX_LEN])
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (w, _) in enumerate(counter.most_common(VOCAB_SIZE - 2), start=2):
        vocab[w] = i
    return vocab


def encode(text, vocab):
    tokens = [vocab.get(w, 1) for w in text.lower().split()[:MAX_LEN]]
    if len(tokens) < MAX_LEN:
        tokens += [0] * (MAX_LEN - len(tokens))
    return np.array(tokens)


def chain_adjacency(length=MAX_LEN):
    adj = np.zeros((length, length), dtype=np.float32)
    for i in range(length - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj


def load_data():
    train_ds = datasets.load_dataset("imdb", split="train[:2000]")
    test_ds = datasets.load_dataset("imdb", split="test[:1000]")

    vocab = build_vocab(train_ds["text"])
    x_train = np.stack([encode(t, vocab) for t in train_ds["text"]])
    y_train = np.array(train_ds["label"], dtype=np.int64)
    x_test = np.stack([encode(t, vocab) for t in test_ds["text"]])
    y_test = np.array(test_ds["label"], dtype=np.int64)

    adj = chain_adjacency()
    a_train = np.broadcast_to(adj, (x_train.shape[0], MAX_LEN, MAX_LEN))
    a_test = np.broadcast_to(adj, (x_test.shape[0], MAX_LEN, MAX_LEN))

    return (x_train, a_train, y_train), (x_test, a_test, y_test), len(vocab)


def graph_batch_generator(nodes, adjs, labels, batch_size=32, seed=0):
    rng = np.random.default_rng(seed)
    n = nodes.shape[0]
    while True:
        idx = rng.choice(n, batch_size, replace=False)
        yield ((nodes[idx], adjs[idx]), labels[idx], np.ones(batch_size))


def build_model(vocab_size):
    return tl.Serial(
        tl.Parallel(tl.Embedding(vocab_size, 32), None),
        gnn.GraphAttentionNet(hidden_sizes=(32, 16), num_heads=2),
        tl.Select([0]),
        tl.Mean(axis=1),
        tl.Dense(2),
        tl.LogSoftmax(),
    )


def main():
    (x_train, a_train, y_train), _, vocab_size = load_data()
    batch_gen = graph_batch_generator(x_train, a_train, y_train, batch_size=16)
    example_batch = next(batch_gen)

    model_with_loss = tl.Serial(
        build_model(vocab_size), tl.CrossEntropyLossWithLogSoftmax()
    )
    initialize_model(model_with_loss, example_batch)

    optimizer = optimizers.Adam(0.001)
    trainer = trainers.Trainer(model_with_loss, optimizer)

    base_rng = fastmath.random.get_prng(0)
    train_model(
        trainer,
        batch_gen,
        num_steps=100,
        base_rng=base_rng,
        device_type=DeviceType.CPU.value,
    )


if __name__ == "__main__":
    main()
