import os

import datasets
import numpy as np

import trax.fastmath as fastmath

from resources.examples.python.base import (
    DeviceType,
    evaluate_model,
    graph_batch_generator,
    initialize_model,
    train_model,
)
from trax import layers as tl
from trax import optimizers
from trax.data.encoder import encoder as text_encoder
from trax.models import gnn
from trax.trainers import jax as trainers

MAX_LEN = 2_000
WINDOW_SIZE = 10
SPM_FILE = "en_32k.sentencepiece"
VOCAB_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "vocabs")
)


def _pad_or_trim(tokens, max_len=MAX_LEN, pad_id=0):
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.shape[0] >= max_len:
        return tokens[:max_len]
    padded = np.full((max_len,), pad_id, dtype=np.int64)
    padded[: tokens.shape[0]] = tokens
    return padded


def window_adjacency(length=MAX_LEN, window_size=10, add_self_loops=False):
    adj = np.zeros((length, length), dtype=np.float32)
    for i in range(length):
        start = max(0, i - window_size)
        end = min(length, i + window_size + 1)
        if add_self_loops:
            adj[i, i] = 1
        if start < i:
            adj[i, start:i] = 1
        if i + 1 < end:
            adj[i, i + 1 : end] = 1
    return adj


def load_data():
    train_ds = datasets.load_dataset("imdb", split="train")
    test_ds = datasets.load_dataset("imdb", split="test")
    # train_ds = datasets.load_dataset("imdb", split="train[:2000]")
    # test_ds = datasets.load_dataset("imdb", split="test[:1000]")

    tokenizer_fn = text_encoder.Tokenize(
        vocab_type="sentencepiece",
        vocab_file=SPM_FILE,
        vocab_dir=VOCAB_DIR,
        n_reserved_ids=0,
    )
    vocab_size = text_encoder.vocab_size(
        vocab_type="sentencepiece", vocab_file=SPM_FILE, vocab_dir=VOCAB_DIR
    )

    x_train = np.stack(
        [_pad_or_trim(tokens) for tokens in tokenizer_fn(iter(train_ds["text"]))]
    )
    y_train = np.array(train_ds["label"], dtype=np.int64)
    x_test = np.stack(
        [_pad_or_trim(tokens) for tokens in tokenizer_fn(iter(test_ds["text"]))]
    )
    y_test = np.array(test_ds["label"], dtype=np.int64)

    adj = window_adjacency(window_size=WINDOW_SIZE, add_self_loops=False)
    a_train = np.broadcast_to(adj, (x_train.shape[0], MAX_LEN, MAX_LEN))
    a_test = np.broadcast_to(adj, (x_test.shape[0], MAX_LEN, MAX_LEN))

    return (x_train, a_train, y_train), (x_test, a_test, y_test), vocab_size


def build_model(vocab_size):
    return tl.Serial(
        tl.Parallel(tl.Embedding(vocab_size, 512), None),
        gnn.GraphAttentionNet(hidden_sizes=(512, 64, 32), num_heads=2),
        tl.Select([0]),
        tl.Mean(axis=1),
        tl.Dense(2),
        tl.Select([0, 2, 3]),
    )


def main():
    DEFAULT_BATCH_SIZE = 2
    STEPS_NUMBER = 20_000

    (x_train, a_train, y_train), (x_test, a_test, y_test), vocab_size = load_data()
    batch_gen = graph_batch_generator(
        x_train, a_train, y_train, batch_size=DEFAULT_BATCH_SIZE
    )
    example_batch = next(batch_gen)

    model_with_loss = tl.Serial(
        build_model(vocab_size), tl.CrossEntropyLossWithLogSoftmax()
    )
    initialize_model(model_with_loss, example_batch)

    optimizer = optimizers.Adam(0.0001)
    trainer = trainers.Trainer(model_with_loss, optimizer)

    base_rng = fastmath.random.get_prng(0)
    train_model(
        trainer,
        batch_gen,
        num_steps=STEPS_NUMBER,
        base_rng=base_rng,
        device_type=DeviceType.GPU.value,
    )

    test_batch_gen = graph_batch_generator(
        x_test, a_test, y_test, batch_size=DEFAULT_BATCH_SIZE
    )

    # Evaluate model on a test set
    test_results = evaluate_model(
        trainer=trainer,
        batch_gen=test_batch_gen,
        device_type=DeviceType.CPU.value,
        num_batches=500,
    )

    print(f"Final test accuracy: {test_results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
