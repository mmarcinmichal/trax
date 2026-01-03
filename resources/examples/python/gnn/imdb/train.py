import os

import datasets
import numpy as np

from layers import LayerNorm

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
    test_ds  = datasets.load_dataset("imdb", split="test")

    tokenizer_fn = text_encoder.Tokenize(
        vocab_type="sentencepiece",
        vocab_file=SPM_FILE,
        vocab_dir=VOCAB_DIR,
        n_reserved_ids=1,   # reserve 0 for PAD
    )
    vocab_size = text_encoder.vocab_size(
        vocab_type="sentencepiece",
        vocab_file=SPM_FILE,
        vocab_dir=VOCAB_DIR,
        n_reserved_ids=1,
    )

    x_train = np.stack([_pad_or_trim(t, pad_id=0) for t in tokenizer_fn(iter(train_ds["text"]))])
    y_train = np.array(train_ds["label"], dtype=np.int64)

    x_test  = np.stack([_pad_or_trim(t, pad_id=0) for t in tokenizer_fn(iter(test_ds["text"]))])
    y_test  = np.array(test_ds["label"], dtype=np.int64)

    # Base window graph (shared for all docs), add self-loops ONCE here:
    base_adj = window_adjacency(length=MAX_LEN, window_size=WINDOW_SIZE, add_self_loops=True).astype(np.float32)

    return (x_train, y_train), (x_test, y_test), vocab_size, base_adj



def build_model(vocab_size):
    eps = 1e-9

    def _masked_mean(h, m):
        # h: (B,N,D), m: (B,N)
        denom = fastmath.numpy.sum(m, axis=1, keepdims=True) + eps   # (B,1)
        num = fastmath.numpy.sum(h * m[..., None], axis=1)            # (B,D)
        return num / denom                                  # (B,D)

    return tl.Serial(
        # (tok, adj, mask) -> (emb, adj, mask)
        tl.Parallel(tl.Serial(tl.Embedding(vocab_size, 512), tl.LayerNorm(), tl.Dropout(0.2),), None, None),

        # run GNN on (emb, adj), carry mask along
        tl.Branch(
            tl.Serial(
                tl.Select([0, 1], n_in=3),  # take (emb, adj) from (emb, adj, mask)
                gnn.GraphAttentionNet(hidden_sizes=(512, 512, 256), num_heads=2),
                tl.Select([0], n_in=2),     # drop adjacency returned by GNN
            ),
            tl.Select([2], n_in=3),         # mask
        ),

        tl.Fn("MaskedMean", _masked_mean, n_out=1),
        tl.Dropout(0.2),
        tl.Dense(2),
    )



def main():
    DEFAULT_BATCH_SIZE = 8
    STEPS_NUMBER = 40_000

    (x_train, y_train), (x_test, y_test), vocab_size, base_adj = load_data()
    batch_gen = graph_batch_generator(
        x_train, y_train, base_adj, batch_size=DEFAULT_BATCH_SIZE
    )
    example_batch = next(batch_gen)

    model_with_loss =  tl.Serial(
        tl.Branch(
            tl.Serial(
                tl.Select([0, 1, 2], n_in=5),   # (x, adj, mask) out of 5 inputs
                build_model(vocab_size),
            ),
            tl.Select([3], n_in=5),             # y
            tl.Select([4], n_in=5),             # w
        ),
        tl.CrossEntropyLossWithLogSoftmax(),
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
        x_test, y_test, base_adj, batch_size=DEFAULT_BATCH_SIZE
    )

    # Evaluate model on a test set
    test_results = evaluate_model(
        trainer=trainer,
        batch_gen=test_batch_gen,
        device_type=DeviceType.CPU.value,
        num_batches=25_000//DEFAULT_BATCH_SIZE,
    )

    print(f"Final test accuracy: {test_results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
