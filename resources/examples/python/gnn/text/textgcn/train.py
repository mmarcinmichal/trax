from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from absl import logging
from jax.experimental import sparse as jsparse
from learning.training.engines import jax as trainers

import trax.fastmath as fastmath

from resources.examples.python.base import (
    DeviceType,
    find_project_root,
    initialize_model,
    train_model,
)
from trax import layers as tl
from trax import optimizers
from trax.models.gnn import GraphConvSparse

logging.set_verbosity(logging.INFO)

PROJECT_ROOT = find_project_root(Path(__file__).resolve())
GRAPH_NPZ = (
    PROJECT_ROOT
    / "resources"
    / "data"
    / "serialized"
    / "graphs"
    / "20_newsgroups_bydate.npz"
)


def load_graph():
    data = np.load(GRAPH_NPZ, allow_pickle=True)

    csr_data = data["adj_data"]
    csr_indices = data["adj_indices"]
    csr_indptr = data["adj_indptr"]
    shape = tuple(data["adj_shape"])

    # etykiety TYLKO dla dokumentów (0..num_docs-1)
    labels_docs = data["labels"].astype(np.int64)  # (num_docs,)
    train_mask_docs = data["train_mask"].astype(bool)  # (num_docs,)
    val_mask_docs = data["val_mask"].astype(bool)  # (num_docs,)
    test_mask_docs = data["test_mask"].astype(bool)  # (num_docs,)

    num_docs = labels_docs.shape[0]
    num_nodes = shape[0]  # docs + words
    vocab_size = num_nodes - num_docs

    logging.info(f"num_docs={num_docs}, num_nodes={num_nodes}, vocab_size={vocab_size}")

    return (
        csr_data,
        csr_indices,
        csr_indptr,
        shape,
        labels_docs,
        train_mask_docs,
        val_mask_docs,
        test_mask_docs,
        num_docs,
        num_nodes,
    )


def build_adj_csr_jax(csr_data, csr_indices, csr_indptr, shape):
    """Konwersja SciPy-like CSR -> jax.experimental.sparse.CSR."""
    data = jnp.array(csr_data)
    indices = jnp.array(csr_indices, dtype=jnp.int32)
    indptr = jnp.array(csr_indptr, dtype=jnp.int32)
    return jsparse.CSR((data, indices, indptr), shape=shape)


def build_model(num_nodes, num_docs, num_classes, adj_csr, mode="train"):
    """TextGCN: embedding nodes + 2-layer GCN + logits for documents only."""
    hidden_dim = 512

    def take_docs(h):
        # h: (N, C) -> (num_docs, C)
        return h[:num_docs]

    return tl.Serial(
        # input: indices [0..N-1] -> (N, hidden_dim)
        tl.Embedding(num_nodes, hidden_dim),
        tl.Dropout(0.5, mode=mode),
        # 2-layer GCN on the full graph
        GraphConvSparse(adj_csr, 124, activation=tl.Relu),
        tl.Dropout(0.5, mode=mode),
        GraphConvSparse(adj_csr, 64, activation=tl.Relu),
        tl.Dropout(0.1, mode=mode),
        GraphConvSparse(adj_csr, num_classes, activation=tl.Serial),
        # slice to document nodes
        tl.Fn("TakeDocs", take_docs, n_out=1),
    )


def full_batch_generator(node_indices, labels_docs, mask_docs):
    while True:
        w = mask_docs.astype(np.float32)
        yield node_indices, labels_docs, w


def evaluate_dataset(eval_model, node_indices, labels_docs, mask_docs):
    dummy_rng = fastmath.random.get_prng(123)

    logits_docs = eval_model(node_indices, rng=dummy_rng)
    preds = jnp.argmax(logits_docs, axis=1)
    labels = jnp.asarray(labels_docs)
    mask = jnp.asarray(mask_docs, dtype=bool)
    return float(jnp.mean((preds == labels)[mask]))


def main():
    (
        csr_data,
        csr_indices,
        csr_indptr,
        shape,
        labels_docs,
        train_mask_docs,
        val_mask_docs,
        test_mask_docs,
        num_docs,
        num_nodes,
    ) = load_graph()

    num_classes = int(labels_docs.max()) + 1

    logging.info(
        f"Label Stats: {labels_docs.min()} - {labels_docs.max()} ({len(np.unique(labels_docs))} classes)"
    )
    logging.info(
        f"Docs Split: Train {train_mask_docs.sum()}, Val {val_mask_docs.sum()}, Test {test_mask_docs.sum()}"
    )

    # Stała macierz A_hat jako CSR w JAX
    adj_csr = build_adj_csr_jax(csr_data, csr_indices, csr_indptr, shape)

    # Indeksy węzłów: 0..num_nodes-1
    node_indices = np.arange(num_nodes, dtype=np.int32)

    # --- model + loss ---
    model = build_model(num_nodes, num_docs, num_classes, adj_csr, mode="train")

    model_with_loss = tl.Serial(
        tl.Branch(
            tl.Serial(
                tl.Select([0], n_in=3),  # x = node_indices
                model,  # -> logits_docs: (num_docs, num_classes)
            ),
            tl.Select([1], n_in=3),  # labels_docs: (num_docs,)
            tl.Select([2], n_in=3),  # weights_docs: (num_docs,)
        ),
        tl.CrossEntropyLossWithLogSoftmax(),
    )

    # Przykładowa "batch": cały graf, ale ważymy tylko train docs
    example_batch = (
        node_indices,
        labels_docs,
        train_mask_docs.astype(np.float32),
    )
    initialize_model(model_with_loss, example_batch)

    optimizer = optimizers.Adam(0.002)
    trainer = trainers.TrainingEngine(model_with_loss, optimizer)

    eval_model = build_model(num_nodes, num_docs, num_classes, adj_csr, mode="eval")
    initialize_model(eval_model, example_batch)

    # full-batch training – cały graf w jednym kroku
    train_gen = full_batch_generator(node_indices, labels_docs, train_mask_docs)

    for epoch in range(1, 1_600):
        key = jax.random.PRNGKey(epoch)
        base_rng = fastmath.random.get_prng(
            fastmath.random.randint(key, shape=(), minval=1, maxval=1_600*1_600)
        )

        train_model(
            trainer,
            train_gen,
            num_steps=10,
            base_rng=base_rng,
            device_type=DeviceType.GPU.value,
        )

        # Evaluate
        eval_model.weights = model.weights
        eval_model.state = model.state

        train_acc = evaluate_dataset(
            eval_model, node_indices, labels_docs, train_mask_docs
        )
        final_val_acc = evaluate_dataset(
            eval_model, node_indices, labels_docs, val_mask_docs
        )
        test_acc = evaluate_dataset(
            eval_model, node_indices, labels_docs, test_mask_docs
        )

        logging.info("-" * 30)
        logging.info("Results:")
        logging.info(f"Train Acc: {train_acc:.4f}")
        logging.info(f"Val Acc:   {final_val_acc:.4f}")
        logging.info(f"Test Acc:  {test_acc:.4f}")



if __name__ == "__main__":
    main()
