import numpy as np

import trax.fastmath as fastmath

from resources.examples.python.base import (
    Dataset,
    DeviceType,
    initialize_model,
    load_dataset,
    train_model,
)
from trax import layers as tl
from trax import optimizers
from trax.models import gnn
from trax.trainers import jax as trainers


def grid_adjacency(height=28, width=28):
    """Returns 4-neighbour adjacency for an image grid."""
    n = height * width
    adj = np.zeros((n, n), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if x > 0:
                adj[idx, idx - 1] = 1
            if x < width - 1:
                adj[idx, idx + 1] = 1
            if y > 0:
                adj[idx, idx - width] = 1
            if y < height - 1:
                adj[idx, idx + width] = 1
    return adj


def create_graph_data(images):
    nodes = images.reshape((images.shape[0], 28 * 28, 1)).astype(np.float32)
    adj = grid_adjacency()
    adj = np.broadcast_to(adj, (images.shape[0], adj.shape[0], adj.shape[1]))
    return nodes, adj


def graph_batch_generator(nodes, adjs, labels, batch_size=32, seed=0):
    rng = np.random.default_rng(seed)
    n = nodes.shape[0]
    while True:
        idx = rng.choice(n, batch_size, replace=False)
        yield ((nodes[idx], adjs[idx]), labels[idx], np.ones(batch_size))


def build_model():
    return tl.Serial(
        gnn.GraphConvNet(hidden_sizes=(32, 16)),
        tl.Select([0]),
        tl.Mean(axis=1),
        tl.Dense(10),
        tl.LogSoftmax(),
    )


def main():
    images, labels = load_dataset(Dataset.MNIST.value)
    nodes, adjs = create_graph_data(images)

    batch_gen = graph_batch_generator(nodes, adjs, labels, batch_size=8)
    example_batch = next(batch_gen)

    model_with_loss = tl.Serial(build_model(), tl.CrossEntropyLossWithLogSoftmax())
    initialize_model(model_with_loss, example_batch)

    optimizer = optimizers.Adam(0.001)
    trainer = trainers.Trainer(model_with_loss, optimizer)

    base_rng = fastmath.random.get_prng(0)
    train_model(trainer, batch_gen, num_steps=100, base_rng=base_rng, device_type=DeviceType.CPU.value)


if __name__ == "__main__":
    main()
