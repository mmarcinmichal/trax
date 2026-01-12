import numpy as np

from data.loader.raw.base import RawDataset, Splits, load_dataset

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
from trax.models import gnn
from trax.trainers import jax as trainers


def grid_adjacency(height=28, width=28):
    """Returns 4-neighbor adjacency for an image grid."""
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


def build_model():
    return tl.Serial(
        gnn.GraphAttentionNet(hidden_sizes=(128, 64, 32, 16)),
        tl.Select([0]),
        tl.Mean(axis=1),
        tl.Dense(10),
        tl.Select([0, 2, 3]),
    )


def main():
    DEFAULT_BATCH_SIZE = 8
    STEPS_NUMBER = 20_000

    images, labels = load_dataset(RawDataset.MNIST.value)
    nodes, adjacency = create_graph_data(images)

    batch_generator = crete_graph_batch_generator(
        nodes, adjacency, labels, batch_size=DEFAULT_BATCH_SIZE
    )
    example_batch = next(batch_generator)

    model_with_loss = tl.Serial(build_model(), tl.CrossEntropyLossWithLogSoftmax())
    initialize_model(model_with_loss, example_batch)

    optimizer = optimizers.Adam(0.0001)
    trainer = trainers.Trainer(model_with_loss, optimizer)

    base_rng = fastmath.random.get_prng(0)
    train_model(
        trainer,
        batch_generator,
        STEPS_NUMBER,
        base_rng,
        device_type=DeviceType.GPU.value,
    )

    images, labels = load_dataset(RawDataset.MNIST.value, Splits.TEST.value)
    nodes, adjacency = create_graph_data(images)

    test_batch_gen = crete_graph_batch_generator(
        nodes, adjacency, labels, batch_size=DEFAULT_BATCH_SIZE
    )

    # Evaluate model on a test set
    test_results = evaluate_model(
        trainer=trainer,
        batch_gen=test_batch_gen,
        device_type=DeviceType.CPU.value,
        num_batches=100,
    )

    print(f"Final test accuracy: {test_results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
