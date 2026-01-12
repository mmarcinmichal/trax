from data.loader.raw.base import RawDataset, Splits, load_dataset

import trax.fastmath as fastmath

from resources.examples.python.base import (
    DeviceType,
    create_batch_generator,
    evaluate_model,
    initialize_model,
    train_model,
)
from trax import layers as tl
from trax import optimizers
from trax.trainers import jax as trainers


def build_model():
    # Build your model with loss function
    model = tl.Serial(
        tl.Dense(16, use_bias=True), tl.Relu(), tl.Dense(10, use_bias=False)
    )
    model_with_loss = tl.Serial(model, tl.CrossEntropyLossWithLogSoftmax())
    return model_with_loss


def main():
    # Default setup
    DEFAULT_BATCH_SIZE = 8
    STEPS_NUMBER = 20_000

    # Load data
    X, y = load_dataset(RawDataset.DIGITS.value)
    batch_generator = create_batch_generator(
        X, y, batch_size=DEFAULT_BATCH_SIZE, seed=42
    )
    example_batch = next(batch_generator)

    # Build and initialize model
    model_with_loss = build_model()
    initialize_model(model_with_loss, example_batch)

    # Setup optimizer and trainers
    optimizer = optimizers.Adam(0.001)
    trainer = trainers.Trainer(model_with_loss, optimizer)

    base_rng = fastmath.random.get_prng(0)

    # Run training on CPU and/or GPU
    train_model(
        trainer,
        batch_generator,
        STEPS_NUMBER,
        base_rng,
        device_type=DeviceType.GPU.value,
    )

    # Load test data
    test_data, test_labels = load_dataset(
        dataset_name=RawDataset.DIGITS.value, split=Splits.TEST.value
    )

    # Create batch generator for test data
    test_batch_gen = create_batch_generator(
        test_data, test_labels, None, DEFAULT_BATCH_SIZE, 0
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
