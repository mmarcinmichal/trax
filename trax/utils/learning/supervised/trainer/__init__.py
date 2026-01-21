"""Trainer entrypoints and config helpers."""

from trax.utils.learning.supervised.trainer.base import (  # noqa: F401
    main,
    num_devices,
    tf_init_tpu,
    train,
)

__all__ = ["main", "num_devices", "tf_init_tpu", "train"]
