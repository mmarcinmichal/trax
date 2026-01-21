"""Compatibility re-exports for supervised training helpers."""

import importlib

__all__ = [
    "Loop",
    "TrainingTask",
    "EvaluationTask",
    "ScheduleBuilder",
    "ensure_optimizer_instance",
    "train",
    "num_devices",
    "epochs",
    "pickle_to_store",
    "unpickle_from_store",
]


_TASK_ATTRS = {
    "TrainingTask",
    "EvaluationTask",
}

_TRAINER_ATTRS = {
    "Loop",
    "ScheduleBuilder",
    "ensure_optimizer_instance",
    "train",
    "num_devices",
    "epochs",
    "pickle_to_store",
    "unpickle_from_store",
}


def __getattr__(name):
    if name in _TASK_ATTRS:
        module = importlib.import_module("trax.learning.training.task")
        return getattr(module, name)
    if name in _TRAINER_ATTRS:
        module = importlib.import_module("trax.learning.training.trainer")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
