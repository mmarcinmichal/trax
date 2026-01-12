# coding=utf-8
"""Public learning API for Trax."""

from learning.training.task import EvaluationTask, TrainingTask
from learning.training.engines import Loop, epochs, num_devices, train

__all__ = [
    "EvaluationTask",
    "TrainingTask",
    "Loop",
    "train",
    "num_devices",
    "epochs",
]
