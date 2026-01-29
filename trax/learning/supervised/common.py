# coding=utf-8
# Copyright 2022 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utilities shared between supervised training entry points."""

from trax import layers as tl
from trax.learning.training.task import EvaluationTask, TrainingTask
from trax.optimizers import base as optim_base


def default_metrics():
    """Returns the default metrics mapping used for supervised training."""

    return {
        "loss": tl.WeightedCategoryCrossEntropy(),
        "accuracy": tl.WeightedCategoryAccuracy(),
        "sequence_accuracy": tl.MaskedSequenceAccuracy(),
        "neg_log_perplexity": tl.Serial(
            tl.WeightedCategoryCrossEntropy(), tl.Negate()
        ),
        "weights_per_batch_per_core": tl.Serial(
            tl.Drop(), tl.Drop(), tl.Sum()
        ),
    }


def _resolve_stream(stream, n_devices):
    if callable(stream):
        try:
            return stream(n_devices)
        except TypeError:
            return stream()
    return stream


def create_train_task(
    inputs,
    loss_layer,
    optimizer,
    lr_schedule_fn,
    n_devices,
    eval_frequency,
    permanent_checkpoint_frequency,
    use_memory_efficient_trainer,
):
    """Builds a ``TrainingTask`` from training entry-point arguments."""

    opt = optimizer
    if not use_memory_efficient_trainer and not isinstance(optimizer, optim_base.Optimizer):
        # For non-memory-efficient trainers the optimizer factory must be called
        # before constructing the TrainingTask.
        opt = optimizer()

    return TrainingTask(
        _resolve_stream(inputs.train_stream, n_devices),
        loss_layer=loss_layer,
        optimizer=opt,
        lr_schedule=lr_schedule_fn(),
        n_steps_per_checkpoint=eval_frequency,
        n_steps_per_permanent_checkpoint=permanent_checkpoint_frequency,
    )


def create_eval_task(stream_or_inputs, metrics_dict, eval_steps, n_devices, export_prefix=None):
    """Builds an ``EvaluationTask`` from training entry-point arguments."""

    names, metrics = zip(*metrics_dict.items())
    if hasattr(stream_or_inputs, "eval_stream"):
        stream = _resolve_stream(stream_or_inputs.eval_stream, n_devices)
    elif hasattr(stream_or_inputs, "train_stream"):
        stream = _resolve_stream(stream_or_inputs.train_stream, n_devices)
    elif hasattr(stream_or_inputs, "stream"):
        stream = _resolve_stream(stream_or_inputs.stream, n_devices)
    else:
        stream = _resolve_stream(stream_or_inputs, n_devices)
    return EvaluationTask(
        stream,
        metrics,
        metric_names=names,
        n_eval_batches=eval_steps,
        export_prefix=export_prefix,
    )

