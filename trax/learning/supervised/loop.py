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
"""Convenience wrappers around :mod:`trax.learning.supervised.training`.

This module exposes a thin compatibility layer over :class:`training.Loop` to
simplify construction of standard supervised training runs. Unlike the legacy
``trainer_lib`` implementation, the utilities here defer all core training logic
and state management to :class:`Loop` while keeping commonly used defaults.
"""
import itertools
from typing import Iterable, List, Optional, Sequence

import gin
import tensorflow as tf

from trax import fastmath
from trax import layers as tl
from trax import optimizers as trax_opt
from trax.data.preprocessing import inputs as trax_inputs
from trax.learning.supervised import common, lr_schedules as lr, training
from trax.trainers import base as trainer_base

_DEFAULT_METRICS = common.default_metrics()
NamedStream = common.NamedStream


@gin.configurable
def num_devices(value=None):
    """Returns how many devices to use (if None, default, use all available)."""

    return value


def log(s, stdout=True):
    tf.get_logger().info(s)
    if stdout:
        print(s)


def epochs(total_steps: int, steps_to_skip: int, epoch_steps: Iterable[int]):
    """Generates the number of steps in each epoch before reaching total_steps."""

    steps_to_go = total_steps - steps_to_skip
    epoch_steps = iter(epoch_steps)

    # Remove the desired number of steps from the stream.
    for steps_this_epoch in epoch_steps:
        if steps_this_epoch > steps_to_skip:
            # Put back the number of steps left in the unfinished epoch.
            epoch_steps = itertools.chain([steps_this_epoch - steps_to_skip], epoch_steps)
        if steps_this_epoch >= steps_to_skip:
            break
        steps_to_skip -= steps_this_epoch

    # Yield the remaining steps per epoch up to total_steps.
    for steps_this_epoch in epoch_steps:
        steps_this_epoch = min(steps_this_epoch, steps_to_go)
        yield steps_this_epoch
        steps_to_go -= steps_this_epoch
        if steps_to_go == 0:
            break


def _prepare_tasks(
    inputs: trax_inputs.Inputs,
    loss_fn,
    optimizer,
    lr_schedule_fn,
    n_devices: int,
    eval_steps: int,
    eval_frequency: Optional[int],
    permanent_checkpoint_frequency: Optional[int],
    use_memory_efficient_trainer: bool,
    metrics,
    additional_train_tasks: Optional[Sequence[training.TrainTask]],
    additional_eval_tasks: Optional[Sequence[training.EvalTask]],
    additional_eval_streams: Optional[Sequence[NamedStream]],
):
    train_task = common.create_train_task(
        inputs,
        loss_layer=loss_fn,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        n_devices=n_devices,
        eval_frequency=eval_frequency,
        permanent_checkpoint_frequency=permanent_checkpoint_frequency,
        use_memory_efficient_trainer=use_memory_efficient_trainer,
    )

    metrics_dict = metrics if metrics is not None else _DEFAULT_METRICS
    eval_task = common.create_eval_task(inputs, metrics_dict, eval_steps, n_devices)

    additional_eval_tasks = list(additional_eval_tasks or [])
    additional_train_tasks = list(additional_train_tasks or [])
    additional_eval_tasks_from_streams = []
    if additional_eval_streams:
        for stream in additional_eval_streams:
            additional_eval_tasks_from_streams.append(
                common.create_eval_task(
                    stream,
                    metrics_dict,
                    eval_steps,
                    n_devices,
                    export_prefix=stream.name,
                )
            )

    return (
        [train_task] + additional_train_tasks,
        [eval_task] + additional_eval_tasks + additional_eval_tasks_from_streams,
    )


def train(
    output_dir,
    model=gin.REQUIRED,
    loss_fn=tl.WeightedCategoryCrossEntropy(),
    inputs=trax_inputs.batcher,
    optimizer=trax_opt.Adafactor,
    lr_schedule_fn=lr.multifactor,
    steps: int = 1000,
    checkpoints_at: Optional[List[int]] = None,
    permanent_checkpoints_at: Optional[List[int]] = None,
    eval_steps: int = 10,
    eval_frequency: int = 100,
    permanent_checkpoint_frequency: Optional[int] = None,
    random_seed: Optional[int] = None,
    metrics=None,
    checkpoint_highest: Optional[str] = None,
    checkpoint_lowest: Optional[str] = None,
    loss_chunk_size: int = 0,
    use_memory_efficient_trainer: bool = False,
    adasum: bool = False,
    init_checkpoint: Optional[str] = None,
    callbacks=None,
    n_weights_shards: int = 1,
    additional_train_tasks: Optional[Sequence[training.TrainTask]] = None,
    additional_eval_tasks: Optional[Sequence[training.EvalTask]] = None,
    additional_eval_streams: Optional[Sequence[NamedStream]] = None,
):
    """Train the model on the inputs via :class:`training.Loop`."""

    trainer_base.N_WEIGHTS_SHARDS = n_weights_shards
    if permanent_checkpoint_frequency is not None and permanent_checkpoints_at is not None:
        raise ValueError(
            'Only one of ["permanent_checkpoint_frequency", "permanent_checkpoints_at"] should be set.'
        )

    n_devices = num_devices() or fastmath.local_device_count()
    if callable(inputs):
        inputs = inputs()

    train_tasks, eval_tasks = _prepare_tasks(
        inputs,
        loss_fn,
        optimizer,
        lr_schedule_fn,
        n_devices,
        eval_steps,
        eval_frequency,
        permanent_checkpoint_frequency,
        use_memory_efficient_trainer,
        metrics,
        additional_train_tasks,
        additional_eval_tasks,
        additional_eval_streams,
    )

    checkpoint_at = (lambda step: step in checkpoints_at) if checkpoints_at is not None else None
    permanent_checkpoint_at = (
        (lambda step: step in permanent_checkpoints_at)
        if permanent_checkpoints_at is not None
        else None
    )

    model_train = model(mode="train")
    model_predict_eval = model(mode="eval")
    if init_checkpoint:
        model_train.init_from_file(init_checkpoint, weights_only=True)
        model_predict_eval.init_from_file(init_checkpoint, weights_only=True)

    loop = training.Loop(
        model_train,
        train_tasks,
        eval_model=model_predict_eval,
        eval_tasks=eval_tasks,
        output_dir=output_dir,
        checkpoint_at=checkpoint_at,
        checkpoint_low_metric=checkpoint_lowest,
        checkpoint_high_metric=checkpoint_highest,
        permanent_checkpoint_at=permanent_checkpoint_at,
        n_devices=n_devices,
        loss_chunk_size=loss_chunk_size,
        use_memory_efficient_trainer=use_memory_efficient_trainer,
        adasum=adasum,
        random_seed=random_seed,
        callbacks=callbacks,
    )

    steps_to_go = steps - loop.step
    if steps_to_go > 0:
        loop.run(steps_to_go)
    else:
        log("Stop training, already reached the total training steps %d" % steps)
    return loop


class LoopTrainer:
    """Lightweight trainer facade built on :class:`training.Loop`."""

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        lr_schedule,
        inputs,
        output_dir=None,
        n_devices=None,
        random_seed=None,
        metrics=None,
        init_checkpoint=None,
    ):
        if callable(inputs):
            inputs = inputs()
        self._n_devices = n_devices or fastmath.local_device_count()
        train_tasks, eval_tasks = _prepare_tasks(
            inputs,
            loss_fn,
            optimizer,
            lr_schedule_fn=lambda: lr_schedule,
            n_devices=self._n_devices,
            eval_steps=1,
            eval_frequency=1,
            permanent_checkpoint_frequency=None,
            use_memory_efficient_trainer=False,
            metrics=metrics,
            additional_train_tasks=None,
            additional_eval_tasks=None,
            additional_eval_streams=None,
        )
        model_train = model(mode="train")
        eval_model = model(mode="eval")
        if init_checkpoint:
            model_train.init_from_file(init_checkpoint, weights_only=True)
            eval_model.init_from_file(init_checkpoint, weights_only=True)
        self._loop = training.Loop(
            model_train,
            train_tasks,
            eval_model=eval_model,
            eval_tasks=eval_tasks,
            output_dir=output_dir,
            n_devices=self._n_devices,
            random_seed=random_seed,
            eval_at=lambda _: False,
        )

    @property
    def model_weights(self):
        return self._loop.model.weights

    @property
    def step(self):
        return self._loop.step

    def train_epoch(self, n_steps: int, n_eval_steps: int):
        self._loop.eval_tasks[0]._n_eval_batches = n_eval_steps  # pylint: disable=protected-access
        self._loop.run(n_steps)
        self._loop.run_evals()

    def close(self):
        return self._loop.close()


__all__ = [
    "LoopTrainer",
    "NamedStream",
    "epochs",
    "log",
    "num_devices",
    "train",
]
