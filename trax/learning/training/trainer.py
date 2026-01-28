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

"""Training loop and task helpers for supervised learning in Trax.

Key classes:

  - :py:class:`Loop`: Core training loop for an n-step training session,
    starting from random initialization.

  - :py:class:`TrainingTask`: Labeled data + feedback mechanism (loss function w/
    optimizer) for modifying a model's weights.

  - :py:class:`EvaluationTask`: How and when to measure model performance as a
    function of training step number.
"""
import collections
import contextlib
import functools
import gzip as gzip_lib
import inspect
import itertools
import os
import pickle
import random
import sys
import time

from typing import Iterable, List, Optional, Sequence

import gin
import jax
import numpy as np
import psutil
import tensorflow as tf

from absl import logging

import trax.utils.board as board

from trax import fastmath
from trax import layers as tl
from trax import optimizers as trax_opt
from trax.data.preprocessing import inputs as trax_inputs
from trax.fastmath import numpy as jnp
from trax.fastmath import random as jax_random
from trax.layers import base
from trax.learning.supervised import common
from trax.learning.supervised import lr_schedules as lr
from trax.optimizers import base as optim_base
from trax.utils import shapes

from .engines import base as trainer_base
from .task import EvaluationTask, TrainingTask
from .utils import history as trax_history
from .utils import orchestration

_Evaluator = collections.namedtuple("_Evaluator", ["weights", "state", "metrics_fn"])
_DEFAULT_METRICS = common.default_metrics()


class ScheduleBuilder:
    """Builds checkpoint and eval schedules for the training loop."""

    def build_checkpoint_managers(
        self, task, checkpoint_at=None, permanent_checkpoint_at=None
    ):
        default_at = _at_step_1_and_every_nth_step(task.n_steps_per_checkpoint)
        permanent_default_at = _at_step_1_and_every_nth_step(
            task.n_steps_per_permanent_checkpoint
        )
        checkpoint_at = checkpoint_at or default_at
        permanent_checkpoint_at = permanent_checkpoint_at or permanent_default_at
        return (
            checkpoint_at,
            permanent_checkpoint_at,
            orchestration.CheckpointManager(checkpoint_at),
            orchestration.CheckpointManager(permanent_checkpoint_at),
        )

    def build_eval_schedule(self, task, eval_at=None, fallback=None):
        default_at = fallback or _at_step_1_and_every_nth_step(
            task.n_steps_per_checkpoint
        )
        return eval_at or default_at


def ensure_optimizer_instance(optimizer):
    """Ensures an optimizer factory is materialized before use."""

    if isinstance(optimizer, optim_base.Optimizer):
        return optimizer
    if isinstance(optimizer, functools.partial):
        return optimizer()
    if inspect.isclass(optimizer) and issubclass(optimizer, optim_base.Optimizer):
        return optimizer()
    return optimizer


class Loop:
    """Loop that can run for a given number of steps to train a supervised model.

    Can train the model on multiple tasks by interleaving updates according to the
    ``which_task`` argument.

    The typical supervised training process randomly initializes a model and
    updates its weights via feedback (loss-derived gradients) from a training
    task, by looping through batches of labeled data. A training loop can also
    be configured to run periodic evals and save intermediate checkpoints.

    For speed, the implementation takes advantage of JAX's composable function
    transformations (specifically, ``jit`` and ``grad``). It creates JIT-compiled
    pure functions derived from variants of the core model; schematically:

      - training variant: `jit(grad(pure_function(model+loss)))`
      - evals variant: `jit(pure_function(model+evals))`

    In training or during evals, these variants are called with explicit
    arguments for all relevant input data, model weights/state, optimizer slots,
    and random number seeds:

      - batch: labeled data
      - model weights/state: trainable weights and input-related state (e.g., as
        used by batch norm)
      - optimizer slots: weights in the optimizer that evolve during the training
        process
      - random number seeds: JAX PRNG keys that enable high-quality, distributed,
        repeatable generation of pseudo-random numbers
    """

    def __init__(
        self,
        model,
        tasks,
        eval_model=None,
        eval_tasks=None,
        output_dir=None,
        checkpoint_at=None,
        checkpoint_low_metric=None,
        checkpoint_high_metric=None,
        permanent_checkpoint_at=None,
        eval_at=None,
        which_task=None,
        n_devices=None,
        random_seed=None,
        loss_chunk_size=0,
        use_memory_efficient_trainer=False,
        adasum=False,
        callbacks=None,
        checkpoint_store=None,
        metrics_sink=None,
        registry=None,
        host_device_initializer=None,
        seed_manager_factory=None,
        model_initializer=None,
        schedule_builder=None,
        callback_assembler=None,
        training_orchestrator_ctor=orchestration.TrainingOrchestrator,
    ):
        """Configures a training ``Loop``, including a random initialization.

        Args:
          model: Trax layer, representing the core model to be trained. Loss
              functions and eval functions (a.k.a. metrics) are considered to be
              outside the core model, taking core model output and data labels as
              their two inputs.
          tasks: List of :py:class:`TrainingTask` instances, which define the training
              data, loss function, and optimizer to be used in respective tasks in
              this training loop. It can also be a single :py:class:`TrainingTask`
              instance which is treated in the same way as a singleton list.
          eval_model: Optional Trax layer, representing model used for evaluation,
              e.g., with dropout turned off. If ``None``, the training model (model)
              will be used.
          eval_tasks: List of :py:class:`EvaluationTask` instances which define how to
              evaluate the model: which validation data to use and which metrics to
              report. Evaluation on each of the tasks and will run and be reported
              separately which allows to score a model on different subtasks. This
              argument can also be ``None``, in which case no evals will be run, or
              a single :py:class:`EvaluationTask`, which wil be treated in the same way
              as a singleton list.
          output_dir: Path telling where to save outputs (evals and checkpoints).
              Can be ``None`` if both ``eval_task`` and ``checkpoint_at`` are
              ``None``.
          checkpoint_at: Function (integer --> boolean) telling, for step n, whether
              that step should have its checkpoint saved. If ``None``, the default
              is periodic checkpointing at ``task.n_steps_per_checkpoint``.
          checkpoint_low_metric: Name of metric, or None. The metric name must
              be one of the metric names from the evals in ``eval_tasks``. At
              checkpoint times determined by ``checkpoint_at``, a separate
              specially named checkpoint will be saved (overwriting any previous
              version) if the designated metric reaches a value less than or equal
              to any previous recorded low value. No such checkpoint is saved if
              arg value is `None`.
          checkpoint_high_metric: Name of metric, or None. The metric name must
              be one of the metric names from the evals in ``eval_tasks``. At
              checkpoint times determined by ``checkpoint_at``, a separate
              specially named checkpoint will be saved (overwriting any previous
              version) if the designated metric reaches a value greater than or
              equal to any previous recorded high value. No such checkpoint is
              saved if arg value is `None`.
          permanent_checkpoint_at: Function (integer --> boolean) telling,
              for step n, whether that step should have its checkpoint saved
              permanently. If ``None``, the default is periodic checkpointing at
              ``task.n_steps_per_permanent_checkpoint``.
          eval_at: Function (integer --> boolean) that says, for training step n,
              whether that step should run evals. If ``None``, run evals on the
              first step and on every N'th step, as determined by the first
              training task.
          which_task: Function (integer --> integer) indicating which task should be
              used at which training step. Can be set to ``None`` in single-task
              training.
          n_devices: integer or ``None``, the number of devices for this
              computation.
          random_seed: the random seed to use; time/os dependent if ``None``
              (default).
          loss_chunk_size: int, if > 0 use chunks of this size to make loss
            computation more more memory-efficient.
          use_memory_efficient_trainer: whether to use a special memory-efficient
            trainers; if set to 2, the memory efficiency if very aggressive
          adasum: if True, use adaptive summation for multi-device gradients
          callbacks: List of subclasses of StepCallback to call on training
            steps.
          checkpoint_store: Optional :py:class:`CheckpointStore` or registered
            name to control checkpoint persistence.
          metrics_sink: Optional :py:class:`MetricsSink` or registered name for
            metric emission.
          registry: Optional registry module exposing registries for checkpoint
            stores and metric sinks.
        """
        host_device_initializer = host_device_initializer or orchestration.HostAndDeviceInitializer(
            init_host_and_devices
        )
        seed_manager_factory = seed_manager_factory or orchestration.SeedManagerFactory(
            use_memory_efficient_trainer
        )
        callback_assembler = callback_assembler or orchestration.CallbackAssembler()
        schedule_builder = schedule_builder or ScheduleBuilder()

        (
            self._is_chief,
            self._n_hosts,
            self._device_manager,
            initial_rng,
        ) = host_device_initializer.initialize(n_devices, random_seed)
        self._n_devices = self._device_manager.n_devices
        if use_memory_efficient_trainer:
            initial_rng = tl.on_cpu(initial_rng)
        self._seed_manager = seed_manager_factory.create(initial_rng)

        # Handle single task case without lists too.
        if not isinstance(tasks, (list, tuple)):
            tasks = [tasks]

        if not tasks:
            raise ValueError("Must provide at least one training task.")
        if eval_tasks is None:
            eval_tasks = []
            eval_at = _never
        else:
            if not isinstance(eval_tasks, (list, tuple)):
                eval_tasks = [eval_tasks]

        expanded_output_dir = os.path.expanduser(output_dir) if output_dir else None
        self._registry = registry or board.base
        self._checkpoint_store = self._init_checkpoint_store(
            checkpoint_store, expanded_output_dir
        )
        self._metrics_sink = self._init_metrics_sink(metrics_sink, expanded_output_dir)

        self._tasks = tasks
        self._model = model
        self._eval_model = eval_model or model

        model_initializer = model_initializer or orchestration.ModelInitializer(
            shapes.signature(tasks[0].sample_batch),
            use_memory_efficient_trainer,
            is_uninitialized=_is_uninitialized,
        )

        self._use_memory_efficient_trainer = use_memory_efficient_trainer
        self._loss_chunk_size = loss_chunk_size
        self._adasum = adasum
        # TODO(lukaszkaiser): can we have different eval models and save memory?
        if use_memory_efficient_trainer:
            assert len(tasks) == 1, "only single task supported for now"
            self._eval_model = model

        if expanded_output_dir is not None:
            self._output_dir = expanded_output_dir
            self._checkpoint_store.makedirs(self._output_dir)
            trax_inputs.load_data_counters(self._output_dir)
        else:
            self._output_dir = None

        (
            self._checkpoint_at,
            self._permanent_checkpoint_at,
            self._checkpoint_manager,
            self._permanent_checkpoint_manager,
        ) = schedule_builder.build_checkpoint_managers(
            tasks[0], checkpoint_at, permanent_checkpoint_at
        )
        self._checkpoint_low_metric = checkpoint_low_metric
        self._checkpoint_high_metric = checkpoint_high_metric

        # Prepare training components.
        self._step = 0
        self._history = trax_history.History()
        if which_task is None:
            # If which task is not passed, then we permute tasks one by one.
            # If len(tasks) = 1, then which_task is a constant function equal to 0.
            def which_task(n):
                return n % len(tasks)
        self._which_task = which_task

        # Initialize using the given random seed.
        # NOTE: If random_seed is None then the rng will be different on
        # different hosts, leading to different weights on the different hosts.
        self._batch_signature = shapes.signature(tasks[0].sample_batch)
        self._model.rng = self.new_rng()
        if not use_memory_efficient_trainer:
            self._eval_model.rng = self.new_rng()
        model_initializer.initialize(
            self._model,
            self._eval_model,
            sync_fn=None if use_memory_efficient_trainer else None,
        )

        # To handle the above case (i.e. random_seed = None), we psum the weights
        # and state and average them.
        # NOTE: This adds time (how much?) so we prefer not to do it if it is
        # unnecessary, i.e. random_seed was set.
        # NOTE: Averaging the weights across devices can screw up the initial weight
        # statistics.
        # TODO(pkozakowski): Broadcast from one of the devices instead?
        if (
            random_seed is None
            and self._n_hosts > 1
            and not use_memory_efficient_trainer
        ):
            logging.info("Syncing weights/state across %d hosts.", self._n_hosts)
            # Do self._sync_weights_and_state_across_hosts() but layer-by-layer
            # to save memory.
            blocks, last_layer = trainer_base.extract_reversible_blocks([self._model])
            all_layers = []
            for std_layer, rev_layers in blocks:
                all_layers.append(tl.Serial(std_layer))
                all_layers.extend(rev_layers)
            all_layers.append(last_layer)
            for layer in all_layers:
                weights_and_state = (layer.weights, layer.state)
                if not _is_empty(weights_and_state):
                    layer.weights, layer.state = tl.on_cpu(
                        self._unreplicate(
                            _make_weights_and_state_same_across_hosts(
                                self._for_n_devices(weights_and_state)
                            )
                        )
                    )

        # Create the optimizer for the training loss function.
        self._trainer_per_task = tuple(self._init_trainer(task) for task in tasks)

        # Sync layers weights/state in memory effcient trainers layers.
        if random_seed is None and self._n_hosts > 1 and use_memory_efficient_trainer:
            logging.info("Syncing layers across %d hosts.", self._n_hosts)
            for layer in self._trainer_per_task[0].all_layers:
                weights_and_state = (layer.weights, layer.state)
                if not _is_empty(weights_and_state):
                    layer.weights, layer.state = tl.on_cpu(
                        self._unreplicate(
                            _make_weights_and_state_same_across_hosts(
                                self._for_n_devices(weights_and_state)
                            )
                        )
                    )

        # Load checkpoint if it exists.
        self.load_checkpoint()

        # Prepare eval components.
        self._eval_at = schedule_builder.build_eval_schedule(
            tasks[0], eval_at, fallback=self._checkpoint_at
        )
        self._eval_tasks = eval_tasks
        loss_names = [task.loss_name for task in self._tasks]
        metric_names = [
            name  # pylint: disable=g-complex-comprehension
            for eval_task in self._eval_tasks
            for name in eval_task.metric_names
        ]
        self._rjust_len = max(map(len, loss_names + metric_names))
        evaluator_factory = orchestration.EvaluatorFactory(self._init_evaluator)
        self._evaluator_per_task = evaluator_factory.create(self._eval_tasks)

        if self._output_dir is None:
            _log("Will not write evaluation metrics, because output_dir is None.")

        def task_output_dir(task_index, task_list):
            if self._output_dir is not None:
                if len(task_list) < 2:
                    output_dir = self._output_dir
                else:
                    output_dir = os.path.join(
                        self._output_dir,
                        task_list[task_index].export_prefix or str(task_index),
                    )
                self._checkpoint_store.makedirs(output_dir)
                return output_dir
            else:
                return None

        self._output_dir_per_eval_task = [
            task_output_dir(i, eval_tasks) for i in range(len(eval_tasks))
        ]
        self._output_dir_per_train_task = [
            task_output_dir(i, tasks) for i in range(len(tasks))
        ]

        callbacks = callbacks or []
        callback_instances = [callback_class(self) for callback_class in callbacks]
        self._callback_pipeline = callback_assembler.assemble(callback_instances)
        self._orchestrator = training_orchestrator_ctor(
            device_manager=self._device_manager,
            seed_manager=self._seed_manager,
            callback_pipeline=self._callback_pipeline,
        )

    def _init_checkpoint_store(self, checkpoint_store, base_dir):
        if isinstance(checkpoint_store, board.base.CheckpointStore):
            return checkpoint_store
        if isinstance(checkpoint_store, str):
            return self._registry.CHECKPOINT_STORE_REGISTRY.get(
                checkpoint_store, base_dir=base_dir
            )
        if checkpoint_store is None:
            return self._registry.CHECKPOINT_STORE_REGISTRY.get(
                "local", base_dir=base_dir
            )
        return checkpoint_store

    def _init_metrics_sink(self, metrics_sink, base_dir):
        if isinstance(metrics_sink, board.base.MetricsSink):
            return metrics_sink
        if isinstance(metrics_sink, str):
            return self._registry.METRICS_SINK_REGISTRY.get(
                metrics_sink, base_dir=base_dir
            )
        if metrics_sink is None:
            if base_dir is None:
                return board.base.NullMetricsSink(base_dir=base_dir)
            return self._registry.METRICS_SINK_REGISTRY.get(
                "jaxboard", base_dir=base_dir
            )
        return metrics_sink

    def _init_trainer(self, task):
        """Initializes the per-task trainers."""
        # Build the per-task model, sharing weights with other tasks.
        if not self._use_memory_efficient_trainer:
            optimizer = ensure_optimizer_instance(task.optimizer)
            task._optimizer = optimizer
            model_in_training = _model_with_ends(
                self._model, [task.loss_layer], shapes.signature(task.sample_batch)
            )
            if base.N_WEIGHTS_SHARDS > 1:
                sharded_weights = fastmath.nested_map(
                    lambda x: x[0], tl.shard(model_in_training.weights)
                )
                optimizer.tree_init(sharded_weights)
            else:
                optimizer.tree_init(model_in_training.weights)
            return trainer_base.TrainingEngine(
                model_in_training, optimizer, adasum=self._adasum
            )
        # In the memory-efficient path, we initialize the model here.
        optimizer_fn = task.optimizer
        if isinstance(optimizer_fn, optim_base.Optimizer):
            def optimizer_fn(opt=optimizer_fn):
                return opt
        task._optimizer = optimizer_fn
        blocks, loss_layer = trainer_base.extract_reversible_blocks(
            [self._model, task.loss_layer], loss_chunk_size=self._loss_chunk_size
        )
        rng = self._model.rng
        sig = shapes.signature(task.sample_batch)
        trainer_base.init_reversible_blocks(blocks, loss_layer, sig, rng)
        # TODO(lukaszkaiser): here optimizer is a function, revisit this.
        return trainer_base.ReversibleSerialTrainer(
            blocks,
            loss_layer,
            optimizer_fn,
            free_accelerators_on_step=(self._use_memory_efficient_trainer == 2),
            adasum=self._adasum,
        )

    def _init_evaluator(self, eval_task):
        """Initializes the per-task evaluator."""
        model_with_metrics = _model_with_metrics(self._eval_model, eval_task)
        if self._use_memory_efficient_trainer:
            return _Evaluator(
                weights=tl.on_cpu(model_with_metrics.weights[1]),
                state=tl.on_cpu(model_with_metrics.state[1]),
                metrics_fn=_accelerate_model_with_metrics(model_with_metrics, 0),
            )
        else:
            return _Evaluator(
                # Replicate the eval part of weights and state.
                weights=self._for_n_devices(model_with_metrics.weights[1]),
                state=self._for_n_devices(model_with_metrics.state[1]),
                metrics_fn=_accelerate_model_with_metrics(
                    model_with_metrics, self.n_devices
                ),
            )

    def _sync_weights_and_state_across_hosts(self):
        """Sync weights and state across all the hosts in the computation."""

        if logging.vlog_is_on(1):
            logging.debug(
                "Input training weights shape: %s",
                fastmath.nested_map(lambda x: x.shape, self._model.weights),
            )
            logging.debug("Input training weights: %s", self._model.weights)
            logging.debug("Input training state: %s", self._model.state)
            logging.debug("Input eval weights: %s", self._eval_model.weights)
            logging.debug("Input eval state: %s", self._eval_model.state)

        (
            self._model.weights,
            self._model.state,
            self._eval_model.weights,
            self._eval_model.state,
        ) = self._unreplicate(
            _make_weights_and_state_same_across_hosts(
                self._for_n_devices(
                    (
                        self._model.weights,
                        self._model.state,
                        self._eval_model.weights,
                        self._eval_model.state,
                    )
                )
            )
        )

        if logging.vlog_is_on(1):
            logging.debug(
                "Output training weights shape: %s",
                fastmath.nested_map(lambda x: x.shape, self._model.weights),
            )
            logging.debug("Output training weights: %s", self._model.weights)
            logging.debug("Output training state: %s", self._model.state)
            logging.debug("Output eval weights: %s", self._eval_model.weights)
            logging.debug("Output eval state: %s", self._eval_model.state)

    def run(self, n_steps=1):
        """Runs this training loop for n steps.

        Optionally runs evals and saves checkpoints at specified points.

        Args:
          n_steps: Stop training after completing n steps.
        """
        with self._open_summary_writers() as (
            train_summary_writers,
            eval_summary_writers,
        ):
            process = psutil.Process(os.getpid())
            loss_acc, step_acc = 0.0, 0
            start_time = time.time()
            optimizer_metrics_acc = collections.defaultdict(float)
            for i in range(n_steps):
                prev_task_index = self._which_task(self._step)
                self._step += 1
                task_index = self._which_task(self._step)
                task_changed = task_index != prev_task_index

                if task_changed:
                    loss_acc, step_acc = 0.0, 0

                loss, optimizer_metrics = self._run_one_step(task_index, task_changed)

                # optimizer_metrics and loss are replicated on self.n_devices, a few
                # metrics are replicated (ex: gradients_l2, weights_l2) - i.e. they are
                # the same across devices, whereas some (ex: loss) aren't because they
                # are different on different devices (due to different data).
                # Taking the average does the correct thing in both the cases.
                #
                # NOTE: Only the weights and gradients are synced across the hosts. This
                # implies the loss here is averaged from this hosts' devices and not
                # across all hosts.
                optimizer_metrics, loss = fastmath.nested_map(
                    functools.partial(tl.mean, self._n_devices),
                    (optimizer_metrics, loss),
                )

                loss_acc += loss
                # Log loss every 50 steps, every step in memory-efficient trainers.
                if self._step % 50 == 0 or self._use_memory_efficient_trainer:
                    self._log_step("Loss: %.4f" % loss, stdout=False)
                step_acc += 1
                for metric_name, value in optimizer_metrics.items():
                    optimizer_metrics_acc[metric_name] += value

                # TODO(yuwenyan): Finds a way to log the last round eval step in
                # history.
                #
                # Right now, the last round eval log is missing in history since the
                # checkpoint is saved before it. However sometimes the eval step will
                # fail for some reasons, and it's not acceptable to loose the whole
                # checkpoint in this case. Stays with the old way for now, and fixes it
                # when the checkpoint format is changed to storing weights separately
                # from a small file with history and other data.
                if self._checkpoint_manager.should_save(self.step):
                    self.save_checkpoint("model")
                if self._permanent_checkpoint_manager.should_save(self.step):
                    self.save_checkpoint(f"model_{self.step}")
                if self._eval_at(self.step):
                    logging.info(
                        "cpu memory use (MB): %.2f",
                        process.memory_info().rss / float(1024 * 1024),
                    )
                    elapsed_time = time.time() - start_time
                    self._log_training_progress(
                        task=self._tasks[task_index],
                        total_loss=loss_acc,
                        n_steps=step_acc,
                        elapsed_time=elapsed_time,
                        optimizer_metrics=optimizer_metrics_acc,
                        summary_writer=train_summary_writers[task_index],
                    )
                    self.run_evals(eval_summary_writers)
                    loss_acc, step_acc = 0.0, 0
                    start_time = time.time()
                    optimizer_metrics_acc = collections.defaultdict(float)

                # For the current step, after all evals are run and recorded in the
                # event history, check if we need to save special checkpoints because
                # of a new low metric value or a new high metric value.
                if self._checkpoint_manager.should_save(self.step):
                    if self._checkpoint_low_metric is not None and self._at_lowest():
                        self.save_checkpoint(f"lowest_{self._checkpoint_low_metric}")
                    if self._checkpoint_high_metric is not None and self._at_highest():
                        self.save_checkpoint(f"highest_{self._checkpoint_high_metric}")

        # Store the final values back into their respective objects, for testing
        # or other inspection/use.
        #
        # We keep the standard model weights/state unreplicated and
        # tl.Accelerate(model) will carry the replicated weights/state.
        # TODO(afrozm): Try to use tl.Accelerate(model) everywhere in the Loop.
        self._eval_model.weights = self._model.weights

    def close(self):
        """Closes resources associated with the loop."""

        # Loop does not hold external resources today.
        return None

    def _at_lowest(self):
        low_items = self.history.get("eval", f"metrics/{self._checkpoint_low_metric}")
        vals = [float(obj[1]) for obj in low_items]
        return vals[-1] == min(vals)

    def _at_highest(self):
        high_items = self.history.get("eval", f"metrics/{self._checkpoint_high_metric}")
        vals = [float(obj[1]) for obj in high_items]
        return vals[-1] == max(vals)

    @property
    def step(self):
        """Returns current step number in this training session."""
        return self._step

    @property
    def history(self):
        """Returns history in this training session."""
        return self._history

    @property
    def n_devices(self):
        """Returns the number of devices to be used in this computation."""
        return self._n_devices

    @property
    def is_chief(self):
        """Returns true if this Loop is the chief."""
        return self._is_chief

    @property
    def model(self):
        """Returns the model that is training."""
        return self._model

    @property
    def tasks(self):
        """Returns the training tasks."""
        return self._tasks

    @property
    def eval_model(self):
        """Returns the model used for evaluation."""
        return self._eval_model

    @property
    def eval_tasks(self):
        """Returns the evaluation tasks."""
        return self._eval_tasks

    @property
    def output_dir(self):
        """Returns the output directory."""
        return self._output_dir

    def new_rng(self):
        """Returns a new single-use random number generator (JAX PRNG key)."""
        return self._seed_manager.new_rng()

    def _for_n_devices(self, x):
        """Replicates/broadcasts ``x`` for n devices if ``self.n_devicess > 1``."""
        return self._device_manager.for_n_devices(x)

    def _unreplicate(self, x):
        return self._device_manager.unreplicate(x)

    def _reshape_by_device(self, x):
        return self._device_manager.reshape_by_device(x)

    def update_weights_and_state(self, weights=None, state=None):
        """Updates the weights and state of the trained model.

        Sends this data both to the singleton model accessible via Loop.model
        and to the replicated model on the accelerator.

        Useful when the weights or state are modified outside of training, e.g.
        during data collection in RL agents.

        Args:
          weights: Model weights or ``None``. If ``None``, don't set.
          state: Model state or ``None``. If ``None``, don't set.
        """
        for trainer in self._trainer_per_task:
            acc_model_with_loss = trainer.accelerated_model_with_loss
            if weights is not None:
                self._model.weights = weights
                acc_model_with_loss.replicate_weights(trainer.model_with_loss.weights)
            if state is not None:
                self._model.state = state
                acc_model_with_loss.replicate_state(trainer.model_with_loss.state)

    def _run_one_step(self, task_index, task_changed):
        """Updates model weights/state and optimizer slots by running one step.

        Args:
          task_index (int): Index of the task to train on.
          task_changed (bool): Whether the state has changed since the last step.

        Returns:
          Tuple (loss, stats) with loss value from one step
          of training and stats, the current optimizer statistics.
        """
        step = self.step
        trainer = self._trainer_per_task[task_index]
        loss, stats = self._orchestrator.run_step(
            trainer,
            self._tasks[task_index],
            step,
            task_changed,
            sync_fn=lambda: self.update_weights_and_state(
                self._model.weights, self._model.state
            ),
        )

        return (loss, stats)

    def _log_training_progress(
        self, task, total_loss, n_steps, elapsed_time, optimizer_metrics, summary_writer
    ):
        """Logs training related metrics.

        Logs:
         * current learning rate,
         * steps per second,
         * average training loss,
         * average metrics returned from the optimizer
        to the provided summary writer. Training loss is also logged to stdout.

        Args:
          task: Current training task.
          total_loss: Total training loss accumulated over n_steps training steps.
          n_steps: Number of steps over which the metrics were accumulated.
          elapsed_time: Time of execution of n_steps training steps.
          optimizer_metrics: Dict from optimizer metric name to metric values.
          summary_writer: Jaxboard summary writer for saving provided metrics.
        """
        # only here do avoid potential divide-by-0
        n_steps = max(1, n_steps)
        _log("")  # Separator for visibility on terminals.
        if self.step == 1:
            self._log_n_weights()
        self._log_step("Ran %d train steps in %0.2f secs" % (n_steps, elapsed_time))
        self.log_summary(
            {task.loss_name: total_loss / float(n_steps)},
            summary_writer,
            "metrics/",
            "train",
        )
        if self.step == 1:
            self._save_gin(summary_writer)
        train_parameters = {
            "learning_rate": task.learning_rate(self.step),
            "steps per second": n_steps / elapsed_time,
        }
        # Average optimizer_metrics over n_steps.
        optimizer_metrics = {k: v / n_steps for k, v in optimizer_metrics.items()}
        train_parameters.update(optimizer_metrics)
        self.log_summary(
            train_parameters, summary_writer, "training/", "train", stdout=False
        )

    def _save_gin(self, summary_writer):
        """ "Saves the operative gin config."""
        if not self.is_chief or self._output_dir is None:
            return
        config_path = os.path.join(self._output_dir, "config.gin")
        config_str = gin.operative_config_str()
        with self._checkpoint_store.open(config_path, "w") as f:
            f.write(config_str)
        if summary_writer is not None:
            summary_writer.log_text(
                "gin_config",
                board.jaxboard.markdownify_operative_config_str(config_str),
                self.step,
            )

    def _log_n_weights(self):
        """ "Logs the number of weights in the training model."""

        def _size(x):
            try:
                return x.size
            except Exception:  # pylint: disable=broad-except
                return 0

        sizes = fastmath.nested_map(_size, self._model.weights)
        total_size = sum(fastmath.tree_flatten(sizes))
        total_size *= base.N_WEIGHTS_SHARDS
        self._log_step("Total number of trainable weights: %d" % total_size)

    # TODO(afrozm): Fix multi-host evals, right now the reported numbers in the
    #   summary writer are only from the chief and not averaged across hosts.
    def run_evals(self, summary_writers=None):
        """Runs and records evals for this training session.

        Args:
          summary_writers: List of per-task metrics sinks to log metrics.
        """
        if summary_writers is None:
            summary_writers = (None,) * len(self._eval_tasks)

        self._eval_model.weights = self._model.weights
        self._eval_model.state = self._model.state

        def recursively_look_for_printable_states(state):
            if isinstance(state, (tuple, list)):
                for substate in state:
                    for item in recursively_look_for_printable_states(substate):
                        yield item
            if isinstance(state, dict):
                for key, value in state.items():
                    if isinstance(key, str) and key.startswith("summary_"):
                        for device_id, device_value in enumerate(value):
                            yield (
                                "device{}/{}".format(device_id, key[len("summary_") :]),
                                device_value,
                            )

        # The most recently trained weights are in this trainers, use those for eval.
        cur_train_task_index = self._which_task(self._step)
        trainer = self._trainer_per_task[cur_train_task_index]

        for eval_task_index in range(len(self._eval_tasks)):
            eval_task = self._eval_tasks[eval_task_index]
            evaluator = self._evaluator_per_task[eval_task_index]
            if eval_task is None:
                continue

            # Extract the actual model weights and state, excluding the loss layer.
            if self._use_memory_efficient_trainer:
                model_weights, model_state = self._model.weights, self._model.state
            else:
                model_weights = trainer.accelerated_model_with_loss.weights[0]
                model_state = trainer.accelerated_model_with_loss.state[0]

            # evaluator.{weights,state} are already replicated.
            metrics_weights = (model_weights, evaluator.weights)
            metrics_state = (model_state, evaluator.state)

            n_batches = eval_task.n_eval_batches
            n_metrics = len(eval_task.metrics)
            sums = np.zeros((n_metrics,))
            for _ in range(n_batches):
                rng = self.new_rng()
                batch = eval_task.next_batch()
                metric_values, _ = evaluator.metrics_fn(
                    batch, metrics_weights, metrics_state, rng
                )
                sums += metric_values
            averages = sums / n_batches
            all_metrics = dict(zip(eval_task.metric_names, averages))
            summary_writer = summary_writers[eval_task_index]
            self.log_summary(all_metrics, summary_writer, "metrics/", "eval")
            summary_metrics = dict(recursively_look_for_printable_states(model_state))
            self.log_summary(summary_metrics, summary_writer, "summary_", "eval")

    def log_summary(
        self, values, summary_writer, value_prefix, log_prefix, stdout=True
    ):
        """Logs and saves provided metrics.

        Args:
          values: Dict from metric name to metric value.
          summary_writer: Metrics sink implementation.
          value_prefix: String appended in front of summary_writer entries.
          log_prefix: String appended in front of logs.
          stdout: Boolean saying if logs should be logged to stdout as well.
        """
        history = self._history
        should_write_summaries = self.is_chief and summary_writer is not None
        for name, value in values.items():
            full_name = value_prefix + name
            s = tuple(jnp.shape(value))
            if not s:
                self._log_step(
                    "%s %s | % .8f"
                    % (log_prefix.ljust(5), name.rjust(self._rjust_len), value),
                    stdout=stdout,
                )
                if should_write_summaries:
                    summary_writer.log_scalar(full_name, value, self.step)
            else:
                if should_write_summaries:
                    summary_writer.log_image(full_name, value, self.step)
            if history:
                history.append(log_prefix, full_name, self.step, value)
        if should_write_summaries:
            summary_writer.flush()

    def _log_step(self, msg, stdout=True):
        """Logs message, labeled with the current training step number."""
        _log("Step % 6d: %s" % (self.step, msg), stdout=stdout)

    def save_checkpoint(self, basename):
        """Saves checkpoint (multiple files) to disk for the current training step.

        Saving a checkpoint will overwrite any previous checkpoint saved with the
        same ``basename``. Use differing ``basename`` values to save multiple
        checkpoints or multiple copies of the same checkpoint.

        Args:
          basename: Basename for saving a checkpoint. Full file paths for the saved
              checkpoint will combine the output dir, basename, and relevant file
              extensions (e.g., `.weights.npy.gz`).
        """
        if self._output_dir is None:
            _log("Did not save checkpoint as output_dir is None")
            return

        trax_inputs.save_data_counters(self._output_dir)
        if not self.is_chief:
            _log("Did not save checkpoint as we are not chief.")
            return

        dir_and_basename = self._checkpoint_store.resolve(
            os.path.join(self._output_dir, basename)
        )
        pkl_file = dir_and_basename + ".pkl.gz"

        _log("Saving checkpoint to %s" % pkl_file, stdout=False)
        weights = self._model.weights
        if base.N_WEIGHTS_SHARDS > 1:
            weights = self._trainer_per_task[0].accelerated_model_with_loss.weights
            weights = tl.unshard(weights)
        state = self._model.state
        compresslevel = 0 if self._use_memory_efficient_trainer else 2
        # Serialize optimizer slots.
        for i, trainer in enumerate(self._trainer_per_task):
            flat_slots = _flatten_and_remove_empty(trainer.slots)
            tl.np_to_file(
                self._to_bits(flat_slots),
                f"{dir_and_basename}.opt_slots{i}.npy.gz",
                compresslevel=compresslevel,
            )
        # We only need the input signature for the body, not for the loss layers.
        # That part is the same across tasks - take it from the first one.
        input_signature = self._batch_signature[: self._model.n_in]
        flat_weights, flat_state = tl.flatten_weights_and_state(weights, state)
        _, flat_eval_state = tl.flatten_weights_and_state(
            weights, self._eval_model.state
        )
        tl.np_to_file(
            self._to_bits(flat_weights),
            f"{dir_and_basename}.weights.npy.gz",
            compresslevel=compresslevel,
        )
        d = {
            "step": self.step,
            "flat_weights": compresslevel,  # for compatibility with older format
            "flat_state": flat_state,
            "flat_eval_state": flat_eval_state,
            "history": self._history.to_dict(),
            "slots_per_task": compresslevel,  # for compatibility with older format
            "input_signature": input_signature,
            "version_timestamp": "Mar-10-2021",  # To update in the future if needed.
        }
        pickle_to_store(self._checkpoint_store, d, pkl_file, gzip=True)
        _log("Checkpoint saved in %s" % pkl_file, stdout=False)

    def _to_bits(self, weights):
        """Converts a list of weights to bit-cast weights and their types."""
        # This is currently needed to pickle bfloat16 arrays from JAX.
        # TODO(lukaszkaiser): remove once it is not needed (the following unit test
        #   checks it: training_test/test_restores_step_bfloat16).
        if not fastmath.is_backend(fastmath.Backend.JAX):
            return weights
        bits = []
        for w in weights:
            if w.dtype == jnp.bfloat16:
                converted = jax.lax.bitcast_convert_type(w, np.uint16)
                bits.append(np.asarray(converted.astype(np.uint16)))
            else:  # for non-bfloat16 weights, be compatible with earlier checkpoints
                bits.append(np.asarray(w))
        return bits

    def _from_bits(self, bits):
        """Converts a list of bit-cast weights back to weights."""
        # This is the reverse of _to_bits, see above for explanation.
        if not fastmath.is_backend(fastmath.Backend.JAX):
            return bits
        weights = []
        for b in bits:
            if b.dtype == np.uint16:  # currently all uint16 are bfloat16s
                w = jax.lax.bitcast_convert_type(b, jnp.bfloat16)
                weights.append(np.asarray(w))
            else:
                weights.append(b)
        return weights

    def load_checkpoint(self, directory=None, filename=None):
        """Loads model weights and step from a checkpoint on disk.

        Args:
          directory: Directory with the checkpoint (self._output_dir by default).
          filename: Checkpoint file name (model.pkl.gz by default).
        """
        directory = directory or self._output_dir
        if directory is None:
            _log("Not loading as both directory and output_dir are None.", stdout=False)
            return
        filename = filename or "model"
        path = self._checkpoint_store.resolve(os.path.join(directory, filename))
        pkl_path = path + ".pkl.gz"
        if not self._checkpoint_store.exists(pkl_path):
            _log(
                f"Not loading as checkpoint file does not exist: {pkl_path}",
                stdout=False,
            )
            return
        _log("Loading checkpoint from %s" % pkl_path, stdout=False)
        d = unpickle_from_store(self._checkpoint_store, pkl_path, gzip=True)
        # Weights are stored in a separate non-pickled file in the new checkpoint
        # format. We support loading old checkpoints with this hack.
        # TODO(lukaszkaiser): remove the hack when not needed any more.
        if isinstance(d["flat_weights"], int):
            weights = tl.np_from_file(
                path + ".weights.npy.gz", compresslevel=d["flat_weights"]
            )
            d["flat_weights"] = weights
        else:
            d["flat_weights"] = d["flat_weights"]
        # The same holds for optimizer slots.
        if "slots" in d:  # Old checkpoints had just 'slots' for one task.
            if len(self._tasks) != 1:
                raise ValueError(
                    "Can't load a single-task checkpoint into a multitask Loop."
                )
            d["slots_per_task"] = [d["slots"]]
        # Read from separate files if optimizer slots are in them.
        if "slots_per_task" in d and isinstance(d["slots_per_task"], int):
            compresslevel = d["slots_per_task"]
            d["slots_per_task"] = []
            for i in range(len(self._trainer_per_task)):
                slots = tl.np_from_file(
                    path + f".opt_slots{i}.npy.gz", compresslevel=compresslevel
                )
                d["slots_per_task"].append(slots)
        for trainer, slots in zip(self._trainer_per_task, d["slots_per_task"]):
            matched_flat_slots = _match_by_shape(
                self._to_bits(_flatten_and_remove_empty(trainer.slots)),
                _flatten_and_remove_empty(slots),
            )
            try:
                matched_slots, _ = fastmath.tree_unflatten(
                    self._from_bits(matched_flat_slots),
                    trainer.slots,
                    copy_from_tree=[None, ()],
                )
                trainer.slots = matched_slots
            except IndexError:
                _log(
                    "Failed loading optimizer slots from checkpoint, using"
                    " newly initialized slots instead.",
                )
        self._step = d["step"]
        self._history = trax_history.History.from_dict(d["history"])
        # This is self._model.init_from_file but optimized to not re-read.
        input_signature = d["input_signature"]
        weights_and_state_sig = self._model.weights_and_state_signature(input_signature)
        flat_init_weights, flat_init_state = tl.flatten_weights_and_state(
            self._model.weights, self._model.state
        )
        if len(d["flat_weights"]) < len(flat_init_weights):
            _log("Checkpoint has less weights than the model, loading first ones.")
        matched_weights = _match_by_shape(
            self._to_bits(flat_init_weights), d["flat_weights"]
        )
        matched_weights = self._from_bits(matched_weights)
        try:
            restored_state = True
            matched_state = _match_by_shape(
                self._to_bits(flat_init_state), d["flat_state"]
            )
            matched_state = self._from_bits(matched_state)
            weights, state = tl.unflatten_weights_and_state(
                matched_weights, matched_state, weights_and_state_sig
            )
            self._model.state = state
        except IndexError:
            _log("Failed loading model state from checkpoint, loading weights only.")
            restored_state = False
            weights, _ = tl.unflatten_weights_and_state(
                matched_weights, (), weights_and_state_sig, weights_only=True
            )
        self._model.weights = weights
        self._eval_model.weights = self._model.weights
        # Restore eval model state; note: it's not always the same as train state.
        if restored_state:
            if "flat_eval_state" in d:
                flat_eval_state = d["flat_eval_state"]
            else:  # It wasn't saved in old checkpoints; remove this branch once done.
                flat_eval_state = d["flat_state"]
            _, eval_state = tl.unflatten_weights_and_state(
                matched_weights, flat_eval_state, weights_and_state_sig
            )
            self._eval_model.state = eval_state
        _log("Checkpoint loaded from %s" % pkl_path, stdout=False)

    @contextlib.contextmanager
    def _open_summary_writers(self):
        """Opens the metrics sinks wrapped by context manager.

        Yields:
          A pair (train_summary_writers, eval_summary_writers) of lists of
          metrics sinks wrapped in a GeneratorContextManager object. Elements of
          the lists correspond to the training and evaluation task directories
          created during initialization. If there was no output_dir provided,
          yields lists of Nones with the appropriate length.
        """
        if self._output_dir is not None:
            _log(f"Metrics will be written in {self._output_dir}.", stdout=False)
            train_writers = [
                self._metrics_sink.with_subpath(
                    os.path.relpath(os.path.join(output_dir, "train"), self._metrics_sink.base_dir)
                    if self._metrics_sink.base_dir
                    else os.path.join(output_dir, "train")
                )
                for output_dir in self._output_dir_per_train_task
            ]
            eval_writers = [
                self._metrics_sink.with_subpath(
                    os.path.relpath(os.path.join(output_dir, "eval"), self._metrics_sink.base_dir)
                    if self._metrics_sink.base_dir
                    else os.path.join(output_dir, "eval")
                )
                for output_dir in self._output_dir_per_eval_task
            ]
            try:
                yield (train_writers, eval_writers)
            finally:
                for writer in train_writers + eval_writers:
                    writer.close()
                _log(f"Metrics were written in {self._output_dir}", stdout=False)
        else:
            yield ([None] * len(self._tasks), [None] * len(self._eval_tasks))


@gin.configurable(module="trax.learning.trainer")
def num_devices(value=None):
    """Returns how many devices to use (if None, default, use all available)."""
    return value


def log(s, stdout=True):
    """Logs to TF logger and optionally stdout."""
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
    additional_train_tasks: Optional[Sequence[TrainingTask]],
    additional_eval_tasks: Optional[Sequence[EvaluationTask]],
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

    return (
        [train_task] + additional_train_tasks,
        [eval_task] + additional_eval_tasks,
    )


def _make_inputs_is_gin_configured() -> bool:
    try:
        gin.query_parameter("trax.data.make_inputs.train_stream")
    except (ValueError, TypeError):
        return False
    return True

@gin.configurable(module="trax.learning.trainer")
def train(
    output_dir,
    model=gin.REQUIRED,
    loss_fn=tl.WeightedCategoryCrossEntropy(),
    inputs=trax_inputs.make_inputs,
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
    additional_train_tasks: Optional[Sequence[TrainingTask]] = None,
    additional_eval_tasks: Optional[Sequence[EvaluationTask]] = None,
):
    """Train the model on the inputs via :class:`Loop`."""

    trainer_base.N_WEIGHTS_SHARDS = n_weights_shards
    if permanent_checkpoint_frequency is not None and permanent_checkpoints_at is not None:
        raise ValueError(
            'Only one of ["permanent_checkpoint_frequency", "permanent_checkpoints_at"] should be set.'
        )

    n_devices = num_devices() or fastmath.local_device_count()
    if inputs is trax_inputs.make_inputs and not _make_inputs_is_gin_configured():
        raise ValueError(
            "train(inputs=trax.data.make_inputs) requires a configured "
            "make_inputs (e.g. a partial from config) or an Inputs instance."
        )
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

    loop = Loop(
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


def _model_with_ends(model, end_layers, batch_signature):
    """Returns a model+ends layer built on an already initialized model.

    Ends can be loss or metric layers.

    Args:
      model: Layer with initialized weights and state.
      end_layers: List of end layers.
      batch_signature: Signature of the model input batch.

    Returns:
      An initialized, combined model+ends layer, preserving the initialization
      of ``model``.
    """
    # TODO(jonni): Redo this function as part of an initialization refactor?
    metrics_layer = tl.Branch(*end_layers)
    metrics_input_signature = model.output_signature(batch_signature)
    _, _ = metrics_layer.init(metrics_input_signature)

    model_with_metrics = tl.Serial(model, metrics_layer)
    return model_with_metrics


def _model_with_metrics(model, eval_task):
    """Returns a model+metrics layer built on an already initialized model.

    Args:
      model: Layer with initialized weights and state.
      eval_task: :py:class:`EvaluationTask` instance.

    Returns:
      An initialized, combined model+metrics layer, preserving the initialization
      of ``model``.
    """
    return _model_with_ends(
        model, eval_task.metrics, shapes.signature(eval_task.sample_batch)
    )


def _never(*args):
    """Returns False for all step numbers."""
    del args
    return False


def _at_step_1_and_every_nth_step(period):
    """A function that's true at 1 and n when n % period == 0."""
    if period is None:
        return lambda step_n: False

    def _at_1_and_periodically(step_n):
        return (step_n == 1) or (step_n > 0 and (step_n % period == 0))

    return _at_1_and_periodically


def _log(s, stdout=True):
    logging.info(s)
    if stdout:
        print(s)
        sys.stdout.flush()


def pickle_to_store(store, obj, file_path, gzip=False):
    """Pickle obj using the provided checkpoint store."""
    tmp_file_path = file_path + "._tmp_"
    with store.open(tmp_file_path, "wb") as f:
        if not gzip:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with gzip_lib.GzipFile(fileobj=f, compresslevel=2) as gzipf:
                pickle.dump(obj, gzipf, protocol=pickle.HIGHEST_PROTOCOL)
    store.rename(tmp_file_path, file_path)


def unpickle_from_store(store, file_path, gzip=False):
    """Unpickle obj using the provided checkpoint store."""
    with store.open(file_path, "rb") as f:
        if not gzip:
            obj = pickle.load(f)
        else:
            with gzip_lib.GzipFile(fileobj=f, compresslevel=2) as gzipf:
                obj = pickle.load(gzipf)
    return obj


def _init_random_number_generators(seed=None):
    """Initializes random generators for Python, NumPy, TensorFlow, and JAX."""
    # Seed Python random (None as seed is okay), then use it to seed the others.
    random.seed(seed)
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    logging.info("using seed %d", seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return jax_random.get_prng(seed)


def init_host_and_devices(n_devices=None, random_seed=None):
    """Initializes host and device attributes for this trainers.

    Args:
      n_devices: Number of devices this trainers will use. If ``None``, get the
          number from the backend.
      random_seed: Random seed as the starting point for all random numbers used
          by the trainers. If ``None``, calculate one from system time and host id.

    Returns:
      is_chief: True if this trainers has special chief responsibilities.
      host_count: Number of hosts in this computation.
      n_devices: The passed in value of n_devices or a computed default (for this
        host).
      random_seed: The passed in value of random_seed or a computed default.
    """
    if fastmath.is_backend(fastmath.Backend.JAX):
        host_id = jax.process_index()
        host_count = jax.host_count()
    else:
        host_id = 0
        host_count = 1
    is_chief = host_id == 0

    logging.info(
        "Initializing hosts and devices: host_id %d, host_count %d, " "is_chief %d",
        host_id,
        host_count,
        is_chief,
    )

    device_count = fastmath.local_device_count()
    n_devices = n_devices or device_count
    # TODO(lukaszkaiser): remove this restriction when possible.
    if n_devices != device_count and fastmath.is_backend(fastmath.Backend.JAX):
        raise ValueError(
            "JAX cannot work yet with n_devices != all devices: "
            "%d != %d" % (n_devices, device_count)
        )

    if random_seed is None and host_count > 1:
        random_seed = int(1e6 * (host_id + time.time())) % 2**31
    return (
        is_chief,
        host_count,
        n_devices,
        _init_random_number_generators(random_seed),
    )


def _accelerate_model_with_metrics(
    model_with_metrics, n_devices, accelerate=True, do_mean=True
):
    if not accelerate:
        return model_with_metrics.pure_fn

    return tl.jit_forward(model_with_metrics.pure_fn, n_devices, do_mean=do_mean)


@functools.partial(fastmath.pmap, axis_name="devices", donate_argnums=(0,))
def _make_weights_and_state_same_across_hosts(weights_and_state):
    """Makes train and eval model's weights and state the same across hosts."""

    # We assume that weights_and_state have been already replicated, i.e the
    # leading axis is self._n_devices

    # This is the total number of devices across all hosts.
    n_devices_total = fastmath.psum(jnp.array(1.0), "devices").astype(jnp.int32)

    # We average the weights and state across all devices.
    # We also make sure we don't change the type of the weights and state.
    return fastmath.nested_map(
        lambda x: (fastmath.psum(x, "devices") / n_devices_total).astype(x.dtype),
        weights_and_state,
    )


def _is_empty(x):
    if isinstance(x, (list, tuple)):
        return all(_is_empty(y) for y in x)
    else:
        return x is None


def _is_uninitialized(model):
    """Checks whether no weights in the model have been initialized."""
    if not _is_empty(model.weights):
        return False
    return all(_is_uninitialized(l) for l in model.sublayers)


def _match_by_shape(full, partial):
    """Puts partial into full matching by shape."""
    partial_idx = 0
    res = []
    for w in full:
        if partial_idx >= len(partial):
            res.append(w)  # read everything from parial list, just fill
        elif w is None and partial[partial_idx] is None:  # both Nones, move on
            res.append(None)
            partial_idx += 1
        elif w is None or partial[partial_idx] is None:  # one None but not both
            res.append(w)
        elif w.shape == partial[partial_idx].shape:
            res.append(partial[partial_idx])
            partial_idx += 1
        else:
            res.append(w)
    if partial_idx < len(partial):
        _log("Did not manage to match shapes in model for all checkpoint weights.")
        for w in partial[:partial_idx]:
            _log("  Inserted tensor of shape %s" % str(w.shape))
        for i, w in enumerate(partial[partial_idx:]):
            _log("  Not inserted tensor of shape %s" % str(w.shape))
            model_idx = i + partial_idx
            if model_idx < len(full):
                model_weight_shape = str(full[model_idx].shape)
            else:
                model_weight_shape = "<no corresponding model weight>"
            _log("  Tensor in that place has shape: %s" % model_weight_shape)
    return res


def _flatten_and_remove_empty(x):
    flat = fastmath.tree_flatten(x)
    if isinstance(flat, tuple) and len(flat) == 2 and isinstance(flat[0], (list, tuple)):
        flat = flat[0]
    if not isinstance(flat, (list, tuple)):
        flat = [flat]
    return [
        f for f in flat
        if f is not None and not (isinstance(f, tuple) and len(f) == 0)
    ]


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

