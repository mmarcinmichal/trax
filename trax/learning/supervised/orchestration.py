# coding=utf-8
"""Orchestration utilities for supervised training loops.

This module defines small single-responsibility helpers that encapsulate the
training setup and runtime behaviors used by :mod:`trax.learning.supervised`.
The goal is to make the main training loop composition-friendly and keep the
core step execution minimal.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Iterable, List, Optional, Protocol

from trax import fastmath
from trax import layers as tl


class Callback(Protocol):
    """Step callback protocol used by :class:`CallbackPipeline`."""

    def call_at(self, step: int) -> bool:
        ...

    def on_step_begin(self, step: int) -> None:  # pragma: no cover - interface
        ...

    def on_step_end(self, step: int) -> None:  # pragma: no cover - interface
        ...


@dataclasses.dataclass(frozen=True)
class DeviceManager:
    """Manages device specific helpers used by training loops."""

    is_chief: bool
    n_hosts: int
    n_devices: int

    def for_n_devices(self, value):
        return tl.for_n_devices(value, self.n_devices)

    def unreplicate(self, value):
        if self.n_devices == 1:
            return value
        return fastmath.nested_map(lambda x: x[0], value)

    def reshape_by_device(self, value):
        if self.n_devices == 1:
            return value
        return tl.reshape_by_device(value, self.n_devices)


class SeedManager:
    """Tracks rng state and produces per-step keys."""

    def __init__(self, initial_rng, use_memory_efficient_trainer: bool = False):
        self._rng = initial_rng
        self._use_memory_efficient_trainer = use_memory_efficient_trainer

    def new_rng(self):
        self._rng, rng = fastmath.random.split(self._rng)
        if self._use_memory_efficient_trainer:
            self._rng = tl.on_cpu(self._rng)
            rng = tl.on_cpu(rng)
        return rng


@dataclasses.dataclass
class HostAndDeviceInitializer:
    """Initializes host/device info and constructs the device manager."""

    init_fn: Callable

    def initialize(self, n_devices=None, random_seed=None):
        is_chief, n_hosts, n_devices, initial_rng = self.init_fn(
            n_devices, random_seed
        )
        device_manager = DeviceManager(
            is_chief=is_chief, n_hosts=n_hosts, n_devices=n_devices
        )
        return is_chief, n_hosts, device_manager, initial_rng


@dataclasses.dataclass
class SeedManagerFactory:
    """Creates :class:`SeedManager` instances."""

    use_memory_efficient_trainer: bool = False

    def create(self, initial_rng):
        return SeedManager(initial_rng, self.use_memory_efficient_trainer)


class ModelInitializer:
    """Initializes models and optionally syncs their weights/state."""

    def __init__(
        self,
        batch_signature,
        use_memory_efficient_trainer: bool = False,
        is_uninitialized: Optional[Callable] = None,
    ):
        self._batch_signature = batch_signature
        self._use_memory_efficient_trainer = use_memory_efficient_trainer
        self._is_uninitialized = is_uninitialized

    def initialize(self, model, eval_model, sync_fn: Optional[Callable] = None):
        if not self._use_memory_efficient_trainer:
            if self._is_uninitialized is not None and self._is_uninitialized(model):
                model.init(self._batch_signature)
            if self._is_uninitialized is not None and self._is_uninitialized(eval_model):
                eval_model.init(self._batch_signature)
        if sync_fn is not None:
            sync_fn()


class CheckpointManager:
    """Handles checkpoint policy decisions."""

    def __init__(self, should_checkpoint: Callable[[int], bool]):
        self._should_checkpoint = should_checkpoint

    def should_save(self, step: int) -> bool:
        return self._should_checkpoint(step)


class EvaluatorFactory:
    """Builds evaluator instances lazily."""

    def __init__(self, init_fn: Callable):
        self._init_fn = init_fn

    def create(self, eval_tasks: Iterable):
        return tuple(self._init_fn(task) for task in eval_tasks)


class CallbackPipeline:
    """Runs callback hooks for a step."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self._callbacks = callbacks or []

    def on_step_begin(self, step: int):
        for callback in self._callbacks:
            if callback.call_at(step):
                callback.on_step_begin(step)

    def on_step_end(self, step: int):
        for callback in self._callbacks:
            if callback.call_at(step):
                callback.on_step_end(step)


class CallbackAssembler:
    """Builds :class:`CallbackPipeline` instances."""

    def assemble(self, callbacks: Optional[List[Callback]] = None):
        return CallbackPipeline(callbacks or [])


@dataclasses.dataclass
class TrainingOrchestrator:
    """Coordinates the per-step training execution."""

    device_manager: DeviceManager
    seed_manager: SeedManager
    callback_pipeline: CallbackPipeline

    def run_step(self, trainer, task, step: int, task_changed: bool, sync_fn=None):
        if task_changed and sync_fn is not None:
            sync_fn()
        self.callback_pipeline.on_step_begin(step)
        learning_rate = task.learning_rate(step)
        batch = self.device_manager.reshape_by_device(task.next_batch())
        rng = self.seed_manager.new_rng()
        loss, stats = trainer.one_step(batch, rng, step=step, learning_rate=learning_rate)
        self.callback_pipeline.on_step_end(step)
        return loss, stats
