"""Interfaces and registries for checkpoints and metrics sinks."""
import abc
import contextlib
import os

from typing import Callable, Dict, Optional

import tensorflow as tf

from absl import logging

from trax.utils.board import jaxboard


class CheckpointStore(abc.ABC):
    """Abstract interface for persisting checkpoint artifacts."""

    def __init__(self, base_dir: Optional[str] = None):
        self._base_dir = base_dir

    @property
    def base_dir(self) -> Optional[str]:
        return self._base_dir

    def resolve(self, path: str) -> str:
        if self._base_dir and not os.path.isabs(path):
            return os.path.join(self._base_dir, path)
        return path

    @abc.abstractmethod
    def makedirs(self, path: str):
        raise NotImplementedError

    @abc.abstractmethod
    def exists(self, path: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def open(self, path: str, mode: str):
        raise NotImplementedError

    @abc.abstractmethod
    def rename(self, src: str, dst: str):
        raise NotImplementedError


class LocalCheckpointStore(CheckpointStore):
    """Checkpoint store backed by the local filesystem."""

    def makedirs(self, path: str):
        tf.io.gfile.makedirs(self.resolve(path))

    def exists(self, path: str) -> bool:
        return tf.io.gfile.exists(self.resolve(path))

    def open(self, path: str, mode: str):
        return tf.io.gfile.GFile(self.resolve(path), mode)

    def rename(self, src: str, dst: str):
        tf.io.gfile.rename(self.resolve(src), self.resolve(dst), overwrite=True)


class CloudCheckpointStore(LocalCheckpointStore):
    """Checkpoint store using tf.io.gfile for cloud paths (GCS/S3)."""

    # tf.io.gfile transparently supports cloud schemes, so we reuse the local impl.
    pass


class MetricsSink(abc.ABC):
    """Interface for emitting metrics to various backends."""

    def __init__(self, base_dir: Optional[str] = None, prefix: str = ""):
        self._base_dir = base_dir
        self._prefix = prefix

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def base_dir(self) -> Optional[str]:
        return self._base_dir

    def with_subpath(self, relative: str, prefix: Optional[str] = None):
        return self.__class__(
            base_dir=os.path.join(self._base_dir, relative) if self._base_dir else None,
            prefix=self._prefix + (prefix or ""),
        )

    @abc.abstractmethod
    def log_scalar(self, name: str, value, step: int):
        raise NotImplementedError

    @abc.abstractmethod
    def log_image(self, name: str, value, step: int):
        raise NotImplementedError

    def log_text(self, name: str, text: str, step: int):
        del name, text, step

    def flush(self):
        """Persist buffered metrics if applicable."""

    def close(self):
        """Release resources held by the sink."""


class NullMetricsSink(MetricsSink):
    def log_scalar(self, name: str, value, step: int):
        del name, value, step

    def log_image(self, name: str, value, step: int):
        del name, value, step


class JaxboardMetricsSink(MetricsSink):
    def __init__(self, base_dir: Optional[str], prefix: str = ""):
        super().__init__(base_dir, prefix)
        self._writer = jaxboard.SummaryWriter(base_dir) if base_dir else None

    def log_scalar(self, name: str, value, step: int):
        if self._writer:
            self._writer.scalar(self.prefix + name, value, step)

    def log_image(self, name: str, value, step: int):
        if self._writer:
            self._writer.image(self.prefix + name, value, step)

    def flush(self):
        if self._writer:
            self._writer.flush()

    def close(self):
        if self._writer:
            self._writer.close()

    def log_text(self, name: str, text: str, step: int):
        if self._writer:
            self._writer.text(self.prefix + name, text, step)

    def with_subpath(self, relative: str, prefix: Optional[str] = None):
        subdir = os.path.join(self._base_dir, relative) if self._base_dir else None
        return JaxboardMetricsSink(subdir, self._prefix + (prefix or ""))


class MlflowMetricsSink(MetricsSink):
    def __init__(self, base_dir: Optional[str] = None, prefix: str = ""):
        super().__init__(base_dir, prefix)
        try:
            import mlflow  # pylint: disable=import-error

            self._mlflow = mlflow
        except Exception:  # pylint: disable=broad-except
            self._mlflow = None
            logging.warning("MLflow not available; metrics will be logged locally only.")

    def log_scalar(self, name: str, value, step: int):
        if self._mlflow:
            self._mlflow.log_metric(self.prefix + name, value, step=step)
        else:
            logging.info("[mlflow stub] %s%d=%s", self.prefix + name, step, value)

    def log_image(self, name: str, value, step: int):
        del name, value, step


class WandbMetricsSink(MetricsSink):
    def __init__(self, base_dir: Optional[str] = None, prefix: str = ""):
        super().__init__(base_dir, prefix)
        try:
            import wandb  # pylint: disable=import-error

            self._wandb = wandb
        except Exception:  # pylint: disable=broad-except
            self._wandb = None
            logging.warning("Weights & Biases not available; metrics will be logged locally only.")

    def log_scalar(self, name: str, value, step: int):
        if self._wandb:
            self._wandb.log({self.prefix + name: value, "step": step})
        else:
            logging.info("[wandb stub] %s%d=%s", self.prefix + name, step, value)

    def log_image(self, name: str, value, step: int):
        del name, value, step


class MetricsSinkRegistry:
    def __init__(self):
        self._registry: Dict[str, Callable[..., MetricsSink]] = {}

    def register(self, name: str, factory: Callable[..., MetricsSink]):
        self._registry[name] = factory

    def get(self, name: str, **kwargs) -> MetricsSink:
        return self._registry[name](**kwargs)


class CheckpointStoreRegistry:
    def __init__(self):
        self._registry: Dict[str, Callable[..., CheckpointStore]] = {}

    def register(self, name: str, factory: Callable[..., CheckpointStore]):
        self._registry[name] = factory

    def get(self, name: str, **kwargs) -> CheckpointStore:
        return self._registry[name](**kwargs)


CHECKPOINT_STORE_REGISTRY = CheckpointStoreRegistry()
CHECKPOINT_STORE_REGISTRY.register("local", LocalCheckpointStore)
CHECKPOINT_STORE_REGISTRY.register("gcs", CloudCheckpointStore)
CHECKPOINT_STORE_REGISTRY.register("s3", CloudCheckpointStore)

METRICS_SINK_REGISTRY = MetricsSinkRegistry()
METRICS_SINK_REGISTRY.register("jaxboard", JaxboardMetricsSink)
METRICS_SINK_REGISTRY.register("tensorboard", JaxboardMetricsSink)
METRICS_SINK_REGISTRY.register("mlflow", MlflowMetricsSink)
METRICS_SINK_REGISTRY.register("wandb", WandbMetricsSink)


@contextlib.contextmanager
def noop_context():
    yield
