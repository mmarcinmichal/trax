# coding=utf-8
"""Training task definitions for Trax learning loops."""

import gin

from trax import fastmath


@gin.configurable(module="trax.learning.trainer")
class TrainingTask:
    """A supervised task (labeled data + feedback mechanism) for training."""

    def __init__(
        self,
        labeled_data,
        loss_layer,
        optimizer,
        lr_schedule=None,
        n_steps_per_checkpoint=100,
        n_steps_per_permanent_checkpoint=None,
        loss_name=None,
        sample_batch=None,
        export_prefix=None,
    ):
        r"""Configures a training task.

        Args:
          labeled_data: Iterator of batches of labeled data tuples. Each tuple has
              1+ data (input value) tensors followed by 1 label (target value)
              tensor. All tensors are NumPy ndarrays or their JAX counterparts.
          loss_layer: Layer that computes a scalar value (the "loss") by comparing
              model output :math:`\hat{y}=f(x)` to the target :math:`y`.
          optimizer: Optimizer object that computes model weight updates from
              loss-function gradients.
          lr_schedule: Learning rate schedule, a function step -> learning_rate.
          n_steps_per_checkpoint: How many steps to run between checkpoints.
          n_steps_per_permanent_checkpoint: How many steps to run between permanent
              checkpoints.
          loss_name: Name for the loss metric.
          sample_batch: Optional sample batch for model initialization. If not
              provided, it will be taken from ``labeled_data``.
          export_prefix: Optional task name to be used as prefix for exporting
              metrics during training in Loop.
        """
        self._export_prefix = export_prefix
        self._labeled_data = labeled_data
        self._loss_layer = loss_layer
        self._optimizer = optimizer
        self._lr_schedule = lr_schedule
        self._sample_batch = sample_batch or next(labeled_data)
        self._n_steps_per_checkpoint = n_steps_per_checkpoint
        self._n_steps_per_permanent_checkpoint = n_steps_per_permanent_checkpoint
        self._loss_name = loss_name or self._loss_layer.name

    @property
    def labeled_data(self):
        return self._labeled_data

    @property
    def sample_batch(self):
        return self._sample_batch

    def next_batch(self):
        """Returns one batch of labeled data: a tuple of input(s) plus label."""
        return next(self._labeled_data)

    @property
    def export_prefix(self):
        return self._export_prefix

    @property
    def loss_layer(self):
        return self._loss_layer

    @property
    def loss_name(self):
        return self._loss_name

    @property
    def n_steps_per_checkpoint(self):
        return self._n_steps_per_checkpoint

    @property
    def n_steps_per_permanent_checkpoint(self):
        return self._n_steps_per_permanent_checkpoint

    @property
    def optimizer(self):
        return self._optimizer

    def learning_rate(self, step):
        """Return the learning rate for the given step."""
        if self._lr_schedule is not None:
            with fastmath.use_backend(fastmath.Backend.NUMPY):
                return self._lr_schedule(step)
        opt = self._optimizer
        if callable(opt):  # when optimizer is a function, like Adam, not Adam()
            opt = opt()
        params = opt._init_opt_params  # pylint: disable=protected-access
        return params["learning_rate"]


@gin.configurable(module="trax.learning.trainer")
class EvaluationTask:
    """Labeled data plus scalar functions for (periodically) measuring a model.

    An eval task specifies how (``labeled_data`` + ``metrics``) and with what
    precision (``n_eval_batches``) to measure a model as it is training.
    The variance of each scalar output is reduced by measuring over multiple
    (``n_eval_batches``) batches and reporting the average from those
    measurements.
    """

    def __init__(
        self,
        labeled_data,
        metrics,
        metric_names=None,
        n_eval_batches=1,
        sample_batch=None,
        export_prefix=None,
    ):
        r"""Configures an eval task: named metrics run with a given data source.

        Args:
          labeled_data: Iterator of batches of labeled data tuples. Each tuple has
              1+ data tensors (NumPy ndarrays) followed by 1 label (target value)
              tensor.
          metrics: List of layers; each computes a scalar value per batch by
              comparing model output :math:`\hat{y}=f(x)` to the target :math:`y`.
          metric_names: List of names, one for each item in ``metrics``, in matching
              order, to be used when recording/reporting eval output. If ``None``,
              generate default names using layer names from metrics.
          n_eval_batches: Integer N that specifies how many eval batches to run;
              the output is then the average of the outputs from the N batches.
          sample_batch: Optional sample batch for model initialization. If not
              provided, it will be taken from ``labeled_data``.
          export_prefix: Optional task name to be used as prefix for exporting
              metrics during evaluation in Loop.
        """
        self._export_prefix = export_prefix
        self._labeled_data = labeled_data
        self._metrics = metrics
        self._metric_names = metric_names or self._default_names()
        self._n_eval_batches = n_eval_batches  # pylint: disable=invalid-name

        self._sample_batch = sample_batch or next(labeled_data)
        self._check_init_values()

    @property
    def labeled_data(self):
        return self._labeled_data

    @property
    def sample_batch(self):
        return self._sample_batch

    def next_batch(self):
        """Returns one batch of labeled data: a tuple of input(s) plus label."""
        return next(self._labeled_data)

    @property
    def export_prefix(self):
        return self._export_prefix

    @property
    def metrics(self):
        return self._metrics

    @property
    def metric_names(self):
        return self._metric_names

    @property
    def n_eval_batches(self):
        return self._n_eval_batches

    def _default_names(self):
        return [m.name for m in self._metrics]

    def _check_init_values(self):
        if len(self._metrics) != len(self._metric_names):
            raise ValueError(
                f"Number of metrics ({len(self._metrics)}) does not equal "
                f"number of metric names ({len(self._metric_names)})."
            )


__all__ = ["TrainingTask", "EvaluationTask"]
