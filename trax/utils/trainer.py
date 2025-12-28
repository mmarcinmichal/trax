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

"""Trax trainers."""
import atexit
import datetime
import functools
import os

import gin
import jax
import tensorflow.compat.v2 as tf

from absl import app, flags, logging
from jax.lib import xla_extension as xc

from trax import fastmath
from trax import layers as tl
from trax import optimizers as trax_opt
from trax.data.preprocessing import inputs as trax_inputs
from trax.learning.supervised import lr_schedules as lr
from trax.learning.supervised import training
from trax.layers import base
from trax.tf import numpy as tf_np

_DEFAULT_METRICS = {
    "loss": tl.WeightedCategoryCrossEntropy(),
    "accuracy": tl.WeightedCategoryAccuracy(),
    "sequence_accuracy": tl.MaskedSequenceAccuracy(),
    "neg_log_perplexity": tl.Serial(
        tl.WeightedCategoryCrossEntropy(), tl.Negate()
    ),
    "weights_per_batch_per_core": tl.Serial(tl.Drop(), tl.Drop(), tl.Sum()),
}

FLAGS = flags.FLAGS
Backend = fastmath.Backend


# TODO(afrozm): Share between trainers.py and rl_trainer.py
def _tf_setup_from_flags():
    """Processes TensorFlow-relevant flags."""
    if FLAGS.enable_eager_execution:
        # In TF2 eager is default; guard to avoid errors if already eager.
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
    if FLAGS.tf_xla:
        tf.config.optimizer.set_jit(True)
        fastmath.tf.set_tf_xla_forced_compile(FLAGS.tf_xla_forced_compile)
    tf.config.optimizer.set_experimental_options(
        {
            "pin_to_host_optimization": FLAGS.tf_opt_pin_to_host,
            "layout_optimizer": FLAGS.tf_opt_layout,
        }
    )
    tf_np.set_allow_float64(FLAGS.tf_allow_float64)


# TODO(afrozm): Share between trainers.py and rl_trainer.py
def _gin_parse_configs():
    """Initializes gin-controlled bindings."""
    # Imports for configurables
    # pylint: disable=g-import-not-at-top,unused-import,g-bad-import-order,reimported,unused-variable

    # pylint: disable=g-import-not-at-top,unused-import,g-bad-import-order,reimported,unused-variable

    configs = FLAGS.config if FLAGS.config is not None else []
    # Override with --dataset and --model
    if FLAGS.dataset:
        configs.append("data_streams.dataset_name='%s'" % FLAGS.dataset)
    if FLAGS.data_dir:
        configs.append("data_streams.data_dir='%s'" % FLAGS.data_dir)
    if FLAGS.model:
        configs.append(
            "trax.supervised.training.train.model=@trax.models.%s"
            % FLAGS.model
        )
    gin.parse_config_files_and_bindings(FLAGS.config_file, configs)


def _output_dir_or_default():
    """Returns a path to the output directory."""
    if FLAGS.output_dir:
        output_dir = FLAGS.output_dir
        logging.info("Using --output_dir %s", output_dir)
        return os.path.expanduser(output_dir)

    # Else, generate a default output dir (under the user's home directory).
    try:
        dataset_name = gin.query_parameter("data_streams.dataset_name")
    except ValueError:
        dataset_name = "random"
    output_name = "{model_name}_{dataset_name}_{timestamp}".format(
        model_name=gin.query_parameter(
            "trax.supervised.training.train.model"
        ).configurable.name,
        dataset_name=dataset_name,
        timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M"),
    )
    output_dir = os.path.join("~", "trax", output_name)
    output_dir = os.path.expanduser(output_dir)
    print()
    logging.info("No --output_dir specified")
    logging.info("Using default output_dir: %s", output_dir)
    return output_dir


# TODO(afrozm): Share between trainers.py and rl_trainer.py
def _jax_and_tf_configure_for_devices():  # pylint: disable=missing-function-docstring
    if FLAGS.use_tpu:
        jax.config.update("jax_platform_name", "tpu")
        jax.config.update("jax_xla_backend", FLAGS.jax_xla_backend)
        jax.config.update("jax_backend_target", FLAGS.jax_backend_target)
    if FLAGS.enable_eager_execution and (
        fastmath.is_backend(Backend.NUMPY) or fastmath.is_backend(Backend.JAX)
    ):
        # Numpy backend doesn't benefit from having the input pipeline run on GPU,
        # and jax backend has GPU memory contention if TF uses the GPU. Gin must be
        # set up first before determining the backend.
        tf.config.experimental.set_visible_devices([], "GPU")


def _train_using_tf(output_dir):
    worker_cpu = tf_init_tpu()
    with tf.device(worker_cpu):
        if num_devices() == 1:
            # TF's device priority is GPU > CPU > TPU, so we need to explicitly make
            # the TPU core the default device here.
            with tf.device("/device:TPU:0"):
                train(output_dir=output_dir)
        else:
            train(output_dir=output_dir)


@gin.configurable
def tf_init_tpu(worker="", protocol=None):
    """Initializes TPU for TensorFlow.

    Args:
      worker: The BNS address of the remote TPU worker. If it's empty (the default
        value), TF will assume the TPU devices are connected to the local host.
      protocol: The network protocol used to connect to the TPU worker.
    Returns:
      The device name of the TPU worker's CPU.
    """
    protocol = protocol or "grpc"
    is_local = worker in ("", "local")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=worker)
    if not is_local:
        tf.config.experimental_connect_to_cluster(resolver, protocol=protocol)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    if is_local:
        return ""
    else:
        return "/job:worker"


def _make_jax_gpu_cluster(host_id, server_ip, n_hosts, server_port=5005):
    """Make JAX GPU Cluster."""

    addr = f"{server_ip}:{server_port}"
    if host_id == 0:
        logging.info("starting service on %s", addr)
        service = xc.get_distributed_runtime_service(addr, n_hosts)
        # We add an explicit call to shut down the service via at exit as Python
        # interpreter may not call the service destructor on process termination.
        atexit.register(service.shutdown)

    logging.info("connecting to service on %s", addr)
    dist_client = xc.get_distributed_runtime_client(addr, host_id)
    dist_client.connect()
    atexit.register(dist_client.shutdown)

    # register dist gpu backend
    factory = functools.partial(
        jax.lib.xla_client.make_gpu_client, dist_client, host_id
    )
    jax.lib.xla_bridge.register_backend_factory("gpu", factory, priority=300)


@gin.configurable(module="trax.supervised.training")
def num_devices(value=None):
    """Returns how many devices to use (if None, default, use all available)."""
    return value


@gin.configurable(module="trax.supervised.training")
def train(
    output_dir,
    model=gin.REQUIRED,
    loss_fn=tl.WeightedCategoryCrossEntropy(),
    inputs=trax_inputs.batcher,
    optimizer=trax_opt.Adafactor,
    lr_schedule_fn=lr.multifactor,
    steps=1000,
    checkpoints_at=None,
    permanent_checkpoints_at=None,
    eval_steps=10,
    eval_frequency=100,
    permanent_checkpoint_frequency=None,
    random_seed=None,
    metrics=None,
    checkpoint_highest=None,
    checkpoint_lowest=None,
    loss_chunk_size=0,
    use_memory_efficient_trainer=False,
    adasum=False,
    init_checkpoint=None,
    callbacks=None,
    n_weights_shards=1,
    additional_train_tasks=None,
    additional_eval_tasks=None,
    additional_eval_streams=None,
):
    base.N_WEIGHTS_SHARDS = n_weights_shards
    if (
        permanent_checkpoint_frequency is not None
        and permanent_checkpoints_at is not None
    ):
        raise ValueError(
            'Only one of ["permanent_checkpoint_frequency", '
            '"permanent_checkpoints_at"] should be set.'
        )

    n_local_devices = num_devices() or fastmath.local_device_count()
    # Prepare the training task.
    # Inputs is either an Inputs instance or a function that returns it.
    if callable(inputs):  # If we pass a function, e.g., through gin, call it.
        inputs = inputs()
    opt = optimizer if use_memory_efficient_trainer else optimizer()
    train_task = training.TrainTask(
        inputs.train_stream(n_local_devices),
        loss_layer=loss_fn,
        optimizer=opt,
        lr_schedule=lr_schedule_fn(),
        n_steps_per_checkpoint=eval_frequency,
        n_steps_per_permanent_checkpoint=permanent_checkpoint_frequency,
    )

    if additional_train_tasks is None:
        additional_train_tasks = []

    # Prepare the evaluation.
    metrics_dict = metrics if metrics is not None else _DEFAULT_METRICS
    names, metrics_layers = zip(*metrics_dict.items())
    eval_task = training.EvalTask(
        inputs.eval_stream(n_local_devices),
        metrics_layers,
        metric_names=names,
        n_eval_batches=eval_steps,
    )

    if additional_eval_tasks is None:
        additional_eval_tasks = []

    additional_eval_tasks_from_streams = []
    if additional_eval_streams is not None:
        for stream in additional_eval_streams:
            additional_eval_tasks_from_streams.append(
                training.EvalTask(
                    stream.stream,
                    metrics_layers,
                    metric_names=names,
                    n_eval_batches=eval_steps,
                    export_prefix=stream.name,
                )
            )

    checkpoint_at = None
    if checkpoints_at is not None:
        checkpoint_at = lambda step: step in checkpoints_at
    permanent_checkpoint_at = None
    if permanent_checkpoints_at is not None:
        permanent_checkpoint_at = lambda step: step in permanent_checkpoints_at

    model_train = model(mode="train")
    model_predict_eval = model(mode="eval")
    if init_checkpoint:
        model_train.init_from_file(init_checkpoint, weights_only=True)
        model_predict_eval.init_from_file(init_checkpoint, weights_only=True)
    loop = training.Loop(
        model_train,
        [train_task] + additional_train_tasks,
        eval_model=model_predict_eval,
        eval_tasks=[eval_task]
        + additional_eval_tasks
        + additional_eval_tasks_from_streams,
        output_dir=output_dir,
        checkpoint_at=checkpoint_at,
        checkpoint_low_metric=checkpoint_lowest,
        checkpoint_high_metric=checkpoint_highest,
        permanent_checkpoint_at=permanent_checkpoint_at,
        n_devices=n_local_devices,
        loss_chunk_size=loss_chunk_size,
        use_memory_efficient_trainer=use_memory_efficient_trainer,
        adasum=adasum,
        random_seed=random_seed,
        callbacks=callbacks,
    )

    steps_to_go = steps - loop.step
    if steps_to_go <= 0:
        logging.info(
            "Stop training, already reached the total training steps %d", steps
        )
        return loop

    loop.run(steps_to_go)
    return loop


def main(_):
    logging.set_verbosity(FLAGS.log_level)

    _tf_setup_from_flags()
    _gin_parse_configs()
    _jax_and_tf_configure_for_devices()

    # Create a JAX GPU cluster if using JAX and given a chief IP.
    if fastmath.is_backend(Backend.JAX) and FLAGS.gpu_cluster_chief_ip:
        _make_jax_gpu_cluster(
            FLAGS.gpu_cluster_host_id,
            FLAGS.gpu_cluster_chief_ip,
            FLAGS.gpu_cluster_n_hosts,
            FLAGS.gpu_cluster_port,
        )

    if FLAGS.disable_jit:
        fastmath.disable_jit()

    output_dir = _output_dir_or_default()
    if FLAGS.use_tpu and fastmath.is_backend(Backend.TFNP):
        _train_using_tf(output_dir)
    else:
        train(output_dir=output_dir)

    logging.info("Finished training.")


if __name__ == "__main__":
    app.run(main)
