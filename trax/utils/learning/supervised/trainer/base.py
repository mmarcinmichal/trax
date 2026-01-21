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

"""Trax trainer entrypoint and shared runtime setup."""

import atexit
import functools

import gin
import jax
import tensorflow.compat.v2 as tf

from absl import app, flags, logging
from jax.lib import xla_extension as xc

from trax import fastmath
from trax.learning.training import trainer as supervised_trainer
from trax.tf import numpy as tf_np
from trax.utils.learning.supervised.trainer import gini as gini_utils
from trax.utils.learning.supervised.trainer import hydra as hydra_utils
from trax.utils.learning.training import trainer_flags  # noqa: F401

FLAGS = flags.FLAGS
Backend = fastmath.Backend


def _tf_setup_from_flags():
    """Processes TensorFlow-relevant flags."""
    if FLAGS.enable_eager_execution:
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


def _jax_and_tf_configure_for_devices():  # pylint: disable=missing-function-docstring
    if FLAGS.use_tpu:
        jax.config.update("jax_platform_name", "tpu")
        jax.config.update("jax_xla_backend", FLAGS.jax_xla_backend)
        jax.config.update("jax_backend_target", FLAGS.jax_backend_target)
    if FLAGS.enable_eager_execution and (
        fastmath.is_backend(Backend.NUMPY) or fastmath.is_backend(Backend.JAX)
    ):
        tf.config.experimental.set_visible_devices([], "GPU")


def _train_using_tf(output_dir):
    worker_cpu = tf_init_tpu()
    with tf.device(worker_cpu):
        if num_devices() == 1:
            with tf.device("/device:TPU:0"):
                train(output_dir=output_dir)
        else:
            train(output_dir=output_dir)


@gin.configurable
def tf_init_tpu(worker="", protocol=None):
    """Initializes TPU for TensorFlow."""
    protocol = protocol or "grpc"
    is_local = worker in ("", "local")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=worker)
    if not is_local:
        tf.config.experimental_connect_to_cluster(resolver, protocol=protocol)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    if is_local:
        return ""
    return "/job:worker"


def _make_jax_gpu_cluster(host_id, server_ip, n_hosts, server_port=5005):
    """Make JAX GPU Cluster."""
    addr = f"{server_ip}:{server_port}"
    if host_id == 0:
        logging.info("starting service on %s", addr)
        service = xc.get_distributed_runtime_service(addr, n_hosts)
        atexit.register(service.shutdown)

    logging.info("connecting to service on %s", addr)
    dist_client = xc.get_distributed_runtime_client(addr, host_id)
    dist_client.connect()
    atexit.register(dist_client.shutdown)

    factory = functools.partial(
        jax.lib.xla_client.make_gpu_client, dist_client, host_id
    )
    jax.lib.xla_bridge.register_backend_factory("gpu", factory, priority=300)


# Re-export loop-based training helpers to avoid duplicate gin registries.
num_devices = supervised_trainer.num_devices
train = supervised_trainer.train


def main(_):
    logging.set_verbosity(FLAGS.log_level)

    _tf_setup_from_flags()
    if FLAGS.use_hydra:
        cfg = hydra_utils.compose_config()
    else:
        gini_utils.parse_configs()
    _jax_and_tf_configure_for_devices()

    if fastmath.is_backend(Backend.JAX) and FLAGS.gpu_cluster_chief_ip:
        _make_jax_gpu_cluster(
            FLAGS.gpu_cluster_host_id,
            FLAGS.gpu_cluster_chief_ip,
            FLAGS.gpu_cluster_n_hosts,
            FLAGS.gpu_cluster_port,
        )

    if FLAGS.disable_jit:
        fastmath.disable_jit()

    if FLAGS.use_hydra:
        output_dir = hydra_utils.output_dir_or_default(cfg)
        if FLAGS.use_tpu and fastmath.is_backend(Backend.TFNP):
            _train_using_tf(output_dir)
        else:
            hydra_utils.train_with_hydra(cfg, output_dir)
    else:
        output_dir = gini_utils.output_dir_or_default()
        if FLAGS.use_tpu and fastmath.is_backend(Backend.TFNP):
            _train_using_tf(output_dir)
        else:
            train(output_dir=output_dir)

    logging.info("Finished training.")


if __name__ == "__main__":
    app.run(main)
