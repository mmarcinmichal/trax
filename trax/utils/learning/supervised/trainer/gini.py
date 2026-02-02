"""Gin configuration helpers for trainer entrypoints."""

import datetime
import os

import gin

from absl import flags
from trax.utils import logging as trax_logging

from trax.utils.learning.training import trainer_flags  # noqa: F401

FLAGS = flags.FLAGS


def parse_configs():
    """Initializes gin-controlled bindings."""
    configs = FLAGS.config if FLAGS.config is not None else []
    if FLAGS.dataset:
        configs.append("data_streams.dataset_name='%s'" % FLAGS.dataset)
    if FLAGS.data_dir:
        configs.append("data_streams.data_dir='%s'" % FLAGS.data_dir)
    if FLAGS.model:
        configs.append(
            "trax.learning.trainer.train.model=@trax.models.%s"
            % FLAGS.model
        )
    gin.parse_config_files_and_bindings(FLAGS.config_file, configs)


def output_dir_or_default():
    """Returns a path to the output directory."""
    if FLAGS.output_dir:
        output_dir = FLAGS.output_dir
        trax_logging.info("Using --output_dir %s", output_dir)
        return os.path.expanduser(output_dir)

    try:
        dataset_name = gin.query_parameter("data_streams.dataset_name")
    except ValueError:
        dataset_name = "random"
    output_name = "{model_name}_{dataset_name}_{timestamp}".format(
        model_name=gin.query_parameter(
            "trax.learning.trainer.train.model"
        ).configurable.name,
        dataset_name=dataset_name,
        timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M"),
    )
    output_dir = os.path.join("~", "trax", output_name)
    output_dir = os.path.expanduser(output_dir)
    print()
    trax_logging.info("No --output_dir specified")
    trax_logging.info("Using default output_dir: %s", output_dir)
    return output_dir