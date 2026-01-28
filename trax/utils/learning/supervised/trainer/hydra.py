"""Hydra configuration helpers for trainer entrypoints."""

import datetime
import os

from absl import flags, logging

from trax.utils.learning.training import trainer_flags  # noqa: F401

FLAGS = flags.FLAGS


def config_dir():
    if FLAGS.hydra_config_dir:
        return os.path.abspath(os.path.expanduser(FLAGS.hydra_config_dir))
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../../resources/learning/supervised/configs/yaml",
        )
    )


def compose_config():
    try:
        from hydra import compose, initialize_config_dir
    except ImportError as exc:
        raise ImportError(
            "Hydra is required for --use_hydra. Install hydra-core."
        ) from exc

    overrides = FLAGS.hydra_overrides if FLAGS.hydra_overrides is not None else []
    with initialize_config_dir(version_base=None, config_dir=config_dir()):
        return compose(config_name=FLAGS.hydra_config_name, overrides=overrides)


def output_dir_or_default(cfg):
    if FLAGS.output_dir:
        output_dir = FLAGS.output_dir
        logging.info("Using --output_dir %s", output_dir)
        return os.path.expanduser(output_dir)

    from omegaconf import OmegaConf

    dataset_name = (
        OmegaConf.select(cfg, "data.data_streams.dataset_name")
        or OmegaConf.select(cfg, "data.dataset_loader.dataset_name")
        or OmegaConf.select(cfg, "dataset.dataset_name")
        or "random"
    )
    model_target = OmegaConf.select(cfg, "train.model._target_")
    if not model_target:
        model_target = OmegaConf.select(cfg, "model.model_fn._target_")
    if not model_target:
        model_target = OmegaConf.select(cfg, "train.model")
    model_name = (
        str(model_target).split(".")[-1] if model_target is not None else "model"
    )
    output_name = "{model_name}_{dataset_name}_{timestamp}".format(
        model_name=model_name,
        dataset_name=dataset_name,
        timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M"),
    )
    output_dir = os.path.join("~", "trax", output_name)
    output_dir = os.path.expanduser(output_dir)
    print()
    logging.info("No --output_dir specified")
    logging.info("Using default output_dir: %s", output_dir)
    return output_dir


def train_with_hydra(cfg, output_dir):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import warnings

    def _inst(path):
        node = OmegaConf.select(cfg, path)
        return None if node is None else instantiate(node)

    train_node = OmegaConf.select(cfg, "train")
    train_args_node = None
    if train_node is not None:
        if isinstance(train_node, dict) and "train" in train_node:
            train_args_node = train_node["train"]
        else:
            train_args_node = train_node

    if train_args_node is not None:
        train_args = instantiate(train_args_node)
        train_cfg = (
            OmegaConf.to_container(train_args, resolve=True)
            if isinstance(train_args, dict) or OmegaConf.is_config(train_args)
            else {}
        )
    else:
        train_cfg = {}

    if not isinstance(train_cfg, dict):
        train_cfg = {}

    inputs = train_cfg.get("inputs") or _inst("data.make_inputs")
    model = train_cfg.get("model") or _inst("model.model_fn")
    optimizer = train_cfg.get("optimizer") or _inst("optim.optimizer")
    lr_schedule_fn = train_cfg.get("lr_schedule_fn")

    if lr_schedule_fn is None:
        lr_schedule_fn = _inst("schedule.lr_schedule_fn")
    if lr_schedule_fn is None:
        schedule_node = OmegaConf.select(cfg, "schedule")
        if isinstance(schedule_node, dict):
            if "multifactor" in schedule_node:
                lr_schedule_fn = instantiate(schedule_node["multifactor"])
            elif len(schedule_node) == 1:
                lr_schedule_fn = instantiate(next(iter(schedule_node.values())))

    for key in ("inputs", "model", "optimizer", "lr_schedule_fn"):
        train_cfg.pop(key, None)

    if inputs is not None:
        train_cfg["inputs"] = inputs
    if model is not None:
        train_cfg["model"] = model
    if optimizer is not None:
        train_cfg["optimizer"] = optimizer
    if lr_schedule_fn is not None:
        train_cfg["lr_schedule_fn"] = lr_schedule_fn

    for key, path, fallback in (
        ("additional_eval_tasks", "train.train.additional_eval_tasks", "train.additional_eval_tasks"),
        ("additional_train_tasks", "train.train.additional_train_tasks", "train.additional_train_tasks"),
    ):
        if key not in train_cfg:
            value = _inst(path)
            if value is None:
                value = _inst(fallback)
            if value is not None:
                train_cfg[key] = value

    ckpt_node = OmegaConf.select(cfg, "ckpt")
    ckpt_cfg = (
        OmegaConf.to_container(ckpt_node, resolve=True) if ckpt_node is not None else {}
    )
    if isinstance(ckpt_cfg, dict) and "ckpt" in ckpt_cfg:
        ckpt_cfg = ckpt_cfg.get("ckpt", {})
    if isinstance(ckpt_cfg, dict):
        for key, value in ckpt_cfg.items():
            if value is not None and value != {}:
                train_cfg.setdefault(key, value)

    from trax.learning.training import trainer as supervised_trainer

    supervised_trainer.train(output_dir=output_dir, **train_cfg)
