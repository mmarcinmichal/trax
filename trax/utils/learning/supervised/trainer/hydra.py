"""Hydra configuration helpers for trainer entrypoints."""

import datetime
import os

from absl import flags
from trax.utils import logging as trax_logging

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
        trax_logging.info("Using --output_dir %s", output_dir)
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
    trax_logging.info("No --output_dir specified")
    trax_logging.info("Using default output_dir: %s", output_dir)
    return output_dir


def train_with_hydra(cfg, output_dir):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    def _flatten_nested_group(root, key):
        node = OmegaConf.select(root, key)
        if node is None or not OmegaConf.is_config(node):
            return
        OmegaConf.set_struct(node, False)
        while OmegaConf.is_config(node) and key in node:
            child = node.get(key)
            if not OmegaConf.is_config(child):
                break
            OmegaConf.set_struct(child, False)
            child_container = OmegaConf.to_container(child, resolve=False)
            if not isinstance(child_container, dict):
                break
            for child_key, child_value in child_container.items():
                if child_key not in node:
                    node[child_key] = child_value
            node = child

    def _hoist_nested_data(root):
        data_node = OmegaConf.select(root, "data")
        if data_node is None or not OmegaConf.is_config(data_node):
            return
        OmegaConf.set_struct(data_node, False)
        while True:
            nested = data_node.get("data")
            if not OmegaConf.is_config(nested):
                break
            OmegaConf.set_struct(nested, False)
            nested_container = OmegaConf.to_container(nested, resolve=False)
            if not isinstance(nested_container, dict):
                break
            for key, value in nested_container.items():
                if key not in data_node:
                    data_node[key] = value
            data_node = nested

    def _audit_and_warn(root):
        for key in ("data.data", "model.model", "optim.optim", "schedule.schedule", "train.train", "ckpt.ckpt"):
            if OmegaConf.select(root, key) is not None:
                trax_logging.warning("Nested config detected at %s; flattening applied.", key)

        overrides = FLAGS.hydra_overrides if FLAGS.hydra_overrides is not None else []
        for override in overrides:
            if not isinstance(override, str) or "=" not in override:
                continue
            raw_key = override.split("=", 1)[0].lstrip("+~")
            if raw_key.startswith("hydra."):
                continue
            if OmegaConf.select(root, raw_key) is None:
                trax_logging.warning(
                    "Override key '%s' not found in composed config; check path.",
                    raw_key,
                )

    def _promote_optimizer_targets(root):
        for name in ("Adam", "Adafactor"):
            target_node = OmegaConf.select(root, f"optim.optim.optim.{name}")
            if target_node is None or "_target_" not in target_node:
                continue
            existing = OmegaConf.select(root, f"optim.{name}")
            if existing is None:
                OmegaConf.update(root, f"optim.{name}", target_node, merge=False)
                continue
            if OmegaConf.is_config(existing) and "_target_" not in existing:
                OmegaConf.update(
                    root, f"optim.{name}._target_", target_node["_target_"], merge=False
                )

    def _select_experiment_root(node):
        experiment_node = OmegaConf.select(node, "experiment")
        if experiment_node is None:
            return node
        if OmegaConf.is_config(experiment_node):
            keys = list(experiment_node.keys())
            if len(keys) == 1:
                inner = experiment_node.get(keys[0])
                if inner is not None:
                    return inner
        return experiment_node

    cfg = _select_experiment_root(cfg)
    for key in ("optim", "model", "schedule", "train", "ckpt", "data"):
        _flatten_nested_group(cfg, key)
    _hoist_nested_data(cfg)
    _promote_optimizer_targets(cfg)
    _audit_and_warn(cfg)

    def _inst(path):
        node = OmegaConf.select(cfg, path)
        return None if node is None else instantiate(node)

    def _resolve_interpolation(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            ref = value[2:-1]
            return OmegaConf.select(cfg, ref)
        return value

    train_node = OmegaConf.select(cfg, "train")
    train_args_node = None
    if train_node is not None:
        if isinstance(train_node, dict) and "train" in train_node:
            train_args_node = train_node["train"]
        else:
            train_args_node = train_node

    def _flatten_train_node(node):
        if node is None:
            return None
        while OmegaConf.is_config(node) and "train" in node:
            child = node.get("train")
            if not OmegaConf.is_config(child):
                break
            child_container = OmegaConf.to_container(child, resolve=False)
            parent_container = OmegaConf.to_container(node, resolve=False)
            if not isinstance(child_container, dict) or not isinstance(parent_container, dict):
                break
            parent_container.pop("train", None)
            merged = {**parent_container, **child_container}
            node = OmegaConf.create(merged)
            if "train" not in node:
                break
        return node

    if train_args_node is not None:
        train_args_node = _flatten_train_node(train_args_node)
        train_cfg = OmegaConf.to_container(train_args_node, resolve=False)
    else:
        train_cfg = {}

    if not isinstance(train_cfg, dict):
        train_cfg = {}

    train_cfg.pop("ckpt", None)

    if isinstance(train_cfg.get("inputs"), dict):
        from trax.data.preprocessing.inputs import StreamBundle

        inputs_dict = train_cfg.get("inputs", {})
        if "train_stream" in inputs_dict:
            train_cfg["inputs"] = StreamBundle(**inputs_dict)

    inputs = train_cfg.get("inputs")
    if inputs is None:
        make_streams_node = OmegaConf.select(cfg, "data.make_streams")
        if make_streams_node is not None:
            try:
                from trax import data as trax_data

                stream_lists = {}
                for stream_key in ("train_stream", "eval_stream", "train_eval_stream"):
                    stream_node = OmegaConf.select(
                        cfg, f"data.make_streams.{stream_key}"
                    )
                    if stream_node is None:
                        continue
                    if OmegaConf.is_config(stream_node):
                        stream_node = OmegaConf.to_container(stream_node, resolve=False)
                    if stream_node is None:
                        continue
                    stream_list = []
                    for item in stream_node:
                        if (
                            isinstance(item, str)
                            and item.startswith("${")
                            and item.endswith("}")
                        ):
                            ref = item[2:-1]
                            resolved = OmegaConf.select(cfg, ref)
                            if resolved is not None:
                                item = resolved
                        if OmegaConf.is_config(item):
                            item = OmegaConf.to_container(item, resolve=False)
                        if isinstance(item, dict) and "_target_" in item:
                            item = instantiate(item)
                        stream_list.append(item)
                    stream_lists[stream_key] = stream_list
                if "train_stream" in stream_lists:
                    inputs = trax_data.make_streams(
                        train_stream=stream_lists.get("train_stream"),
                        eval_stream=stream_lists.get("eval_stream"),
                        train_eval_stream=stream_lists.get("train_eval_stream"),
                    )
            except Exception:
                inputs = None
    if inputs is None:
        inputs = _inst("data.make_streams")
    def _find_target_node(node):
        if isinstance(node, dict):
            if "_target_" in node:
                return node
            for value in node.values():
                found = _find_target_node(value)
                if found is not None:
                    return found
        return None

    model = train_cfg.get("model") or _inst("model.model_fn")
    model = _resolve_interpolation(model)
    if OmegaConf.is_config(model):
        model = OmegaConf.to_container(model, resolve=False)
    if isinstance(model, dict):
        model = instantiate(model)
    if model is None:
        for candidate in (
            "model.MLP",
            "model.RNNLM",
            "model.BERT",
            "model.model.MLP",
            "model.model.RNNLM",
            "model.model.BERT",
        ):
            model = _inst(candidate)
            if model is not None:
                break
    if model is None:
        model_node = OmegaConf.select(cfg, "model")
        if model_node is not None:
            model_container = (
                OmegaConf.to_container(model_node, resolve=True)
                if OmegaConf.is_config(model_node)
                else model_node
            )
            target_node = _find_target_node(model_container)
            if target_node is not None:
                model = instantiate(target_node)
    optimizer = train_cfg.get("optimizer") or _inst("optim.optimizer")
    optimizer = _resolve_interpolation(optimizer)
    if OmegaConf.is_config(optimizer):
        optimizer = OmegaConf.to_container(optimizer, resolve=False)
    if isinstance(optimizer, dict):
        optimizer = instantiate(optimizer)
    lr_schedule_fn = train_cfg.get("lr_schedule_fn")
    lr_schedule_fn = _resolve_interpolation(lr_schedule_fn)
    if OmegaConf.is_config(lr_schedule_fn):
        lr_schedule_fn = OmegaConf.to_container(lr_schedule_fn, resolve=False)
    if isinstance(lr_schedule_fn, dict):
        lr_schedule_fn = instantiate(lr_schedule_fn)

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
    train_cfg.pop("ckpt", None)

    from trax.learning.training import trainer as supervised_trainer

    supervised_trainer.train(output_dir=output_dir, **train_cfg)