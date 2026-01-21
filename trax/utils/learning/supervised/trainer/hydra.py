# entrypoint.py
import os
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from trax.utils.learning.supervised import trainer as trax_trainer

def main():
    # 1) Kompozycja hydra config (jak w _compose_hydra_config)
    config_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "resources/learning/supervised/configs/yaml",
        )
    )
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=transformer_lm1b_8gb_testing",
                # dodatkowe overrides...
            ],
        )

    # 2) Wyciągnięcie sekcji "train" i resolvowanie wartości
    train_node = OmegaConf.select(cfg, "train")
    train_cfg = (
        OmegaConf.to_container(train_node, resolve=True)
        if train_node is not None
        else {}
    )
    if not isinstance(train_cfg, dict):
        train_cfg = {}

    # 3) instantiate kluczowych elementów (jak w _train_with_hydra)
    def _inst(path):
        node = OmegaConf.select(cfg, path)
        return None if node is None else instantiate(node)

    inputs = _inst("train.inputs") or _inst("data.batcher")
    model = _inst("train.model") or _inst("model.model_fn")
    optimizer = _inst("train.optimizer") or _inst("optim.optimizer")
    lr_schedule_fn = _inst("train.lr_schedule_fn") or _inst("schedule.lr_schedule_fn")

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

    # 4) Uruchomienie treningu (train = supervised_trainer.train)
    output_dir = "runs/my_run"
    trax_trainer.train(output_dir=output_dir, **train_cfg)

if __name__ == "__main__":
    main()
