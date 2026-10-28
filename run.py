import os
import sys
from logging import getLogger

import torch
import yaml

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.trainer import Trainer
from recbole.utils import (
    get_environment,
    get_flops,
    init_logger,
    init_seed,
    set_color,
)

from model.mainModel import mainModel
from utils.flowcf_dataloader import patch_flowcf_dataloader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and/or evaluate mainModel generator")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path for eval-only mode")
    args, remaining = parser.parse_known_args()
    # Strip our custom flags from sys.argv so RecBole's internal argparse doesn't see them
    sys.argv = [sys.argv[0]] + remaining

    with open(args.config, "r", encoding="utf-8") as file:
        yaml_config = yaml.safe_load(file)
    model_config = yaml_config.get("model", None)

    patch_flowcf_dataloader()

    config = Config(model=locals()[model_config], config_file_list=[args.config])
    init_seed(config["seed"], config["reproducibility"])

    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = locals()[config["model"]](config, train_data.dataset).to(config["device"])

    logger.info(model)

    trainer = Trainer(config, model)

    do_train = config["do_train"] if "do_train" in config else True

    if do_train:
        transform = construct_transform(config)
        flops = get_flops(model, dataset, config["device"], logger, transform)
        logger.info(set_color("FLOPs", "blue") + f": {flops}")

        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, show_progress=config["show_progress"]
        )
        logger.info(set_color("best valid", "yellow") + f": {best_valid_result}")
    else:
        # 纯评估模式：从 checkpoint 加载权重
        ckpt_path = args.checkpoint or (config["checkpoint_path"] if "checkpoint_path" in config else "")
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"do_train=False 时需要指定有效的 checkpoint。"
                f"请在 config 中设置 checkpoint_path 或用 --checkpoint 参数传入。"
                f"当前值: '{ckpt_path}'"
            )
        checkpoint = torch.load(ckpt_path, map_location=config["device"], weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        logger.info(set_color("Loaded checkpoint", "green") + f": {ckpt_path}")

    load_best = do_train  # eval-only: weights already loaded manually
    test_result = trainer.evaluate(test_data, load_best_model=load_best, show_progress=config["show_progress"])
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    try:
        environment_tb = get_environment(config)
        logger.info(
            "The running environment of this training is as follows:\n"
            + environment_tb.draw()
        )
    except ModuleNotFoundError as exc:
        logger.warning(f"Skip environment table because optional dependency is missing: {exc.name}")
