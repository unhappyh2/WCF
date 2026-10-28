from typing import Literal

import recbole.data.utils as data_utils
from recbole.data.dataloader import (
    FullSortEvalDataLoader,
    NegSampleEvalDataLoader,
    UserDataLoader,
)
from recbole.utils import InputType


def patch_flowcf_dataloader():
    """为本项目的 FlowCF 复现注册 AE 风格 dataloader。

    官方 FlowCF 需要修改 RecBole 的 get_dataloader 注册表；这里用运行时 patch
    保持项目自包含，不直接改 site-packages。
    """
    original_get_dataloader = data_utils.get_dataloader

    def get_dataloader(config, phase: Literal["train", "valid", "test", "evaluation"]):
        input_type = config["MODEL_INPUT_TYPE"]
        if config["model"] in {"mainModel", "FlowCF"} and input_type == InputType.LISTWISE:
            if phase == "evaluation":
                phase = "test"
            if phase == "train":
                return UserDataLoader
            eval_mode = config["eval_args"]["mode"][phase]
            if eval_mode == "full":
                return FullSortEvalDataLoader
            return NegSampleEvalDataLoader
        return original_get_dataloader(config, phase)

    data_utils.get_dataloader = get_dataloader
    return original_get_dataloader
