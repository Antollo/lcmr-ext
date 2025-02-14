from numbers import Real
from typing import Collection

import torch.nn as nn
from lcmr.grammar.scene_data import SceneData
from lcmr.utils.guards import typechecked

from lcmr_ext.loss.base_loss import BaseLoss


@typechecked
class CombinedLoss(BaseLoss):
    def __init__(self, *args: tuple[Real, BaseLoss]):
        super().__init__()

        self.w_arr: list[Real] = [w for w, _ in args]
        self.loss_fn_list: Collection[BaseLoss] = nn.ModuleList([loss_fn for _, loss_fn in args])

    def value(
        self,
        y_true: SceneData,
        y_pred: SceneData,
    ):
        return sum([w * loss_fn(y_true, y_pred) for w, loss_fn in zip(self.w_arr, self.loss_fn_list)])

    def reset(self):
        super().reset()
        for loss_fn in self.loss_fn_list:
            loss_fn.reset()

    def __str__(self) -> str:
        return f"{type(self).__name__}: {self.compute():.4f} " + ", ".join([f"{str(loss_fn)}*{w}" for w, loss_fn in zip(self.w_arr, self.loss_fn_list)])
