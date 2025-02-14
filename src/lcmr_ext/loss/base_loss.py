import abc
from typing import Union

import torch
import torch.nn as nn
from lcmr.grammar.scene_data import SceneData
from lcmr.utils.guards import typechecked
from torchmetrics.aggregation import MeanMetric


@typechecked
class BaseLoss(nn.Module, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.mean_metric = MeanMetric()

    def forward(
        self,
        y_true: SceneData,
        y_pred: SceneData,
        update: bool = True,
    ):
        value = self.value(y_true, y_pred)

        if update:
            with torch.no_grad():
                self.mean_metric.update(value.detach())
        #print(y_true.device, y_pred.device, self.mean_metric.device)

        return value

    @abc.abstractmethod
    def value(self, y_true: SceneData, y_pred: SceneData) -> torch.Tensor:
        pass

    def update(self, value: Union[float, torch.Tensor]) -> torch.Tensor:
        self.mean_metric.update(value)
        return value

    def compute(self) -> torch.Tensor:
        return self.mean_metric.compute()

    def reset(self):
        self.mean_metric.reset()
    
    def __str__(self) -> str:
        return f"{type(self).__name__}: {self.compute():.4f}"
    
    def show(self):
        print(str(self))
