from typing import Collection

import torch
from lcmr.grammar import Scene
from lcmr.grammar.scene_data import SceneData
from lcmr.utils.guards import typechecked
from torchmetrics.aggregation import MeanMetric

from lcmr_ext.loss.base_loss import BaseLoss


@typechecked
class SceneLoss(BaseLoss):
    def __init__(self, fields: str = "tsrceb"):
        super().__init__()

        self.fields = fields
        self.metrics: Collection[MeanMetric] = torch.nn.ModuleList(MeanMetric() for _ in range(len(fields) + 1))

    def value(
        self,
        y_true: SceneData,
        y_pred: SceneData,
    ):
        dist = Scene.dist(y_true.scene, y_pred.scene, fields=self.fields, aggregate=False)
        for d, m in zip(dist, self.metrics):
            m.update(d.detach().mean())
        return sum(dist)

    def reset(self):
        super().reset()
        for m in self.metrics:
            m.reset()

    def __str__(self) -> str:
        return f"{type(self).__name__}: {self.compute():.4f} " + ", ".join([f"{f}: {m.compute():.4f}" for f, m in zip(["con"] + list(self.fields), self.metrics)])
