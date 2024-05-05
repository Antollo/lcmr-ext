import torch
import torch.nn as nn
from torchtyping import TensorType

from lcmr.grammar import Scene
from lcmr.modeler import Modeler
from lcmr.utils.guards import typechecked, batch_dim, reduced_height_dim, reduced_width_dim, channel_dim
from lcmr_ext.modeler.modeler_config import ModelerConfig


from .modeler_head import ModelerHead


@typechecked
class SimpleModeler(Modeler):
    def __init__(self, config: ModelerConfig, hidden_dim: int = 256):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(config.encoder_feature_dim, hidden_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.head = ModelerHead(config, hidden_dim)
        self.to_efd = self.head.to_efd

    def forward(self, x: TensorType[batch_dim, reduced_height_dim, reduced_width_dim, channel_dim, torch.float32]) -> Scene:
        return self.head(self.model(x)[:, None, ...])
