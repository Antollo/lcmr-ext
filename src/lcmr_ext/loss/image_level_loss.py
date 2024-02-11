import abc
import torch
import torch.nn as nn
from torchtyping import TensorType

from lcmr.utils.guards import batch_dim, height_dim, width_dim, channel_dim


class ImageLevelLoss(nn.Module, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        y_true: TensorType[batch_dim, height_dim, width_dim, channel_dim, torch.float32],
        y_pred: TensorType[batch_dim, height_dim, width_dim, channel_dim, torch.float32],
    ):
        pass
