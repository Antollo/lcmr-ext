import torch
import torch.nn as nn
from torchtyping import TensorType
from torchvision.models import resnet50, ResNet50_Weights
from typing import Optional

from lcmr.utils.guards import typechecked, batch_dim, height_dim, width_dim, reduced_height_dim, reduced_width_dim

from .pretrained_encoder import PretrainedEncoder


@typechecked
class ResNet50Encoder(PretrainedEncoder):
    def __init__(self, replace_stride_with_dilation: list[bool, bool, bool] = [False, False, False], input_size: Optional[tuple[int, int]] = None):

        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights, replace_stride_with_dilation=replace_stride_with_dilation)
        model = nn.Sequential(*list(model.children())[:-2])

        super().__init__(model, input_size=input_size)

    def forward(
        self, x: TensorType[batch_dim, height_dim, width_dim, 3, torch.float32]
    ) -> TensorType[batch_dim, 2048, reduced_height_dim, reduced_width_dim, torch.float32]:
        return super().forward(x)