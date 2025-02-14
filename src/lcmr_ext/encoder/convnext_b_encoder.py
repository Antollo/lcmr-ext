from typing import Optional

import torch
import torch.nn as nn
from lcmr.encoder.pretrained_encoder import PretrainedEncoder
from lcmr.utils.guards import ImageBHWC3, batch_dim, reduced_height_dim, reduced_width_dim, typechecked
from torchtyping import TensorType
from torchvision.models import ConvNeXt_Base_Weights, convnext_base


@typechecked
class ConvNextBEncoder(PretrainedEncoder):
    def __init__(self, input_size: Optional[tuple[int, int]] = None, feature_index: int = -1):

        weights = ConvNeXt_Base_Weights.DEFAULT
        model = convnext_base(weights=weights).features
        model = nn.Sequential(*list(model)[: len(model) + feature_index + 1])

        super().__init__(model, input_size=input_size)

    def forward(self, x: ImageBHWC3) -> TensorType[batch_dim, -1, reduced_height_dim, reduced_width_dim, torch.float32]:
        return super().forward(x)
dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')