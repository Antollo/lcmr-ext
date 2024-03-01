import torch
from torchtyping import TensorType
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from typing import Optional

from lcmr.utils.guards import typechecked, ImageBHWC3, batch_dim, reduced_height_dim, reduced_width_dim

from lcmr.encoder.pretrained_encoder import PretrainedEncoder


@typechecked
class ConvNextBEncoder(PretrainedEncoder):
    def __init__(self, input_size: Optional[tuple[int, int]] = None):

        weights = ConvNeXt_Base_Weights.DEFAULT
        model = convnext_base(weights=weights).features

        super().__init__(model, input_size=input_size)

    def forward(self, x: ImageBHWC3) -> TensorType[batch_dim, 1024, reduced_height_dim, reduced_width_dim, torch.float32]:
        return super().forward(x)
