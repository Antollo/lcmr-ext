from typing import Optional

import torch
from lcmr.encoder.pretrained_encoder import PretrainedEncoder
from lcmr.utils.guards import ImageBHWC3, batch_dim, reduced_height_dim, reduced_width_dim, typechecked
from torchtyping import TensorType


@typechecked
class DinoV2Encoder(PretrainedEncoder):
    def __init__(self, input_size: Optional[tuple[int, int]] = None):
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        super().__init__(model, input_size=input_size)

    def forward(self, x: ImageBHWC3) -> TensorType[batch_dim, -1, reduced_height_dim, reduced_width_dim, torch.float32]:
        x = x.permute(0, 3, 1, 2)
        x = self.transform(x)
        _, _, H, W = x.shape
        x = self.model.forward_features(x)["x_norm_patchtokens"]
        B, _, C = x.shape
        x = x.reshape(B, H // 14, W // 14, C).permute(0, 3, 1, 2)
        return x
