import torch
import torch.nn as nn
from torchtyping import TensorType
from torchvision.models import resnet50, ResNet50_Weights

from lcmr.encoder import Encoder
from lcmr.utils.guards import typechecked, batch_dim, height_dim, width_dim, reduced_height_dim, reduced_width_dim


@typechecked
class ResNet50Encoder(Encoder):
    def __init__(self, replace_stride_with_dilation: [bool, bool, bool] = [False, False, False]):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights, replace_stride_with_dilation=replace_stride_with_dilation)

        self.model = nn.Sequential(*list(model.children())[:-2])

        # normalization for trained resnet
        mean = torch.tensor([0.485, 0.456, 0.406])[..., None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[..., None, None]

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(
        self, x: TensorType[batch_dim, height_dim, width_dim, 3, torch.float32]
    ) -> TensorType[batch_dim, 2048, reduced_height_dim, reduced_width_dim, torch.float32]:
        # BHWC to BCHW
        x = x.permute(0, 3, 1, 2)
        x = (x - self.mean) / self.std
        return self.model(x)
