import torch
import torch.nn as nn
from torchtyping import TensorType
from torchvision.transforms.transforms import Normalize, Resize, Compose
from typing import Optional

from lcmr.encoder import Encoder
from lcmr.utils.guards import typechecked, ImageBHWC3, batch_dim, reduced_height_dim, reduced_width_dim, channel_dim


@typechecked
class PretrainedEncoder(Encoder):
    def __init__(self, model: nn.Module, input_size: Optional[tuple[int, int]] = None):
        super().__init__()

        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        transforms = [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        if input_size != None:
            transforms.append(Resize(size=input_size))

        self.transform = Compose(transforms)

    def forward(
        self, x: ImageBHWC3
    ) -> TensorType[batch_dim, channel_dim, reduced_height_dim, reduced_width_dim, torch.float32]:
        # BHWC to BCHW
        x = x.permute(0, 3, 1, 2)
        x = self.transform(x)
        return self.model(x)