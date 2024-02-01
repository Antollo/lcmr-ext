import torch
from torchtyping import TensorType
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.transforms.functional import resize
from typing import Optional

from lcmr.encoder import Encoder
from lcmr.utils.guards import typechecked, batch_dim, height_dim, width_dim, reduced_height_dim, reduced_width_dim


@typechecked
class ConvNextBaseEncoder(Encoder):
    def __init__(self, input_size: Optional[tuple[int, int]] = None):
        super().__init__()
        self.input_size = input_size
        weights = ConvNeXt_Base_Weights.DEFAULT
        self.model = convnext_base(weights=weights)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # normalization for trained ConvNeXt model
        mean = torch.tensor([0.485, 0.456, 0.406])[..., None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[..., None, None]

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(
        self, x: TensorType[batch_dim, height_dim, width_dim, 3, torch.float32]
    ) -> TensorType[batch_dim, 1024, reduced_height_dim, reduced_width_dim, torch.float32]:
        # BHWC to BCHW
        x = x.permute(0, 3, 1, 2)
        x = (x - self.mean) / self.std
        if self.input_size != None:
            x = resize(x, self.input_size, antialias=False)
        return self.model.features(x)
