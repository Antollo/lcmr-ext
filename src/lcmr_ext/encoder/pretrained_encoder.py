import torch
from torchtyping import TensorType
from torchvision.transforms.functional import resize
import timm
from typing import Optional

from lcmr.encoder import Encoder
from lcmr.utils.guards import typechecked, batch_dim, height_dim, width_dim, reduced_height_dim, reduced_width_dim


@typechecked
class PretrainedEncoder(Encoder):
    def __init__(self, model_name, input_size: Optional[tuple[int, int]] = None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.model = timm.create_model(model_name, pretrained=True, **kwargs)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self, x: TensorType[batch_dim, height_dim, width_dim, 3, torch.float32]
    ) -> TensorType[batch_dim, -1, reduced_height_dim, reduced_width_dim, torch.float32]:
        # BHWC to BCHW
        x = x.permute(0, 3, 1, 2)
        if self.input_size != None:
            x = resize(x, self.input_size, antialias=False)
        return self.model.forward_features(x)  # .permute(0, 3, 1, 2)
