import torch
from torchtyping import TensorType
import timm

from lcmr.encoder import Encoder
from lcmr.utils.guards import typechecked, batch_dim, height_dim, width_dim, reduced_height_dim, reduced_width_dim


@typechecked
class PretrainedEncoder(Encoder):
    def __init__(self, model_name, **kwargs):
        super().__init__()           
        self.model = timm.create_model(model_name, pretrained=True, **kwargs)

    def forward(
        self, x: TensorType[batch_dim, height_dim, width_dim, 3, torch.float32]
    ) -> TensorType[batch_dim, -1, reduced_height_dim, reduced_width_dim, torch.float32]:
        # BHWC to BCHW
        x = x.permute(0, 3, 1, 2)
        return self.model.forward_features(x)
