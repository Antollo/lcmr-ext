import torch
from torchtyping import TensorType
import lpips

from lcmr.utils.guards import batch_dim, height_dim, width_dim, typechecked
from lcmr_ext.loss.image_level_loss import ImageLevelLoss


@typechecked
class LPIPSLoss(ImageLevelLoss):
    def __init__(self):
        super().__init__()
        self.model = lpips.LPIPS(net="vgg")

    def forward(
        self,
        y_true: TensorType[batch_dim, height_dim, width_dim, 3, torch.float32],
        y_pred: TensorType[batch_dim, height_dim, width_dim, 3, torch.float32],
    ):

        return self.model(y_true.permute(0, 3, 1, 2), y_pred.permute(0, 3, 1, 2), normalize=True).mean()
