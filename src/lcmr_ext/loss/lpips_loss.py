import torch
from torchtyping import TensorType
import lpips

from lcmr.utils.guards import typechecked, ImageBHWC3
from lcmr_ext.loss.image_level_loss import ImageLevelLoss


@typechecked
class LPIPSLoss(ImageLevelLoss):
    def __init__(self):
        super().__init__()
        self.model = lpips.LPIPS(net="vgg")

    def forward(
        self,
        y_true: ImageBHWC3,
        y_pred: ImageBHWC3,
    ):
        return self.model(y_true.permute(0, 3, 1, 2), y_pred.permute(0, 3, 1, 2), normalize=True).mean()
