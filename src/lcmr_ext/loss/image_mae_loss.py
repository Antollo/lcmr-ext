from lcmr.grammar.scene_data import SceneData
from lcmr.utils.guards import typechecked
from torch.nn.functional import l1_loss

from lcmr_ext.loss.base_loss import BaseLoss


@typechecked
class ImageMaeLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def value(
        self,
        y_true: SceneData,
        y_pred: SceneData,
    ):
        return l1_loss(y_true.image[..., :3], y_pred.image[..., :3])
