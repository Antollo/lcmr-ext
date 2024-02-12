import torch
from torchvision.transforms.transforms import Resize, Compose
from torchtyping import TensorType
import open_clip
from typing import Optional
from collections import OrderedDict

from lcmr.utils.guards import typechecked, ImageBHWC3
from lcmr_ext.loss.image_level_loss import ImageLevelLoss


# Based on https://github.com/yael-vinker/CLIPasso/blob/main/models/loss.py

@typechecked
class CLIPLoss(ImageLevelLoss):
    def __init__(self, input_size: Optional[tuple[int, int]] = None):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")

        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        transforms = [preprocess.transforms[-1]]  # Normalize
        if input_size != None:
            transforms.append(Resize(size=input_size))

        self.transform = Compose(transforms)

        self.feature_maps = OrderedDict()

        for i in range(12):
            model.visual.transformer.resblocks[i].register_forward_hook(self.make_hook(i))

    def make_hook(self, name: int):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.feature_maps[name] = output.permute(1, 0, 2)
            else:
                self.feature_maps[name] = output
        return hook
        
    def encode_image(self, image: ImageBHWC3):
        fc = self.model.encode_image(self.transform(image.permute(0, 3, 1, 2)))
        feature_maps = self.feature_maps
        return fc, feature_maps
        

    def forward(
        self,
        y_true: ImageBHWC3,
        y_pred: ImageBHWC3,
    ):
        fc_true, fm_true = self.encode_image(y_true)
        fc_pred, fm_pred = self.encode_image(y_pred)
        
        fc_loss = (1 - torch.cosine_similarity(fc_true, fc_pred, dim=1)).mean()
        
        fm_loss = 0
        for i in [2, 3]:
            fm_loss = fm_loss + torch.square(fm_true[i] - fm_pred[i]).mean()

        return 0.1 * fc_loss + fm_loss
