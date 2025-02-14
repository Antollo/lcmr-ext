from collections import OrderedDict
from typing import Optional

import open_clip
import torch
from lcmr.grammar.scene_data import SceneData
from lcmr.utils.guards import ImageBHWC3, typechecked
from torchvision.transforms.transforms import Compose, Resize
from torch.nn.functional import mse_loss
from lcmr_ext.loss.base_loss import BaseLoss

from open_clip.transformer import _expand_token

# Based on https://github.com/yael-vinker/CLIPasso/blob/main/models/loss.py


@typechecked
class CLIPLoss(BaseLoss):
    def __init__(self, input_size: Optional[tuple[int, int]] = (224, 224)):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")

        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        transforms = [preprocess.transforms[-1]]  # Normalize
        if input_size != None:
            transforms.append(Resize(size=input_size, antialias=True))

        self.transform = Compose(transforms)

        self.feature_maps = OrderedDict()

        # for i in range(12):
        #    model.visual.transformer.resblocks[i].register_forward_hook(self.make_hook(i))

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

    def encode_image_2(self, image: ImageBHWC3):
        feature_maps = []
        x = self.transform(image.permute(0, 3, 1, 2))
        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.model.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)

        x = self.model.visual.patch_dropout(x)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        for r in self.model.visual.transformer.resblocks[:3]:
            x = r(x)
            feature_maps.append(x)
        return feature_maps

    def value(
        self,
        y_true: SceneData,
        y_pred: SceneData,
    ):
        # fc_true, fm_true = self.encode_image(y_true.image[..., :3])
        # fc_pred, fm_pred = self.encode_image(y_pred.image[..., :3])

        fm_true = self.encode_image_2(y_true.image[..., :3])
        fm_pred = self.encode_image_2(y_pred.image[..., :3])

        # fc_loss = (1 - torch.cosine_similarity(fc_true, fc_pred, dim=1)).mean()
        # print(len(fm_true), fm_true[0].shape)

        fm_loss = sum([mse_loss(fm_true[i], fm_pred[i]) for i in [1, 2]])

        return fm_loss  # + 0.1 * fc_loss
