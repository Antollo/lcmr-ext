import torch
import torch.nn as nn
from torchtyping import TensorType

from lcmr.grammar import Scene
from lcmr.modeler import Modeler
from lcmr.utils.guards import typechecked, batch_dim, reduced_height_dim, reduced_width_dim, channel_dim

# Adapted from: https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb


@typechecked
class DemoDETRModeler(Modeler):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * single linear layer as output module (instead of MLP)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        nheads: int = 2,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_queries: int = 7,
        encoder_feature_dim: int = 2048,
        input_size: tuple[int, int] = (4, 4)
    ):
        super().__init__()

        self.input_projection = nn.Conv2d(encoder_feature_dim, hidden_dim, kernel_size=1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, batch_first=True, dropout=0.0)
        # Warning: dropout seems to be harmful

        # prediction heads
        # note that DETR has 3-layer MLPs
        self.to_translation = nn.Linear(hidden_dim, 2)
        self.to_scale = nn.Linear(hidden_dim, 2)
        self.to_color = nn.Linear(hidden_dim, 3)
        self.to_confidence = nn.Linear(hidden_dim, 1)
        self.to_angle = nn.Linear(hidden_dim, 2)

        # output positional encodings (object queries)
        self.query_position_embedding = nn.Parameter(torch.rand(num_queries, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embedding = nn.Parameter(torch.rand(input_size[0], hidden_dim // 2))
        self.col_embedding = nn.Parameter(torch.rand(input_size[1], hidden_dim // 2))

    def forward(self, x: TensorType[batch_dim, reduced_height_dim, reduced_width_dim, channel_dim, torch.float32]) -> Scene:
        # convert from 2048 to 256 feature planes for the transformer
        batch_len = x.shape[0]
        features = self.input_projection(x)

        # construct positional encodings
        H, W = features.shape[-2:]
        positional_encoding = (
            torch.cat(
                [
                    self.col_embedding[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embedding[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(0)
        )

        # propagate through the transformer
        hidden_state = self.transformer(
            positional_encoding + 0.1 * features.flatten(2).transpose(-1, -2), self.query_position_embedding.unsqueeze(0).repeat(batch_len, 1, 1)
        )

        # finally project transformer outputs to class labels and bounding boxes
        device = next(self.parameters()).device
        translation = torch.sigmoid(self.to_translation(hidden_state)).unsqueeze(1)
        scale = torch.sigmoid(self.to_scale(hidden_state)).unsqueeze(1)
        color = torch.sigmoid(self.to_color(hidden_state)).unsqueeze(1)
        confidence = torch.sigmoid(self.to_confidence(hidden_state)).unsqueeze(1)
        rotation_vec = torch.tanh(self.to_angle(hidden_state))
        rotation_vec = nn.functional.normalize(rotation_vec, dim=-1)
        angle = torch.atan2(rotation_vec[..., 0, None], rotation_vec[..., 1, None]).unsqueeze(1)

        return Scene.from_tensors_sparse(
            translation=translation,
            scale=scale,
            color=color,
            confidence=confidence,
            angle=angle,
        ).to(device)
