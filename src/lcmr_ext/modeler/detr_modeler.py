import torch
import torch.nn as nn
from torchtyping import TensorType

from lcmr.grammar import Scene
from lcmr.modeler import Modeler
from lcmr.utils.guards import typechecked, batch_dim, reduced_height_dim, reduced_width_dim, channel_dim

# https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=h91rsIPl7tVl


@typechecked
class DETRModeler(Modeler):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """

    def __init__(self, hidden_dim=128, nheads=2, num_encoder_layers=4, num_decoder_layers=4, num_queries=7):
        super().__init__()

        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, batch_first=True, dropout=0.0)
        # UWAGA: dropout w tym transformerze bardzo przeszkadza!

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.to_translation = nn.Linear(hidden_dim, 2)
        self.to_scale = nn.Linear(hidden_dim, 2)
        self.to_color = nn.Linear(hidden_dim, 3)
        self.to_confidence = nn.Linear(hidden_dim, 1)
        # self.to_angle = nn.Linear(hidden_dim, 1)
        self.to_angle_a = nn.Linear(hidden_dim, 1)
        self.to_angle_b = nn.Linear(hidden_dim, 1)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, x: TensorType[batch_dim, reduced_height_dim, reduced_width_dim, channel_dim, torch.float32]) -> Scene:
        # convert from 2048 to 256 feature planes for the transformer
        batch_len = x.shape[0]
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(0)
        )

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).transpose(-1, -2), self.query_pos.unsqueeze(0).repeat(batch_len, 1, 1))

        # finally project transformer outputs to class labels and bounding boxes
        return Scene.from_tensors(
            translation=torch.sigmoid(self.to_translation(h)).unsqueeze(1),
            scale=torch.sigmoid(self.to_scale(h)).unsqueeze(1),
            color=torch.sigmoid(self.to_color(h)).unsqueeze(1),
            confidence=torch.sigmoid(self.to_confidence(h)).unsqueeze(1),
            angle=torch.atan2(torch.tanh(self.to_angle_a(h)), torch.tanh(self.to_angle_b(h))).unsqueeze(
                1
            ),  # torch.sigmoid(self.to_angle(h)).unsqueeze(1) * np.pi * 2
        ).to(next(self.parameters()).device)
