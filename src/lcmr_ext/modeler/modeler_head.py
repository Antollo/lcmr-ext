import torch
import torch.nn as nn
from lcmr.grammar import Scene
from lcmr.grammar.shapes import Shape2D
from lcmr.modeler import Modeler
from lcmr.utils.guards import batch_dim, object_dim, typechecked
from torchtyping import TensorType
from transformers.models.detr.modeling_detr import DetrMLPPredictionHead

from lcmr_ext.modeler.efd_module import EfdModule
from lcmr_ext.modeler.modeler_config import ModelerConfig


@typechecked
class ModelerHead(Modeler):
    def __init__(self, config: ModelerConfig, hidden_dim: int):
        super().__init__()

        self.use_single_scale = config.use_single_scale
        self.num_queries = config.num_slots

        # prediction heads
        def make_head(output_dim: int):
            return DetrMLPPredictionHead(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=config.prediction_head_layers)

        self.to_translation = make_head(output_dim=2)

        self.to_scale = make_head(output_dim=(1 if self.use_single_scale else 2))

        self.to_color = make_head(output_dim=3)

        if config.use_confidence:
            self.to_confidence = make_head(output_dim=1)
        else:
            self.to_confidence = None

        self.to_angle = make_head(output_dim=2)

        self.to_background_color = make_head(output_dim=3)

        efd_module_config = config.efd_module_config
        if efd_module_config != None:
            efd_module_config.input_dim = hidden_dim
            efd_module_config.hidden_dim = hidden_dim
            efd_module_config.num_layers = config.prediction_head_layers
            self.to_efd = EfdModule(efd_module_config)

    def forward(self, hidden_state: TensorType[batch_dim, object_dim, -1, torch.float32]) -> Scene:
        device = next(self.parameters()).device
        batch_size, object_size, _ = hidden_state.shape

        translation = torch.sigmoid(self.to_translation(hidden_state)).unsqueeze(1)

        scale = torch.sigmoid(self.to_scale(hidden_state)).unsqueeze(1)
        if self.use_single_scale:
            scale = scale.expand(-1, -1, -1, 2)

        color = torch.sigmoid(self.to_color(hidden_state)).unsqueeze(1)

        if self.to_confidence != None:
            confidence = torch.sigmoid(self.to_confidence(hidden_state)).unsqueeze(1)
        else:
            confidence = torch.ones((batch_size, 1, self.num_queries, 1), dtype=torch.float32, device=device)

        rotation_vec = torch.tanh(self.to_angle(hidden_state))
        rotation_vec = nn.functional.normalize(rotation_vec, dim=-1)
        angle = torch.atan2(rotation_vec[..., 0, None], rotation_vec[..., 1, None]).unsqueeze(1)

        efd = self.to_efd(hidden_state).unsqueeze(1) if self.to_efd != None else None

        background_color = torch.sigmoid(self.to_background_color(hidden_state.mean(dim=-2)))

        objectShape = Shape2D.EFD_SHAPE.value if self.to_efd != None else Shape2D.DISK.value

        return Scene.from_tensors_sparse(
            translation=translation,
            scale=scale,
            color=color,
            confidence=confidence,
            angle=angle,
            efd=efd,
            objectShape=torch.ones((batch_size, 1, self.num_queries, 1), dtype=torch.uint8, device=device) * objectShape,
            backgroundColor=background_color,
            device=device,
        )
