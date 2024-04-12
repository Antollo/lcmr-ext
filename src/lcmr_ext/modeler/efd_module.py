import torch
import torch.nn as nn
from torchtyping import TensorType
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from lcmr.utils.guards import typechecked, batch_dim, object_dim
from lcmr.utils.fourier_shape_descriptors import normalize_efd, FourierDescriptorsGenerator, FourierDescriptorsGeneratorOptions


class EfdModuleMode(Enum):
    Direct = 1
    Prototype = 2


@typechecked
@dataclass
class EfdModuleConfig:
    order: int
    num_prototypes: int
    mode: EfdModuleMode
    # These should be set set automatically by modeler:
    input_dim: Optional[int] = None
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None


@typechecked
class MLP(nn.Module):
    def __init__(self, config: EfdModuleConfig, output_dim: int):
        super().__init__()

        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.output_dim = output_dim

        self.num_layers = self.num_layers
        h = [self.hidden_dim] * (self.num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([self.input_dim] + h, h + [self.output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@typechecked
class EfdModule(nn.Module):
    def __init__(self, config: EfdModuleConfig):
        super().__init__()

        self.config = config

        if config.mode == EfdModuleMode.Direct:
            self.mlp = MLP(config, config.order * 4)
        elif config.mode == EfdModuleMode.Prototype:
            self.mlp = MLP(config, config.num_prototypes)
            generator = FourierDescriptorsGenerator(FourierDescriptorsGeneratorOptions(order=config.order))
            self.prototypes = nn.Parameter(generator.sample(config.num_prototypes))
            # noisy_circle_efd = torch.randn((config.num_prototypes, config.order, 4), dtype=torch.float32) / 10
            # noisy_circle_efd[..., 0, 0] = 1.0
            # noisy_circle_efd[..., 0, 3] = -1.0
            # noisy_circle_efd[..., 0, 1:3] = 0
            # noisy_circle_efd = normalize_efd(noisy_circle_efd)
            # self.prototypes = nn.Parameter(noisy_circle_efd)
            self.prototypes.requires_grad = False
        else:
            assert False

    def forward(self, hidden_state: TensorType[batch_dim, object_dim, -1, torch.float32]) -> TensorType[batch_dim, object_dim, -1, 4, torch.float32]:
        if self.config.mode == EfdModuleMode.Direct:
            efd = self.mlp(hidden_state).unflatten(-1, (-1, 4))
            efd[..., 0, 0] = 1.0
            efd[..., 0, 3] = -torch.sigmoid(efd[..., 0, 3])
            efd[..., 0, 1:3] = 0.0
            efd[..., 1:, :] = torch.tanh(efd[..., 1:, :])
        elif self.config.mode == EfdModuleMode.Prototype:
            selection = self.mlp(hidden_state)
            selection = nn.functional.softmax(selection, dim=-1)
            self.selection = selection
            selection = selection[..., None, None]
            prototypes = normalize_efd(self.prototypes)[None, None]
            efd = (selection * prototypes).sum(dim=-3)
            
        # TODO: else throw

        efd = normalize_efd(efd)
        return efd
