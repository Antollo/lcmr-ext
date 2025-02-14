from copy import copy
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from typing import Optional

import torch
import torch.nn as nn
from lcmr.utils.elliptic_fourier_descriptors import EfdGenerator, EfdGeneratorOptions, normalize_efd
from lcmr.utils.guards import batch_dim, object_dim, typechecked
from torchtyping import TensorType


class EfdModuleMode(Enum):
    Direct = 1
    Prototype = 2
    Latent = 3
    PrototypeAttention = 4


@typechecked
@dataclass
class EfdModuleConfig:
    order: int
    mode: EfdModuleMode
    num_prototypes: int = 1
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
        layers = [(nn.LayerNorm(n), nn.Linear(n, k), nn.GELU()) for n, k in zip([self.input_dim] + h, h + [self.output_dim])]
        layers = list(chain(*layers))[:-1]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim, n_hidden_blocks):
        super().__init__()

        self.output_dim = output_dim

        self.input_layer = nn.Linear(latent_dim, hidden_dim)
        self.resnet_blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(n_hidden_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        for resnet_block in self.resnet_blocks:
            x = nn.functional.gelu(x + resnet_block(x))

        x = self.output_layer(x)

        return x


@typechecked
class EfdModule(nn.Module):
    def __init__(self, config: EfdModuleConfig):
        super().__init__()

        self.config = config

        if config.mode == EfdModuleMode.Direct:
            self.mlp = MLP(config, config.order * 4)
        elif config.mode == EfdModuleMode.Latent:
            self.mlp = MLP(config, 64)
            self.batch_norm = nn.BatchNorm1d(64)
            self.decoder = Decoder(latent_dim=64, output_dim=256, hidden_dim=256, n_hidden_blocks=2)
            # self.decoder.load_state_dict(torch.load("early_stopping.state_dict", weights_only=True))
        elif config.mode == EfdModuleMode.Prototype:
            self.mlp = MLP(config, config.num_prototypes)
            generator = EfdGenerator(EfdGeneratorOptions(order=config.order))
            self.prototypes = nn.Parameter(generator.sample(config.num_prototypes))
            self.prototypes.requires_grad = False
        elif config.mode == EfdModuleMode.PrototypeAttention:
            self.query = MLP(config, output_dim=config.input_dim)
            config_copy = copy(config)
            config_copy.input_dim = config.order * 4
            self.key = MLP(config_copy, output_dim=config.input_dim)

            generator = EfdGenerator(EfdGeneratorOptions(order=config.order))
            self.prototypes = nn.Parameter(generator.sample(config.num_prototypes))
            self.prototypes.requires_grad = False
        else:
            assert False

    @staticmethod
    def normalized_efd_activation(efd: TensorType[batch_dim, object_dim, -1, 4, torch.float32]) -> TensorType[batch_dim, object_dim, -1, 4, torch.float32]:
        efd[..., 0, 0] = 1.0
        efd[..., 0, 3] = -torch.sigmoid(efd[..., 0, 3])
        efd[..., 0, 1:3] = 0.0
        efd[..., 1:, :] = torch.tanh(efd[..., 1:, :])
        efd = normalize_efd(efd)
        return efd

    def forward(self, hidden_state: TensorType[batch_dim, object_dim, -1, torch.float32]) -> TensorType[batch_dim, object_dim, -1, 4, torch.float32]:
        if self.config.mode == EfdModuleMode.Direct:
            efd = self.mlp(hidden_state).unflatten(-1, (-1, 4))
            efd = EfdModule.normalized_efd_activation(efd)
        elif self.config.mode == EfdModuleMode.Latent:
            latent = self.mlp(hidden_state)
            batch_len, object_len, _ = latent.shape
            latent = latent.flatten(0, 1)
            latent = self.batch_norm(latent)
            latent = latent.unflatten(0, (batch_len, object_len))
            efd = self.decoder(latent)
            efd = efd.unflatten(-1, (-1, 4))
            efd = EfdModule.normalized_efd_activation(efd)
        elif self.config.mode == EfdModuleMode.Prototype:
            selection = self.mlp(hidden_state)
            selection = nn.functional.softmax(selection, dim=-1)
            self.selection = selection
            selection = selection[..., None, None]
            prototypes = normalize_efd(self.prototypes)[None, None]
            efd = (selection * prototypes).sum(dim=-3)
            efd = normalize_efd(efd)
        elif self.config.mode == EfdModuleMode.PrototypeAttention:
            prototypes = normalize_efd(self.prototypes)

            query = self.query(hidden_state)
            key = self.key(prototypes.detach().flatten(-2, -1))
            attention = torch.matmul(query, key.transpose(-1, -2)) / (self.config.num_prototypes**0.5)
            attention = nn.functional.softmax(attention, dim=-1).clip(0.0001, 0.9999)
            self.selection = attention
            efd = (attention[..., None, None] * prototypes[None, None]).sum(dim=-3)

            # query = self.query(hidden_state)
            # key = self.key(prototypes.detach().flatten(-2, -1))[None]
            # value = prototypes.flatten(-2, -1)[None]
            # efd = nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0.01, scale=(self.config.num_prototypes**0.5)).unflatten(-1, (-1, 4))

            efd = normalize_efd(efd)
        else:
            assert False

        return efd


# TODO: move
import matplotlib.pyplot as plt
import seaborn as sns
from lcmr.utils.elliptic_fourier_descriptors import normalize_efd, reconstruct_contour

sns.set_theme()


def plot_prototypes(prototypes):
    contours = reconstruct_contour(normalize_efd(prototypes), n_points=256).detach().cpu().numpy()
    fig, axs = plt.subplots(nrows=1, ncols=len(prototypes))
    for i, (ax, contour) in enumerate(zip(axs, contours)):
        ax.title.set_text(f"{i}")
        ax.plot(contour[..., 0], contour[..., 1])
        ax.axis("square")
    fig.set_size_inches(1.5 * contours.shape[0], 2)
    fig.tight_layout()
    plt.show()
