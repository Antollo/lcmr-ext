import torch
from typing import Type
from torchtyping import TensorType
from dataclasses import dataclass

from lcmr.utils.guards import typechecked
from lcmr.renderer.renderer2d import Renderer2D
from lcmr_ext.renderer.renderer2d import PyTorch3DRenderer2D


@typechecked
@dataclass
class DatasetOptions:
    data_len: int = 0
    seed: int = 123
    num_blobs: int = 7
    background_color: TensorType[4, torch.float32] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    raster_size: tuple[int, int] = (128, 128)
    Renderer: Type[Renderer2D] = PyTorch3DRenderer2D
    scenes: bool = True
    device: torch.device = torch.device("cpu")
    concurrent: bool = True
    pool_size: int = 8
    use_cache: bool = False
    cache_filename: str = ""
