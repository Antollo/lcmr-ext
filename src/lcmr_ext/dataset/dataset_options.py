import torch
from typing import Type
from torchtyping import TensorType
from dataclasses import dataclass

from lcmr.utils.guards import typechecked
from lcmr.renderer.renderer2d import Renderer2D
from lcmr.renderer.renderer2d import OpenGLRenderer2D
from lcmr.utils.colors import colors


@typechecked
@dataclass
class DatasetOptions:
    name: str = ""
    split: str = "train"
    data_len: int = 1
    seed: int = 123
    num_blobs: int = 1
    background_color: TensorType[4, torch.float32] = colors.black
    raster_size: tuple[int, int] = (128, 128)
    Renderer: Type[Renderer2D] = OpenGLRenderer2D
    scenes: bool = True
    device: torch.device = torch.device("cpu")
    concurrent: bool = True
    pool_size: int = 8
    use_cache: bool = False
    cache_filename: str = ""
