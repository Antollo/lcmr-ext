import abc
import torch
from pytorch3d.structures import Meshes
from typing import Optional

from lcmr.grammar import Object
from lcmr.grammar.shapes import Shape2D
from lcmr.utils.guards import typechecked


@typechecked
class Pytorch3DShapeBuilder(abc.ABC):
    def __init__(self, objectShape: Shape2D, raster_size: tuple[int, int], n_verts: int, device: torch.device):
        self.objectShape = objectShape
        self.n_verts = n_verts

        h, w = raster_size
        self.translation = torch.tensor([-h / w if w > h else 0, w / h if h > w else 0, 0], device=device)
        self.scale = torch.tensor([w / h if w > h else 1, -h / w if h > w else -1, 1], device=device)

        self.device = device

    @abc.abstractmethod
    def _build(self, objects: Object) -> Meshes:
        pass

    def build(self, objects: Object) -> Optional[Meshes]:
        mask = (objects.objectShape == self.objectShape.value).squeeze(-1)
        if mask.sum() > 0:
            return self._build(objects)
        else:
            return None
