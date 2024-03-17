import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, TexturesVertex

from lcmr.grammar import Object
from lcmr.grammar.shapes import Shape2D
from lcmr.utils.guards import typechecked
from .pytorch3d_shape_builder import Pytorch3DShapeBuilder


@typechecked
class Pytorch3DDiskBuilder(Pytorch3DShapeBuilder):
    def __init__(self, raster_size: tuple[int, int], n_verts: int, device: torch.device):
        super().__init__(Shape2D.DISK, raster_size, n_verts, device)

        # prepare disk primitive
        angles = torch.linspace(0, 2 * np.pi, n_verts)

        x = (torch.cos(angles))[..., None]
        y = (torch.sin(angles))[..., None]
        z = torch.ones(n_verts)[..., None]

        # shape: batch, layer, object, ...
        self.verts_disk = torch.cat((x, y, z), dim=-1)[None, None, None, ...].to(device)

        # make circle from triangles with points only on edge of the circle
        # (triangles looks like sharp teeth on cartoon character drawing)
        self.faces_disk = torch.tensor(
            [[i, i + 1, n_verts - i - 1] for i in range(n_verts // 2)] + [[n_verts - i - 1, n_verts - i - 2, i + 1] for i in range(n_verts // 2)], device=device
        )[None, None, None, ...]

    def _build(self, objects: Object) -> Meshes:
        batch_len, layer_len, object_len = objects.shape

        verts = self.verts_disk @ torch.transpose(objects.transformation.matrix, -1, -2)

        # assign confidence as "z" (would fix z-fighting when using traditional shaders))
        verts[:, :, :, :, 2, None] = objects.appearance.confidence[..., None, :]
        
        faces = self.faces_disk.repeat(batch_len, layer_len, object_len, 1, 1)
        faces_disk_offset = torch.arange(object_len, dtype=torch.float32, device=self.device)[None, None, :, None, None] * (self.n_verts)

        faces = faces + faces_disk_offset

        object_idx = torch.arange(start=1, end=object_len + 1, dtype=torch.float32, device=self.device)[None, None, :, None, None]
        object_idx = object_idx.repeat(batch_len, layer_len, 1, 1, 1)
        colors = torch.cat((objects.appearance.color[..., None, :], objects.appearance.confidence[..., None, :], object_idx), dim=-1)

        # repeat color for each vertex (shape: batch, layer, object, vertex, channel)
        colors = colors.repeat(1, 1, 1, self.n_verts, 1)

        # flatten objects in layers, flatten batch and layers
        verts = verts.flatten(2, 3).flatten(0, 1)
        faces = faces.flatten(2, 3).flatten(0, 1)
        colors = colors.flatten(2, 3).flatten(0, 1)

        verts *= self.scale
        verts += self.translation

        return Meshes(verts=verts, faces=faces, textures=TexturesVertex(colors))
