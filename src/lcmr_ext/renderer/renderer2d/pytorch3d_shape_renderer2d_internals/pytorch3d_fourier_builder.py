import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, TexturesVertex

from lcmr.grammar import Object
from lcmr.grammar.shapes import Shape2D
from lcmr.utils.guards import typechecked
from lcmr.utils.fourier_shape_descriptors import reconstruct_contour, triangularize_contour
from .pytorch3d_shape_builder import Pytorch3DShapeBuilder


@typechecked
class Pytorch3DFourierBuilder(Pytorch3DShapeBuilder):
    def __init__(self, raster_size: tuple[int, int], n_verts: int, device: torch.device):
        super().__init__(Shape2D.FOURIER_SHAPE, raster_size, n_verts, device)

    def _build(self, objects: Object) -> Meshes:
        batch_len, layer_len, object_len = objects.shape

        # flatten batch and layers
        fourierCoefficients = objects.fourierCoefficients.flatten(0, 1)
        transformation = objects.transformation.matrix.flatten(0, 1)

        # Shape: batch, layer, object, vertex, channel
        object_idx = torch.arange(start=1, end=object_len + 1, dtype=torch.float32, device=self.device)[None, None, :, None, None]
        object_idx = object_idx.expand(batch_len, layer_len, -1, -1, -1)
        colors = torch.cat((objects.appearance.color[..., None, :], objects.appearance.confidence[..., None, :], object_idx), dim=-1)
        colors = colors.expand(-1, -1, -1, self.n_verts, -1).flatten(2, 3).flatten(0, 1)  # flatten objects in layers, flatten layers and batch

        verts = reconstruct_contour(fourierCoefficients, n_points=self.n_verts)

        verts = torch.nn.functional.pad(verts, (0, 1), "constant", 1.0)
        verts = (transformation[..., None, :, :] @ verts[..., None]).squeeze(-1)
        # assign confidence as "z" (would fix z-fighting when using traditional shaders)
        verts[:, :, :, 2, None] = objects.appearance.confidence.clamp(0.001, 0.999).flatten(0, 1)[..., None, :]  # should be in range (-inf, 1)

        faces = [triangularize_contour(v.squeeze(0)) for v in verts[..., :2].split(1)]
        max_len = max([idx.shape[0] for idx in faces])
        faces = [np.pad(idx, ((0, max_len - idx.shape[0]), (0, 0)), constant_values=-1)[None, ...] for idx in faces]
        faces = torch.from_numpy(np.concatenate(faces, axis=0)).to(self.device)

        verts = verts.flatten(1, 2)
        verts *= self.scale
        verts += self.translation

        return Meshes(verts=verts, faces=faces, textures=TexturesVertex(colors))
