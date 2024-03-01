import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, TexturesVertex

from lcmr.grammar import Object
from lcmr.grammar.shapes import Shape2D
from lcmr.utils.guards import typechecked
from .pytorch3d_shape_builder import Pytorch3DShapeBuilder

from lcmr.renderer.renderer2d.opengl_renderer2d_internals.opengl_fourier_renderer import reconstruct_contour, simplify_contour, triangularize_contour


@typechecked
class Pytorch3DFourierBuilder(Pytorch3DShapeBuilder):
    def __init__(self, raster_size: tuple[int, int], n_verts: int, device: torch.device):
        super().__init__(Shape2D.FOURIER_SHAPE, raster_size, n_verts, device)

    def _build(self, objects: Object) -> Meshes:
        batch_len, layer_len, object_len = objects.shape

        # flatten layers and batch
        fourierCoefficients = objects.fourierCoefficients.flatten(0, 1)
        transformation = objects.transformation.matrix.flatten(0, 1)
        
        colors = torch.cat((objects.appearance.color, objects.appearance.confidence), dim=-1).flatten(0, 1)
        colors = colors.repeat(1, 1, self.n_verts).reshape(batch_len * layer_len, -1, 4)


        verts = reconstruct_contour(fourierCoefficients, n_points=self.n_verts)

        verts = torch.nn.functional.pad(verts, (0, 1), "constant", 1.0)
        verts = (transformation[..., None, :, :] @ verts[..., None]).squeeze(-1)
        verts[:, :, :, 2, None] = torch.arange(-object_len, 0, dtype=torch.float32, device=self.device)[None, :, None, None]

        faces = [triangularize_contour(v.squeeze(0)) for v in verts[..., :2].split(1)]
        max_len = max([idx.shape[0] for idx in faces])
        faces = [np.pad(idx, ((0, max_len-idx.shape[0]), (0, 0)), constant_values=-1)[None, ...] for idx in faces]      
        faces = torch.from_numpy(np.concatenate(faces, axis=0)).to(self.device)
        
        verts = verts.flatten(1, 2)
        verts *= self.scale
        verts += self.translation

        return Meshes(verts=verts, faces=faces, textures=TexturesVertex(colors))
