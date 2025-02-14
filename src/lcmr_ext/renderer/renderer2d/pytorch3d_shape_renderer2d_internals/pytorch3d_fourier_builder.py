import torch
from lcmr.grammar import Object
from lcmr.grammar.shapes import Shape2D
from lcmr.utils.elliptic_fourier_descriptors import reconstruct_contour, triangulate_contour
from lcmr.utils.guards import typechecked
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.structures.utils import list_to_packed

from .pytorch3d_shape_builder import Pytorch3DShapeBuilder


@typechecked
class Pytorch3DEfdBuilder(Pytorch3DShapeBuilder):
    def __init__(self, raster_size: tuple[int, int], n_verts: int, device: torch.device):
        super().__init__(Shape2D.EFD_SHAPE, raster_size, n_verts, device)

    def _build(self, objects: Object) -> Meshes:
        batch_len, layer_len, object_len = objects.shape

        # flatten batch and layers
        efd = objects.efd.flatten(0, 1)
        transformation = objects.transformation.matrix.flatten(0, 1)

        # Shape: batch, layer, object, vertex, channel
        object_idx = torch.arange(start=1, end=object_len + 1, dtype=torch.float32, device=self.device)[None, None, :, None, None]
        object_idx = object_idx.expand(batch_len, layer_len, -1, -1, -1)
        colors = torch.cat((objects.appearance.color[..., None, :], objects.appearance.confidence[..., None, :], object_idx), dim=-1)
        colors = colors.expand(-1, -1, -1, self.n_verts, -1).flatten(2, 3).flatten(0, 1)  # flatten objects in layers, flatten layers and batch

        verts = reconstruct_contour(efd, n_points=self.n_verts)

        verts = torch.nn.functional.pad(verts, (0, 1), "constant", 1.0)
        verts = (transformation[..., None, :, :] @ verts[..., None]).squeeze(-1)
        # assign confidence as "z" (would fix z-fighting when using traditional shaders)
        verts[:, :, :, 2, None] = objects.appearance.confidence.clamp(0.001, 0.999).flatten(0, 1)[..., None, :]  # should be in range (-inf, 1)

        faces, faces_list = triangulate_contour(verts[..., :2], return_list=True)

        verts = verts.flatten(1, 2)
        verts *= self.scale
        verts += self.translation

        mesh = Meshes(verts=verts, faces=faces, textures=TexturesVertex(colors))

        # Warning: the following offer a minor speedup by optimizing some pytorch3d internals (Mesh._compute_packed method)
        # We know that each shape have the same "verts" shape so calculating these indices is trivial

        # mesh._compute_packed()
        # assert torch.allclose(mesh._mesh_to_verts_packed_first_idx, torch.arange(0, batch_len * layer_len * object_len * self.n_verts, object_len * self.n_verts, device=self.device))
        # assert torch.allclose(mesh._verts_packed_to_mesh_idx, torch.arange(0, batch_len * layer_len, 1, device=self.device)[..., None].expand(-1, object_len * self.n_verts).flatten())

        mesh._faces_list = faces_list
        mesh._verts_list = verts
        mesh._verts_packed = verts
        mesh._mesh_to_verts_packed_first_idx = torch.arange(0, batch_len * layer_len * object_len * self.n_verts, object_len * self.n_verts, device=self.device)
        mesh._verts_packed_to_mesh_idx = torch.arange(0, batch_len * layer_len, 1, device=self.device)[..., None].expand(-1, object_len * self.n_verts).flatten()

        faces_list_to_packed = list_to_packed(mesh.faces_list())
        faces_packed = faces_list_to_packed[0]
        mesh._mesh_to_faces_packed_first_idx = faces_list_to_packed[2]
        mesh._faces_packed_to_mesh_idx = faces_list_to_packed[3]
        faces_packed_offset = mesh._mesh_to_verts_packed_first_idx[mesh._faces_packed_to_mesh_idx]
        mesh._faces_packed = faces_packed + faces_packed_offset.view(-1, 1)

        return mesh
