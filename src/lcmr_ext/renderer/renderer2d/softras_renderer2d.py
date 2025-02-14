from typing import Optional, Sequence

import numpy as np
import torch
from lcmr.grammar import Appearance, Layer, Object, Scene
from lcmr.grammar.scene_data import SceneData
from lcmr.grammar.shapes import Shape2D
from lcmr.renderer.renderer2d import Renderer2D
from lcmr.utils.colors import colors
from lcmr.utils.elliptic_fourier_descriptors import reconstruct_contour
from lcmr.utils.guards import typechecked, batch_dim, vec_dim
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import AmbientLights, BlendParams, FoVOrthographicCameras, MeshRasterizer, MeshRenderer, RasterizationSettings, TexturesVertex, look_at_view_transform
from pytorch3d.renderer.mesh.shader import Fragments, ShaderBase
from pytorch3d.structures import Meshes
from torchtyping import TensorType
from lcmr.utils.fourier_descriptors import cart_to_polar, reconstruct_rho, normalize_fourier_descriptors
from .pytorch3d_shape_renderer2d_internals import Pytorch3DDiskBuilder, Pytorch3DEfdBuilder
from kornia.filters import box_blur


class F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        s = grad_output.shape[-2]
        s = torch.linspace(1, 1e-1, s, device=grad_output.device)[None, None, None, :, None]
        #grad_output = grad_output * s * 1e-5
        return grad_output


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(contour):
    x1 = contour[..., :-1, 0]
    x2 = contour[..., 1:, 0]
    y1 = contour[..., :-1, 1]
    y2 = contour[..., 1:, 1]

    x1_2 = x1[..., None, :]
    x2_2 = x2[..., None, :]
    y1_2 = y1[..., None, :]
    y2_2 = y2[..., None, :]

    x1 = x1[..., None]
    x2 = x2[..., None]
    y1 = y1[..., None]
    y2 = y2[..., None]

    A = (x1, y1)
    B = (x2, y2)
    C = (x1_2, y1_2)
    D = (x2_2, y2_2)

    temp_1 = ccw(A, C, D) != ccw(B, C, D)
    temp_2 = ccw(A, B, C) != ccw(A, B, D)
    s = torch.logical_and(temp_1, temp_2)  # .to(torch.float32)
    # print(s.shape)
    s = torch.diagonal_scatter(s, torch.zeros((*s.shape[:2], s.shape[2] - 1), device=s.device), 1, -2, -1)
    s[..., 0, -1] = 0
    s[..., -1, 0] = 0
    s = torch.nonzero(s.triu(), as_tuple=True)

    # print(torch.cat((s[2][..., None], s[3][..., None]), dim=-1))

    z = torch.ones_like(x1[..., 0])

    z[s[0], s[1], s[2] + 1] *= -1
    z[s[0], s[1], s[3]] *= -1

    # print((s[2] > s[3]).to(torch.float32).sum())
    z = torch.cumprod(z, dim=-1)
    return z


@typechecked
def soft_shift(
    v: TensorType[batch_dim, vec_dim, torch.float32], shift: TensorType[batch_dim, 1, torch.float32], temperature: float = 0.5
) -> TensorType[batch_dim, vec_dim, torch.float32]:
    _, v_len = v.shape
    device = v.device

    # Convert 0-1 shift to absolute units (maintain gradient)
    shift_scaled = (shift + 1) * v_len  # (B,)

    # Create grid indices
    j = torch.arange(v_len, device=device).view(1, 1, -1)  # (1, 1, v)
    i = torch.arange(v_len, device=device).view(1, -1, 1)  # (1, v, 1)

    # Calculate target positions with circular wrapping
    target = (i - shift_scaled.view(-1, 1, 1)) % v_len  # (B, v, 1)

    # Compute minimal circular distance
    distance = torch.abs(j - target)
    distance = torch.min(distance, v_len - distance)  # (B, v, v)

    # Create shift matrix using softmax over distances
    shift_matrix = torch.softmax(-distance / temperature, dim=-1)  # (B, v, v)

    # Apply batched matrix multiplication
    return torch.bmm(shift_matrix, v.unsqueeze(-1)).squeeze(-1)


# TODO: support composition, support layer.scale
@typechecked
class SoftRasRenderer2D(Renderer2D):
    def __init__(
        self,
        raster_size: tuple[int, int],
        background_color: Optional[TensorType[4, torch.float32]] = None,
        device: torch.device = torch.device("cpu"),
        n_verts: int = 128,
        return_alpha: bool = False,
        sigma: float = 1 / 400,
    ):
        super().__init__(raster_size=raster_size, device=device)

        if background_color == None:
            background_color = colors.black

        self.background = background_color[None, None, ...].to(device).expand(*self.raster_size, -1)[None, ...]

        self.return_alpha = return_alpha
        self.n_verts = n_verts
        self.sigma = sigma

        xs = torch.linspace(0, 1, steps=self.raster_size[0], device=self.device)
        ys = torch.linspace(0, 1, steps=self.raster_size[1], device=self.device)
        self.xy = torch.roll(torch.cartesian_prod(ys, xs).view(1, *self.raster_size, 2), 1, dims=-1)

    # @torch.compile
    def render(
        self,
        scene: Scene,
    ) -> SceneData:
        assert scene.device == self.device, f"Scene ({scene.device}) should be on the same device as {type(self).__name__} ({self.device})"

        batch_len, layer_len, object_len = scene.layer.object.shape

        # TODO: handle layers
        object: Object = scene.layer.object[:, 0]  # .flatten(0, 1)

        if object.fd != None:

            object: Object = object.flatten(0, 1)

            fd = object.fd
            fd = F.apply(fd)
            
            #print(fd[..., 0, :])
            translation = object.transformation.translation
            angle = object.transformation.angle
            scale = object.transformation.scale
            
            #print(angle.shape)

            rho = reconstruct_rho(fd, n_points=self.n_verts, a=angle[..., None])
            #rho = soft_shift(rho, angle / (torch.pi * 2), temperature=1.0)

            xy_translated = self.xy - translation[..., None, None, :]
            dist = xy_translated.norm(dim=-1, p=2)

            with torch.no_grad():
                phi_grid = torch.atan2(xy_translated[..., 1], xy_translated[..., 0])
                idx = ((phi_grid + torch.pi) * (rho.shape[-1] / torch.pi / 2)).to(torch.int64) % rho.shape[-1]

            rho_grid = torch.gather(rho[:, None, :].expand(-1, self.raster_size[0], -1), 2, idx)
            rho_grid = box_blur(rho_grid[:, None], kernel_size=3)[:, 0] * scale.mean(dim=-1)[:, None, None]
            
            #print((dist).shape, scale.mean(dim=-1).shape)

            dist = ((rho_grid - dist)).unflatten(0, (batch_len, object_len))

        elif object.efd != None:
            # efd = F.apply(object.efd)
            efd = object.efd
            transformation = object.transformation.matrix

            verts = reconstruct_contour(efd, n_points=self.n_verts)

            verts = torch.nn.functional.pad(verts, (0, 1), "constant", 1.0)
            verts = (transformation[..., None, :, :] @ verts[..., None]).squeeze(-1)

            x1 = verts[..., :-1, 0, None, None]
            x2 = verts[..., 1:, 0, None, None]
            y1 = verts[..., :-1, 1, None, None]
            y2 = verts[..., 1:, 1, None, None]

            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - y1 * x2

            x, y = self.xy[..., 0], self.xy[..., 1]

            dist = -(a * x + b * y + c)
            dist = dist / (torch.sqrt(a**2 + b**2) + 0.000001)

            dist = dist * intersect(verts)[..., None, None]

            segment_dist = torch.sqrt((x - (x1 + x2) / 2) ** 2 + (y - (y1 + y2) / 2) ** 2) - torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / 2
            idx = torch.argmin(segment_dist, dim=-3, keepdim=True)
            dist = torch.gather(dist, -3, idx)[..., 0, :, :]

        mask = torch.sigmoid(dist / self.sigma)
        mask = mask[..., None]

        color = object.appearance.color[..., None, None, :]

        color = (color * torch.nn.functional.softmax(mask * 10, dim=1)).sum(dim=1)

        alpha = 1 - torch.prod(1 - mask, dim=1)

        if scene.backgroundColor != None:
            background_color = scene.backgroundColor
            background_color = background_color[:, None, None, :].expand(-1, *self.raster_size, -1)
        else:
            background_color = self.background

        color = alpha * color + (1 - alpha) * background_color[..., :3]

        color = torch.cat((color, torch.ones(*color.shape[:-1], 1, device=color.device)), dim=-1)

        return SceneData(
            scene=scene,
            image=color,
            mask=None,  # mask if self.return_alpha else None,
            batch_size=[batch_len],
        )
