import torch
import numpy as np
from torchtyping import TensorType
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
    TexturesVertex,
    AmbientLights,
    BlendParams,
)
from pytorch3d.renderer.mesh.shader import ShaderBase, Fragments
from pytorch3d.ops import interpolate_face_attributes
from typing import Sequence, Union

from lcmr.grammar import Scene
from lcmr.renderer.renderer2d import Renderer2D
from lcmr.utils.guards import typechecked, ImageBHWC4, height_dim, width_dim
from .pytorch3d_shape_renderer2d_internals import Pytorch3DDiskBuilder, Pytorch3DFourierBuilder


def simple_flat_shading_rgba(meshes: Meshes, fragments: Fragments, lights, cameras, materials) -> torch.Tensor:
    """
    Apply per vertex shading. Don't compute any lighting. Interpolate the vertex shaded
    colors using the barycentric coordinates to get a color per pixel.

    Based on `gouraud_shading`.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights parameters
        cameras: Cameras class containing a batch of cameras parameters
        materials: Materials class containing a batch of material properties

    Returns:
        colors: (N, H, W, K, 3)
    """
    if not isinstance(meshes.textures, TexturesVertex):
        raise ValueError("Mesh textures must be an instance of TexturesVertex")

    faces = meshes.faces_packed()  # (F, 3)
    verts_colors = meshes.textures.verts_features_packed()  # (V, D)

    face_colors = verts_colors[faces]
    colors = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_colors)
    return colors


# TODO: remove the for loop, process "in batch", tensor of all masks will be required later to count ARI for multiple objects!
def select_max_value_for_each_index(indices: torch.Tensor, values: torch.Tensor):
    n_objects = int(torch.max(indices).item())
    for i in range(n_objects):
        mask = indices == (i + 1)
        max_v, max_i = (values * mask).max(dim=-1, keepdim=True)
        max_v *= mask.any(dim=-1, keepdim=True)
        values[mask] = 0
        values.scatter_add_(3, index=max_i, src=max_v)
    return values


# torch.compile seems to work fine
try:
    select_max_value_for_each_index = torch.compile(select_max_value_for_each_index)
except:
    print("Failed to use torch.compile")


def simple_flat_rgba_blend(colors: torch.Tensor, fragments: Fragments, blend_params: BlendParams, return_alpha: bool) -> Sequence[torch.Tensor]:
    """
    Simple weighted sum blending of top K faces to return an RGBA image
      - **RGB** - sum(rgb * alpha) / sum(alpha)
      - **A** - min(sum(alpha), 1)

    Based on `hard_rgb_blend`.

    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        fragments: the outputs of rasterization. From this we use
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image. This is used to
              determine the output shape.
        blend_params: BlendParams instance that contains a background_color
        field specifying the color for the background
    Returns:
        RGBA pixel_colors: (N, H, W, 4)
    """

    # N, H, W, K = fragments.pix_to_face.shape

    # 0.0005 "enlarge" the triangles by a bit, reduce weird artifact on the edges
    prob_map = torch.sigmoid(-(fragments.dists - 0.0005) / blend_params.sigma)

    # each "layer" (dimension named "K") in colors represents i-th face (triangle) present in this pixel
    # unfortunately more than one face of the same object may be present

    rgb = colors[..., :3]
    alpha = colors[..., 3, None] * prob_map[..., None]
    object_idx = colors[..., 4, None].round().to(torch.int32)

    alpha = select_max_value_for_each_index(object_idx[..., 0], alpha[..., 0].contiguous().clamp(0.001, 0.999))
    alpha = alpha[..., None]
    alpha_sum = alpha.sum(dim=-2)

    rgb_blended = torch.nan_to_num((rgb * alpha).sum(dim=-2) / alpha_sum, 0.0, 1.0, 0.0)
    rgba = torch.cat((rgb_blended, alpha.sum(dim=-2)), dim=-1)

    rgba = rgba.clamp(0.0, 1.0)
    if return_alpha:
        alpha_sum = alpha_sum.clamp(0.0, 1.0)
        return rgba, alpha_sum
    else:
        return (rgba,)


class SimpleFlatRgbaShader(ShaderBase):
    """
    Simple flat per vertex lighting.

    Based on `SoftGouraudShader`.
    """

    return_alpha: bool = False

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        pixel_colors = simple_flat_shading_rgba(
            meshes=meshes,
            fragments=fragments,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = simple_flat_rgba_blend(pixel_colors, fragments, self.blend_params, self.return_alpha)
        return images


# TODO: support composition, support layer.scale
@typechecked
class PyTorch3DRenderer2D(Renderer2D):
    def __init__(
        self,
        raster_size: tuple[int, int],
        background_color: TensorType[4, torch.float32] = torch.zeros(4),
        device: torch.device = torch.device("cpu"),
        n_verts: int = 48,
        faces_per_pixel: int = 32,
        return_alpha: bool = False,
    ):
        super().__init__(raster_size)

        self.device = device
        self.background = background_color[None, None, ...].to(device).expand(*self.raster_size, -1)[None, ...]

        # https://pytorch3d.org/tutorials/fit_textured_mesh
        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=raster_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
            faces_per_pixel=faces_per_pixel,  # IMPORTANT: faces_per_pixel should be set considering max_object_count, raster size, and object size
        )
        R, T = look_at_view_transform(at=((0, 0, 0),), up=((0, 1, 0),))
        camera = FoVOrthographicCameras(device=device, R=R, T=T, min_x=-1, max_x=0, min_y=-1, max_y=0)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
            shader=SimpleFlatRgbaShader(
                device=device,
                cameras=camera,
                lights=AmbientLights(device=device),  # Not used by our shader
                blend_params=BlendParams(),
            ),
        )
        self.renderer.shader.return_alpha = return_alpha
        self.builders = [Pytorch3DDiskBuilder(raster_size, n_verts, device), Pytorch3DFourierBuilder(raster_size, n_verts, device)]

    def render(
        self,
        scene: Scene,
    ) -> Union[ImageBHWC4, tuple[ImageBHWC4, TensorType[-1, height_dim, width_dim, 1, torch.float32]]]:
        # TODO: return a dataclass with 2 fields?
        assert scene.device == self.device, f"Scene ({scene.device}) should be on the same device as {type(self).__name__} ({self.device})"

        batch_len, layer_len, _ = scene.layer.object.shape

        meshes = [builder.build(scene.layer.object) for builder in self.builders]
        meshes = [mesh for mesh in meshes if mesh is not None]
        assert len(meshes) == 1, "Only one object shape type at time allowed"  # TODO
        meshes = meshes[0]

        # This might be useful to concat more than one Meshes structures without manually
        # setting indices for each instance
        # import pytorch3d.structures
        # meshes = pytorch3d.structures.join_meshes_as_batch(meshes)

        # draw colors
        rendered = self.renderer(meshes)
        rgba = rendered[0]

        # unflatten layers and batch
        rgba = rgba.unflatten(0, (batch_len, layer_len))

        if scene.backgroundColor != None:
            background_color = scene.backgroundColor
            background_color = torch.cat((background_color, torch.ones(*background_color.shape[:-1], 1, device=background_color.device)), dim=-1)
            background = background_color[:, None, None, :].expand(-1, *self.raster_size, -1)
        else:
            background = self.background.clone()

        for layer_idx in range(layer_len):
            # TODO: follow Scene's blending property
            background = self.alpha_compositing(rgba[:, layer_idx], background)

        rgba = background
        if self.renderer.shader.return_alpha:
            alpha = rendered[1].to(self.device)
            return rgba, alpha
        else:
            return rgba
