from typing import Optional, Sequence

import numpy as np
import torch
from lcmr.grammar.scene_data import SceneData
from lcmr.grammar import Scene
from lcmr.renderer.renderer2d import Renderer2D
from lcmr.utils.colors import colors
from lcmr.utils.guards import typechecked
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import AmbientLights, BlendParams, FoVOrthographicCameras, MeshRasterizer, MeshRenderer, RasterizationSettings, TexturesVertex, look_at_view_transform
from pytorch3d.renderer.mesh.shader import Fragments, ShaderBase
from pytorch3d.structures import Meshes
from torchtyping import TensorType

from .pytorch3d_shape_renderer2d_internals import Pytorch3DDiskBuilder, Pytorch3DEfdBuilder


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
    values2 = torch.zeros_like(values)
    for i in range(n_objects):
        mask = indices == (i + 1)
        max_v, max_i = (values * mask).max(dim=-1, keepdim=True)
        max_v = max_v * mask.any(dim=-1, keepdim=True)
        values2 = values2.scatter_add(3, index=max_i, src=max_v)
    return values2


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
    
    eps = 1e-7

    # N, H, W, K = fragments.pix_to_face.shape

    # 0.0005 "enlarge" the triangles by a bit, reduce weird artifact on the edges
    prob_map = torch.sigmoid(-(fragments.dists - 0.0005) / blend_params.sigma)
    #prob_map = -fragments.dists
    #prob_map = prob_map - prob_map.amin()
    #prob_map = prob_map / prob_map.amax()
    #prob_map = prob_map.clip(0, 1)

    # each "layer" (dimension named "K") in colors represents i-th face (triangle) present in this pixel
    # unfortunately more than one face of the same object may be present

    rgb = colors[..., :3]
    #print(torch.unique(colors[..., 3]))
    #alpha = colors[..., 3, None] * prob_map
    object_idx = colors[..., 4, None].round().to(torch.int32)

    prob_map = select_max_value_for_each_index(object_idx[..., 0], prob_map.contiguous())[..., None] # .clamp(0.001, 0.999)
    

    weight = prob_map * torch.exp((colors[..., 3, None] - 1) * 10)
    weight_sum = weight.sum(dim=-2).clamp(min=eps)
    #print(alpha_sum)

    #print(rgb.shape, alpha.shape, (rgb * alpha).shape, alpha_sum.shape)
    rgb_blended = (rgb * weight).sum(dim=-2) / weight_sum
    
    mask = (colors[..., 3, None]) # > 0.5 ).to(torch.float32)
    
    alpha = 1.0 - torch.prod((1.0 - prob_map * mask.to(torch.float32)), dim=-2) # * colors[..., 3, None]
    #print(rgb_blended.shape, alpha.shape)
    rgba = torch.cat((rgb_blended, alpha), dim=-1)

    if return_alpha:
        return rgba, weight_sum
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
        background_color: Optional[TensorType[4, torch.float32]] = None,
        device: torch.device = torch.device("cpu"),
        n_verts: int = 128,
        faces_per_pixel: int = 32,
        return_alpha: bool = False,
        sigma: float = 1e-4,
    ):
        super().__init__(raster_size=raster_size, device=device)

        if background_color == None:
            background_color = colors.black

        self.background = background_color[None, None, ...].to(device).expand(*self.raster_size, -1)[None, ...]

        # https://pytorch3d.org/tutorials/fit_textured_mesh
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
        self.n_verts = n_verts

    @property
    def n_verts(self):
        return self._n_verts

    @n_verts.setter
    def n_verts(self, value):
        self._n_verts = value
        self.builders = [Pytorch3DDiskBuilder(self.raster_size, self._n_verts, self.device), Pytorch3DEfdBuilder(self.raster_size, self._n_verts, self.device)]

    def render(
        self,
        scene: Scene,
    ) -> SceneData:
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

        return SceneData(
            scene=scene,
            image=rgba,
            mask=rendered[1].to(self.device) if self.renderer.shader.return_alpha else None,
            batch_size=[batch_len],
        )
