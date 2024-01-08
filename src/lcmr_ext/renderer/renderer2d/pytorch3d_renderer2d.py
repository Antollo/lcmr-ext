import torch
import numpy as np
from typing import Union
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
from skimage.morphology import disk

from lcmr.grammar import Scene
from lcmr.renderer.renderer2d import Renderer2D
from lcmr.utils.guards import typechecked, batch_dim, height_dim, width_dim


def simple_flat_shading_rgba(meshes, fragments, lights, cameras, materials) -> torch.Tensor:
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

    verts_colors_shaded = verts_colors
    face_colors = verts_colors_shaded[faces]
    colors = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_colors)
    return colors


#closing_kernel = torch.from_numpy(disk(5, dtype=np.float32)).to("cuda:0")


def simple_flat_rgba_blend(colors: torch.Tensor, fragments, blend_params: BlendParams) -> torch.Tensor:
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
    
    #N, H, W, K = fragments.pix_to_face.shape
    
    prob_map = torch.sigmoid(-(fragments.dists - 0.0004) / blend_params.sigma)
    colors[..., 3] *= prob_map#**0.1

    # each "layer" (dimension named "K") in colors represents i-th face (triangle) present in this pixel
    # unfortunately more than one face of the same object may be presents

    # TRICK 1

    # the trick is to z-buffer, each object has its own depth that we can use as index
    zbuf = fragments.zbuf[..., None]
    # trick with abs
    # z=-1 == background, z=2,3,4,... == objects    --->    abs(z)-1 == good indices (0,1,2,...)
    index = (torch.abs(zbuf) - 1).round().to(torch.int64)

    # TRICK 2

    # if we know number of faces per object then floor(face_idx / faces_per_object) is object index
    # faces_per_object = 16
    # pix_to_face = fragments.pix_to_face[..., None]
    # index = torch.div(pix_to_face, 16, rounding_mode="floor") + 1

    # assign each object to separate "layer"
    # downside: "layer" number limits how many objects we can have
    # TODO: better solution is to divide alphas by object reps in each pixel ()
    new_colors = torch.zeros_like(colors)
    new_colors.scatter_reduce_(-2, torch.broadcast_to(index, colors.shape), colors, reduce="amax")  # scatter_reduce_ with reduce="amax" to get right alpha
    colors = new_colors

    
    
    # alpha = colors[..., 3]#, None]
    # alpha = alpha.permute(0, 3, 1, 2).flatten(0, 1)
    # alpha = alpha
    # alpha = closing(alpha[:, None, ...], kernel=closing_kernel)[:, 0, ...]
    # alpha = gaussian_blur2d(alpha[:, None, ...], (7, 7), (1, 1))[:, 0, ...]
    # alpha = alpha.unflatten(0, (N, K)).permute(0, 2, 3, 1)
    # alpha = alpha[..., None]
    
    # "shading" starts here:
    
    alpha = colors[..., 3, None]
    
    
    rgb = colors[..., :3]
    alpha_sum = alpha.sum(dim=-2)

    # alpha_sum might be 0
    rgb_blended = torch.nan_to_num((rgb * alpha).sum(dim=-2) / alpha_sum, 0.0, 0.0, 0.0)
    rgba = torch.cat((rgb_blended, alpha.sum(dim=-2)), dim=-1)

    return rgba.clamp(0.0, 1.0)


class SimpleFlatRgbaShader(ShaderBase):
    """
    Simple flat per vertex lighting.

    Based on `SoftGouraudShader`.
    """

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
        images = simple_flat_rgba_blend(pixel_colors, fragments, self.blend_params)
        return images


# TODO: support shape, support composition, support layer.scale
@typechecked
class PyTorch3DRenderer2D(Renderer2D):
    def __init__(
        self,
        raster_size: tuple[int, int],
        background_color: TensorType[4, torch.float32] = torch.zeros(4),
        device: torch.device = torch.device("cpu"),
        with_alpha: bool = True,
        v_count: int = 64,
    ):
        super().__init__(raster_size)

        self.device = device
        self.v_count = v_count
        self.with_alpha = with_alpha  # TODO: to be replaced with Scene's blending property
        self.background = background_color[None, None, ...].to(device).repeat(*self.raster_size, 1)[None, ...]

        h, w = raster_size
        self.translation = torch.tensor([-h / w if w > h else 0, w / h if h > w else 0, 0], device=device)
        self.scale = torch.tensor([w / h if w > h else 1, -h / w if h > w else -1, 1], device=device)

        # prepare disk primitive
        radius = 1
        angles = torch.linspace(0, 2 * np.pi, v_count)

        x = radius * (torch.cos(angles))[..., None]
        y = radius * (torch.sin(angles))[..., None]
        z = torch.ones(v_count)[..., None]

        # shape: batch, layer, object, ...
        self.verts_disk = torch.cat((x, y, z), dim=-1)[None, None, None, ...].to(device)

        # make circle from triangles with points only on edge of the circle
        # (triangles looks like sharp teeth on cartoon character drawing)
        self.faces_disk = torch.tensor(
            [[i, i + 1, v_count - i - 1] for i in range(v_count // 2)] + [[v_count - i - 1, v_count - i - 2, i + 1] for i in range(v_count // 2)], device=device
        )[None, None, None, ...]

        # https://pytorch3d.org/tutorials/fit_textured_mesh
        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=raster_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
            faces_per_pixel=20,
        )
        R, T = look_at_view_transform(at=((0, 0, 0),), up=((0, 1, 0),))
        camera = FoVOrthographicCameras(device=device, R=R, T=T, min_x=-1, max_x=0, min_y=-1, max_y=0)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
            shader=SimpleFlatRgbaShader(
                device=device,
                cameras=camera,
                lights=AmbientLights(device=device), # Not used by our shader
                blend_params=BlendParams(),
            ),
        )

        self.closing_kernel = torch.from_numpy(disk(1, dtype=np.float32)).to(device)

    def render(self, scene: Scene, with_alpha: Union[bool, None] = None) -> TensorType[batch_dim, height_dim, width_dim, 4, torch.float32]:
        assert scene.device == self.device, f"Scene ({scene.device}) should be on the same device as {type(self).__name__} ({self.device})"

        batch_len, layer_len, object_len = scene.layer.object.shape

        verts = self.verts_disk @ torch.transpose(scene.layer.object.transformation.matrix, -1, -2)

        # assing consecutive "z"s for each object (would fix z-fighting when using traditional shaders)
        verts[:, :, :, :, 2, None] = torch.arange(-object_len, 0, dtype=torch.float32, device=self.device)[None, None, :, None, None]

        faces = self.faces_disk.repeat(batch_len, layer_len, object_len, 1, 1)
        self.faces_disk_offset = torch.arange(object_len, dtype=torch.float32, device=self.device)[None, None, :, None, None] * (self.v_count)

        faces = faces + self.faces_disk_offset # this can be cached

        colors = torch.cat((scene.layer.object.appearance.color[..., None, :], scene.layer.object.appearance.confidence[..., None, :]), dim=-1)
        
        if self.with_alpha if with_alpha == None else with_alpha:
            pass
        else:
            colors[..., 3] = 1
            
        # Repeat color for each point, shape: batch, layer, object, vertex, channel
        colors = colors.repeat(1, 1, 1, self.v_count, 1)

        # flatten objects in layers, flatten layers and batch
        verts = verts.flatten(2, 3).flatten(0, 1)
        faces = faces.flatten(2, 3).flatten(0, 1)
        colors = colors.flatten(2, 3).flatten(0, 1)

        verts *= self.scale
        verts += self.translation

        textures = TexturesVertex(colors)
        # (?) somehow all vertices for each layer ends up as single "mesh" in pytorch3d's shaders
        # not a problem for now but might cause some troubles in future
        meshes = Meshes(verts=verts, faces=faces, textures=textures)

        # This might be useful to concat more than one Meshes structures without manually
        # setting indices for each instance (that would be painful if `v_count` is different
        # in those structures)
        # meshes = pytorch3d.structures.join_meshes_as_scene([meshes1, meshes2])

        # TODO: Implement pytorch3d "shader" to render alpha in single pass
        # https://github.com/facebookresearch/pytorch3d/issues/737

        # draw colors
        color_rendered = self.renderer(meshes)

        # unflatten layers and batch
        color_rendered = color_rendered.unflatten(0, (batch_len, layer_len))

        final_images = self.background.clone()
        for layer_idx in range(layer_len):
            # TODO: follow Scene's blending property
            final_images = self.alpha_compositing(color_rendered[:, layer_idx], final_images) 
        return final_images
