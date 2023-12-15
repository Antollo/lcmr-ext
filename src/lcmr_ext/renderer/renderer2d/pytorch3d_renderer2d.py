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
    SoftGouraudShader,
    TexturesVertex,
    AmbientLights,
)
from skimage.morphology import disk
from kornia.morphology import closing

from lcmr.grammar import Scene
from lcmr.renderer.renderer2d import Renderer2D
from lcmr.utils.guards import typechecked, batch_dim, height_dim, width_dim


# TODO: support shape, support composition, support layer.scale
@typechecked
class PyTorch3DRenderer2D(Renderer2D):
    def __init__(
        self, raster_size: tuple[int, int], background_color: TensorType[4, torch.float32] = torch.zeros(4), device: torch.device = torch.device("cpu")
    ):
        super().__init__(raster_size)

        self.device = device
        self.background = background_color[None, None, ...].to(device).repeat(*self.raster_size, 1)[None, ...]

        h, w = raster_size
        self.translation = torch.tensor([-h / w if w > h else 0, w / h if h > w else 0, 0], device=device)
        self.scale = torch.tensor([w / h if w > h else 1, -h / w if h > w else -1, 0], device=device)

        # prepare disk primitive
        self.v_count = 63
        radius = 1
        angles = torch.linspace(0, 2 * np.pi + 2 * np.pi / self.v_count, self.v_count)

        x = radius * torch.cat((torch.zeros(1), torch.cos(angles)))[..., None]
        y = radius * torch.cat((torch.zeros(1), torch.sin(angles)))[..., None]
        z = torch.ones(self.v_count + 1)[..., None]

        # shape: batch, layer, object, ...
        self.verts_disk = torch.cat((x, y, z), dim=-1)[None, None, None, ...].to(device)
        # "0, i + 1, i + 2" = non-overlapping triangles
        # "0, i + 1, i + 3" = overlapping triangles
        self.faces_disk = torch.tensor([[0, i + 1, i + 3] for i in range(self.v_count - 2)], device=device)[None, None, None, ...]
        self.object_len = -1
        self.colors_disk = torch.ones(1, 1, 1, 1, self.v_count + 1, 1, device=device)

        # https://pytorch3d.org/tutorials/fit_textured_mesh
        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=raster_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
            faces_per_pixel=50,
        )
        R, T = look_at_view_transform(at=((0, 0, 1),), up=((0, 1, 0),))
        camera = FoVOrthographicCameras(device=device, R=R, T=T, min_x=-1, max_x=0, min_y=-1, max_y=0)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
            shader=SoftGouraudShader(device=device, cameras=camera, lights=AmbientLights(device=device)),
        )

        self.closing_kernel = torch.from_numpy(disk(1, dtype=np.float32)).to(device)

    def render(self, scene: Scene, with_alpha: bool = True) -> TensorType[batch_dim, height_dim, width_dim, 4, torch.float32]:
        assert scene.device == self.device, f"Scene ({scene.device}) should be on the same device as {type(self).__name__} ({self.device})"

        batch_len, layer_len, object_len = scene.layer.object.shape

        verts = self.verts_disk @ torch.transpose(scene.layer.object.transformation.matrix, -1, -2)
        faces = self.faces_disk.repeat(batch_len, layer_len, object_len, 1, 1)
        if self.object_len != object_len:
            self.faces_disk_offset = torch.arange(object_len, dtype=torch.float32, device=self.device)[None, None, :, None, None] * (self.v_count + 1)
            self.object_len = object_len
        faces = faces + self.faces_disk_offset
        colors = self.colors_disk * scene.layer.object.appearance.color[..., None, :]

        # flatten objects in layers, flatten layers and batch
        verts = verts.flatten(2, 3).flatten(0, 1)
        faces = faces.flatten(2, 3).flatten(0, 1)
        colors = colors.flatten(3, 4).flatten(0, 2)

        verts *= self.scale
        verts += self.translation

        textures = TexturesVertex(colors)
        meshes = Meshes(verts=verts, faces=faces, textures=textures)

        # TODO: Implement pytorch3d "shader" to render alpha in single pass
        # https://github.com/facebookresearch/pytorch3d/issues/737

        # draw colors
        color_rendered = self.renderer(meshes)

        if with_alpha:
            # draw confidence aka alpha as single channel
            colors = self.colors_disk * scene.layer.object.appearance.confidence[..., None, :]
            colors = colors.flatten(3, 4).flatten(0, 2)
            textures = TexturesVertex(colors)
            meshes.textures = textures
            alpha_rendered = self.renderer(meshes)  # still renders 3 channels for color

            color, silhouette = color_rendered.split((3, 1), dim=-1)
            # silhouette = silhouette ** 0.5 # "fixes" weird artifacts on edges? Maybe add more problems? TODO: investigate
            # silhouette = closing(silhouette.permute(0, 3, 1, 2), self.closing_kernel).permute(0, 2, 3, 1)
            alpha, _ = alpha_rendered.split((1, 3), dim=-1)
            color_rendered = torch.cat((color, alpha * silhouette), dim=-1)

        # unflatten layers and batch
        color_rendered = color_rendered.unflatten(0, (batch_len, layer_len))

        final_images = self.background.clone()
        for layer_idx in range(layer_len):
            final_images = self.alpha_compositing(color_rendered[:, layer_idx], final_images)
        return final_images
