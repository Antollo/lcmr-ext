import torch
from torchtyping import TensorType
import pydiffvg

from lcmr.grammar import Scene, Layer
from lcmr.renderer.renderer2d import Renderer2D
from lcmr.utils.guards import typechecked, ImageBHWC4, ImageHWC4


# TODO: support shape, support composition, support layer.scale
@typechecked
class PyDiffVgRenderer2D(Renderer2D):
    def __init__(
        self,
        raster_size: tuple[int, int],
        samples: int = 2,
        background_color: TensorType[4, torch.float32] = torch.zeros(4),
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(raster_size)

        self.device = device
        # WARNING: pydiffvg device is global
        pydiffvg.set_device(device)
        self.center = torch.zeros(2, dtype=torch.float32, device=device)
        self.radius = torch.ones(1, dtype=torch.float32, device=device)
        self.scale_matrix = torch.diag(torch.tensor([*reversed(raster_size), 1], dtype=torch.float32, device=device))
        self.background = background_color[None, None, ...].to(device).repeat(*raster_size, 1)
        self.transparent_background = torch.zeros((*raster_size, 4), device=device)
        self.samples = samples

        self.last_length = -1

    def render(self, scene: Scene) -> ImageBHWC4:
        assert scene.device == self.device, f"Scene ({scene.device}) should be on the same device as PyDiffVgRenderer2D ({self.device})"

        imgs = []
        for single_scene in scene:
            img = self.background

            # composition is ignored
            for layer in single_scene.layer:
                rendered_layer = self.render_layer(layer)
                img = self.alpha_compositing(rendered_layer, img)

            imgs.append(img[None, ...])

        return torch.vstack(imgs)

    def render_layer(self, layer: Layer) -> ImageHWC4:
        shape_to_canvas_arr = self.scale_matrix @ layer.object.transformation.matrix
        fill_color_arr = torch.cat((layer.object.appearance.color, layer.object.appearance.confidence), dim=-1)

        shapes = []
        shape_groups = []

        for i, (shape_to_canvas, fill_color) in enumerate(zip(shape_to_canvas_arr, fill_color_arr)):
            # object_shape is ignored
            disk = pydiffvg.Circle(radius=self.radius, center=self.center)
            ellipse_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([i]), fill_color=fill_color, shape_to_canvas=shape_to_canvas)
            shapes.append(disk)
            shape_groups.append(ellipse_group)

        scene_args = pydiffvg.RenderFunction.serialize_scene(*reversed(self.raster_size), shapes, shape_groups)
        torch.cuda.synchronize()  # pydiffvg might crash without cuda synchronization, is it asynchronous?
        img = pydiffvg.RenderFunction.apply(*reversed(self.raster_size), self.samples, self.samples, 0, self.transparent_background, *scene_args)

        return img
