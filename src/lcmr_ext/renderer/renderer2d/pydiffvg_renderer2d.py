from typing import Optional

import pydiffvg
import torch
from lcmr.grammar.scene_data import SceneData
from lcmr.grammar import Layer, Scene
from lcmr.grammar.shapes import Shape2D
from lcmr.renderer.renderer2d import Renderer2D
from lcmr.utils.colors import colors
from lcmr.utils.elliptic_fourier_descriptors import reconstruct_contour, triangulate_contour
from lcmr.utils.guards import ImageBHWC4, ImageHWC4, typechecked
from torchtyping import TensorType


# TODO: support shape, support composition, support layer.scale
@typechecked
class PyDiffVgRenderer2D(Renderer2D):
    def __init__(
        self,
        raster_size: tuple[int, int],
        samples: int = 2,
        background_color: Optional[TensorType[4, torch.float32]] = None,
        device: torch.device = torch.device("cpu"),
        n_verts: int = 128,
    ):
        super().__init__(raster_size=raster_size, device=device)

        if background_color == None:
            background_color = colors.black

        self.device = device
        self.n_verts = n_verts
        # WARNING: pydiffvg device is global
        pydiffvg.set_device(device)
        self.center = torch.zeros(2, dtype=torch.float32, device=device)
        self.radius = torch.ones(1, dtype=torch.float32, device=device)
        self.scale_matrix = torch.diag(torch.tensor([*reversed(raster_size), 1], dtype=torch.float32, device=device))
        self.background = background_color[None, None, ...].to(device).expand(*raster_size, -1)
        # self.transparent_background = torch.zeros((*raster_size, 4), device=device)
        self.samples = samples

        self.last_length = -1

    def render(self, scene: Scene) -> SceneData:
        assert scene.device == self.device, f"Scene ({scene.device}) should be on the same device as PyDiffVgRenderer2D ({self.device})"

        imgs = []

        for single_scene in scene:
            if single_scene.backgroundColor != None:
                background_color = single_scene.backgroundColor
                background_color = torch.cat((background_color, torch.ones(*background_color.shape[:-1], 1, device=background_color.device)), dim=-1)
                img = background_color[None, None, :].expand(*self.raster_size, -1)
            else:
                img = self.background.clone()

            # composition is ignored
            for layer in single_scene.layer:
                background = self.background.clone()
                background[..., 3] = 0
                rendered_layer = self.render_layer(layer, background)
                img = self.alpha_compositing(rendered_layer, img)

            imgs.append(img[None, ...])

        imgs = torch.vstack(imgs)
        return SceneData(
            scene=scene,
            image=imgs,
            batch_size=[len(scene)],
        )

    def render_layer(self, layer: Layer, background: ImageHWC4) -> ImageHWC4:
        shape_to_canvas_arr = self.scale_matrix @ layer.object.transformation.matrix
        fill_color_arr = torch.cat((layer.object.appearance.color, layer.object.appearance.confidence), dim=-1)

        shapes = []
        shape_groups = []

        object_shape_arr = layer.object.objectShape.cpu()
        if layer.object.efd != None:
            efd_arr = reconstruct_contour(layer.object.efd, n_points=self.n_verts)
        else:
            efd_arr = [None] * len(layer.object)

        for i, (shape_to_canvas, fill_color, object_shape, efd) in enumerate(zip(shape_to_canvas_arr, fill_color_arr, object_shape_arr, efd_arr)):
            object_shape = object_shape.item()
            if object_shape == Shape2D.DISK.value:
                shape = pydiffvg.Circle(radius=self.radius, center=self.center)
            elif object_shape == Shape2D.EFD_SHAPE.value:
                shape = pydiffvg.Polygon(points=efd, is_closed=True)
            else:
                # TODO
                assert False

            shape_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([i]), fill_color=fill_color, shape_to_canvas=shape_to_canvas)
            shapes.append(shape)
            shape_groups.append(shape_group)

        scene_args = pydiffvg.RenderFunction.serialize_scene(*reversed(self.raster_size), shapes, shape_groups)
        torch.cuda.synchronize()  # pydiffvg might crash without cuda synchronization, is it asynchronous?
        img = pydiffvg.RenderFunction.apply(*reversed(self.raster_size), self.samples, self.samples, 0, background, *scene_args)

        return img
