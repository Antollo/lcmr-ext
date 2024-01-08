import torch
import random
import slideio

from lcmr_ext.dataset.dataset_base import DatasetBase
from lcmr_ext.dataset.dataset_options import DatasetOptions


class DatasetSlideio(DatasetBase):
    def __init__(self, options: DatasetOptions):
        options.cache_filename = f"{type(self).__name__}-len-{options.data_len}-seed-{options.seed}-raster_size-{'-'.join(map(str, options.raster_size))}-background_color-{'-'.join(map(str, options.background_color.tolist()))}.pickle"

        super().__init__(options)

    def get_images(self):
        self.set_seed()

        patch_size = (1024, 1024)
        slide = slideio.open_slide("GTEX-1128S-0126.svs")
        scene = slide.get_scene(0)

        _, _, s_w, s_h = scene.rect
        w, h = patch_size

        image = scene.read_block((random.randint(0, s_w - w), random.randint(0, s_h - h), *patch_size), size=(128, 128))

        counter = 0
        images = []

        while counter < self.options.data_len:
            image = scene.read_block((random.randint(0, s_w - w), random.randint(0, s_h - h), *patch_size), size=self.options.raster_size)
            if image.var() > 2000:
                images.append(torch.from_numpy(image).to(torch.float32) / 255)
                counter += 1

        return images
