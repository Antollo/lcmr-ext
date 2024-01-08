import abc
import torch
import random
import pickle
import platform
import numpy as np
from torch.utils.data import Dataset
from torch.multiprocessing import Pool, set_start_method
from itertools import repeat
from more_itertools import chunked, flatten
from math import ceil
from pathlib import Path

from lcmr_ext.utils import optimize_params
from lcmr_ext.dataset.dataset_options import DatasetOptions

if platform.system() == "Linux":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

# TODO: class for data entry


def scene_from_blobs(images, raster_size, background_color, num_blobs):
    import torch
    from random import randint
    from .scene_from_blobs import SceneFromBlobs

    device = torch.device(f"cuda:{randint(0, torch.cuda.device_count() - 1)}") if torch.cuda.is_available() else torch.device("cpu")
    scene_from_blobs = SceneFromBlobs(raster_size=raster_size, background_color=background_color, device=device)
    scenes = [scene_from_blobs.predict(img.to(device), num_blobs=num_blobs).cpu() for img in images]
    return scenes


class DatasetBase(Dataset, abc.ABC):
    def __init__(self, options: DatasetOptions):
        self.options = options

        if self.options.use_cache and Path(self.options.cache_filename).is_file():
            with open(self.options.cache_filename, "rb") as handle:
                self.data = pickle.load(handle)
        else:
            images = self.get_images()
            images = [image[None, ...] for image in images]
            if not options.scenes:
                self.data = images
            else:
                data_len = len(images)

                pool_size = max(min(self.options.pool_size, data_len // 2), 1)

                with Pool(pool_size) as p:
                    scenes = flatten(
                        p.starmap(
                            scene_from_blobs,
                            zip(
                                chunked(images, ceil(data_len / pool_size)),
                                repeat(self.options.raster_size, pool_size),
                                repeat(self.options.background_color, pool_size),
                                repeat(self.options.num_blobs, pool_size),
                            ),
                        )
                    )

                optimized_scenes = []
                batch_len = 128
                renderer = self.options.Renderer(
                    raster_size=self.options.raster_size, background_color=self.options.background_color, device=self.options.device
                )

                for img, scene in zip(chunked(images, batch_len), chunked(scenes, batch_len)):
                    img = torch.cat(img, dim=0).to(self.options.device)
                    scene = torch.cat(scene, dim=0).to(self.options.device)

                    t = scene.layer.object.transformation.translation
                    s = scene.layer.object.transformation.scale
                    a = scene.layer.object.transformation.angle
                    c = scene.layer.object.appearance.color
                    optimize_params(scene, img, renderer, [t, s, a, c])

                    scene = [s.cpu()[None, ...] for s in scene]
                    optimized_scenes.extend(scene)

                scenes = optimized_scenes
                self.data = list(zip(images, scenes))

            if self.options.use_cache:
                with open(self.options.cache_filename, "wb") as handle:
                    pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def set_seed(self):
        random.seed(self.options.seed)
        np.random.seed(self.options.seed)
        torch.manual_seed(self.options.seed)

    @abc.abstractmethod
    def get_images(self):
        pass
