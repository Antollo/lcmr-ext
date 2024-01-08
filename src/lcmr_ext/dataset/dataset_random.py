import torch
from torch.utils.data import Dataset

from lcmr.grammar.scene import Scene
from lcmr_ext.dataset.dataset_options import DatasetOptions


class RandomDataset(Dataset):
    def __init__(self, options: DatasetOptions):
        self.options = options
        renderer = self.options.Renderer(raster_size=self.options.raster_size, background_color=self.options.background_color)

        torch.manual_seed(self.options.seed)

        object_len = options.num_blobs

        scenes = [
            Scene.from_tensors(
                torch.rand(1, 1, object_len, 2),
                torch.rand(1, 1, object_len, 2) / 5 + 0.01,
                torch.rand(1, 1, object_len, 1),
                torch.rand(1, 1, object_len, 3),
                torch.rand(1, 1, object_len, 1) / 4 + 0.75,
            )
            for _ in range(options.data_len)
        ]

        images = [renderer.render(scene)[..., :3] for scene in scenes]

        self.data = list(zip(images, scenes))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
