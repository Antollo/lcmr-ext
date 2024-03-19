import torch
from torch.utils.data import Dataset
from typing import Literal
from multi_object_datasets_torch import MultiDSprites

from lcmr.dataset.dataset_options import DatasetOptions


class MultiDSpritesDataset(Dataset):
    def __init__(self, options: DatasetOptions, version: Literal["binarized", "colored_on_colored", "colored_on_grayscale"]):
        self.options = options

        assert self.options.return_images
        assert not self.options.return_scenes
        assert options.raster_size == (64, 64)

        self.data = MultiDSprites("~/datasets", version=version, split=options.split.capitalize())

    def __getitem__(self, idx):
        return self.data[idx]["image"].permute(1, 2, 0).to(torch.float32)[None, ...] / 255

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return torch.cat(batch, dim=0)
