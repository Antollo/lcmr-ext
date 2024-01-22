import torch
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor

from lcmr.grammar.scene import Scene
from lcmr_ext.dataset.dataset_options import DatasetOptions


def regenerate_job(options: DatasetOptions):
    renderer = options.Renderer(raster_size=options.raster_size, background_color=options.background_color, device=options.device)
    
    object_len = options.num_blobs

    scenes = [
        Scene.from_tensors_sparse(
            torch.rand(1, 1, object_len, 2),
            torch.rand(1, 1, object_len, 2) / 5 + 0.05,
            torch.rand(1, 1, object_len, 1),
            torch.rand(1, 1, object_len, 3),
            torch.rand(1, 1, object_len, 1) / 4 + 0.75,
        )
        for _ in range(options.data_len)
    ]
    scenes = scenes
    images = [renderer.render(scene)[..., :3].cpu() for scene in scenes]
    return list(zip(images, scenes))
        
class RandomDataset(Dataset):
    def __init__(self, options: DatasetOptions):
        self.options = options

        if options.concurrent:
            self.pool = ProcessPoolExecutor(max_workers=options.pool_size)
            self.futures = []
            for _ in range(options.pool_size):
                self.append_new_job()
        self.regenerate()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    
    def append_new_job(self):
        self.futures.append(self.pool.submit(regenerate_job, self.options))

    def regenerate(self):
        if self.options.concurrent:
            self.append_new_job()
            self.data = self.futures[0].result()
            del self.futures[0]
        else:
            self.data = regenerate_job(self.options)
        


class OnlineRegeneratingRandomDataset(Dataset):
    def __init__(self, options: DatasetOptions):
        self.options = options
        self.renderer = None

    def __getitem__(self, idx):
        if self.renderer == None:
            self.renderer = self.options.Renderer(raster_size=self.options.raster_size, background_color=self.options.background_color)

        object_len = self.options.num_blobs

        scene = Scene.from_tensors_sparse(
            torch.rand(1, 1, object_len, 2),
            torch.rand(1, 1, object_len, 2) / 5 + 0.05,
            torch.rand(1, 1, object_len, 1),
            torch.rand(1, 1, object_len, 3),
            torch.rand(1, 1, object_len, 1) / 4 + 0.75,
        )

        image = self.renderer.render(scene)[..., :3]

        return (image, scene)

    def __len__(self):
        return self.options.data_len
