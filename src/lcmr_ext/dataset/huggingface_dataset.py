import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from typing import Optional
import multiprocessing
import platform

from lcmr_ext.dataset.dataset_options import DatasetOptions

# "fork" start method might result in deadlock
# https://discuss.huggingface.co/t/dataset-map-stuck-with-torch-set-num-threads-set-to-2-or-larger/37984
if platform.system() == "Linux":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass


class HuggingFaceDataset(Dataset):
    def __init__(self, options: DatasetOptions, data_dir: Optional[str] = None):
        self.options = options
        name = (options.name + ("" if data_dir == None else data_dir)).replace("/", "-").replace("\\", "-").replace("~", "")
        cache_filename = f"{name}-{options.split}-seed-{options.seed}-raster_size-{'-'.join(map(str, options.raster_size))}-background_color-{'-'.join(map(str, options.background_color.tolist()))}"

        try:
            self.dataset = load_from_disk(cache_filename)
        except:
            dataset = load_dataset(options.name, data_dir=data_dir, num_proc=options.pool_size).with_format("torch")
            if ("val" not in dataset and "validation" not in dataset) or "test" not in dataset:
                # split manually
                validation_ratio = 0.15
                test_ratio = 0.15
                train_test_val = dataset["train"].train_test_split(validation_ratio + test_ratio, seed=options.seed)
                train_dataset = train_test_val["train"]

                test_val = train_test_val["test"].train_test_split(test_ratio, seed=options.seed)
                val_dataset = test_val["train"]
                test_dataset = test_val["test"]

                dataset = dict(train=train_dataset, val=val_dataset, test=test_dataset)

            dataset = dataset[options.split]
            dataset.map(
                HuggingFaceDataset.transform,
                batched=True,
                batch_size=min(128, len(dataset) // options.pool_size + 1),
                num_proc=options.pool_size,
                load_from_cache_file=False,
                fn_kwargs=dict(options=options),
                new_fingerprint=str(hash(cache_filename)),
            ).save_to_disk(cache_filename)
            self.dataset = load_from_disk(cache_filename)

    def __getitem__(self, idx) -> dict:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def transform(data: dict, options: DatasetOptions) -> dict:
        import torch
        from torchvision.transforms.functional import resize
        from lcmr.renderer.renderer2d import Renderer2D

        batch = data["image"].to(options.device)
        batch = resize(batch.permute(0, 3, 1, 2), options.raster_size, antialias=True).permute(0, 2, 3, 1)
        batch = batch.to(torch.float32) / 255
        background = options.background_color[None, None, None, ...].to(options.device)
        batch = Renderer2D.alpha_compositing(batch, background)[..., :3]
        batch = (batch * 255).to(torch.uint8)
        batch = batch.cpu()
        return {"image": batch}

    @staticmethod
    def collate_fn(batch):
        batch = torch.cat([x["image"][None, ...] for x in batch], dim=0)
        batch = batch.to(torch.float32) / 255
        return batch
