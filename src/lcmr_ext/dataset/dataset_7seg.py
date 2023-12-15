import torch
import random
import numpy as np
from torch.utils.data import Dataset
from multiprocess import Pool
from itertools import repeat
from more_itertools import chunked, flatten
from math import ceil
from pathlib import Path
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageOps

from typing import Type

from lcmr.renderer.renderer2d import Renderer2D
from lcmr_ext.renderer.renderer2d import PyTorch3DRenderer2D
from lcmr_ext.utils import optimize_params

font_file = "fonts-DSEG_v050b1/DSEG7-Classic-MINI/DSEG7ClassicMini-Bold.ttf"
if not Path(font_file).is_file():
    import urllib.request
    import zipfile

    zip_name = "fonts-DSEG_v050b1.zip"
    urllib.request.urlretrieve("https://github.com/keshikan/DSEG/releases/download/v0.50beta1/fonts-DSEG_v050b1.zip", zip_name)
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(zip_name.split(".")[0])
    Path(zip_name).unlink()

# TODO: class for data entry


def random_image_7seg(background_color, num_characters, raster_size, font):
    w, h = raster_size
    img = Image.new("RGB", raster_size, background_color)

    for _ in range(num_characters):
        color = int((random.random() * 0.8 + 0.2) * 255), int((random.random() * 0.8 + 0.2) * 255), int((random.random() * 0.8 + 0.2) * 255)
        text = random.choice("8")

        txt = Image.new("L", font.getbbox(text)[2:4])
        d = ImageDraw.Draw(txt)
        d.text((0, 0), text, font=font, fill=255)
        txt = txt.rotate(random.random() * 365, expand=1, resample=Image.Resampling.BICUBIC)
        pos = int(random.random() * (w - txt.width)), int(random.random() * (h - txt.height))
        img.paste(ImageOps.colorize(txt, (0, 0, 0), color), pos, txt)

    return torch.from_numpy(np.array(img, dtype=np.float32) / 255)


def scene_from_blobs(images, raster_size, background_color):
    import torch
    from random import randint
    from .scene_from_blobs import SceneFromBlobs

    device = torch.device(f"cuda:{randint(0, torch.cuda.device_count() - 1)}") if torch.cuda.is_available() else torch.device("cpu")
    scene_from_blobs = SceneFromBlobs(raster_size=raster_size, background_color=background_color, device=device)
    scenes = [scene_from_blobs.predict(img.to(device), num_blobs=7).cpu() for img in images]
    return scenes


class Dataset7Seg(Dataset):
    def __init__(
        self,
        data_len: int,
        raster_size: tuple[int, int] = (128, 128),
        background_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        pool_size: int = 8,
        seed: int = 123,
        device: torch.device = torch.device("cpu"),
        Renderer: Type[Renderer2D] = PyTorch3DRenderer2D,
        use_cache: bool = True,
    ):
        filename_file = f"Dataset7Seg-len-{data_len}-seed-{seed}-raster_size-{'-'.join(map(str, raster_size))}-background_color-{'-'.join(map(str, background_color.tolist()))}.pickle"

        if use_cache and Path(filename_file).is_file():
            with open(filename_file, "rb") as handle:
                self.data = pickle.load(handle)
        else:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            font_size = min(*raster_size) // 2
            font = ImageFont.truetype(font_file, font_size)

            images = [
                random_image_7seg(tuple((background_color[0:3].cpu().numpy() * 255).astype(np.uint8).tolist()), 1, raster_size, font)[None, ...]
                for _ in range(data_len)
            ]

            pool_size = min(pool_size, data_len // 2)

            with Pool(pool_size) as p:
                scenes = flatten(
                    p.starmap(
                        scene_from_blobs,
                        zip(chunked(images, ceil(data_len / pool_size)), repeat(raster_size, pool_size), repeat(background_color, pool_size)),
                    )
                )

            optimized_scenes = []
            batch_len = 128
            renderer = Renderer(raster_size=raster_size, background_color=background_color, device=device)

            for img, scene in zip(chunked(images, batch_len), chunked(scenes, batch_len)):
                img = torch.cat(img, dim=0).to(device)
                scene = torch.cat(scene, dim=0).to(device)

                t = scene.layer.object.transformation.translation
                s = scene.layer.object.transformation.scale
                a = scene.layer.object.transformation.angle
                c = scene.layer.object.appearance.color
                optimize_params(scene, img, renderer, [t, s, a, c])

                scene = [s.cpu()[None, ...] for s in scene]
                optimized_scenes.extend(scene)

            scenes = optimized_scenes
            self.data = list(zip(images, scenes))

            if use_cache:
                with open(filename_file, "wb") as handle:
                    pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
