import torch
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageOps

from lcmr_ext.dataset.dataset_base import DatasetBase
from lcmr_ext.dataset.dataset_options import DatasetOptions

font_file = "fonts-DSEG_v050b1/DSEG7-Classic-MINI/DSEG7ClassicMini-Bold.ttf"
if not Path(font_file).is_file():
    import urllib.request
    import zipfile

    zip_name = "fonts-DSEG_v050b1.zip"
    urllib.request.urlretrieve("https://github.com/keshikan/DSEG/releases/download/v0.50beta1/fonts-DSEG_v050b1.zip", zip_name)
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(zip_name.split(".")[0])
    Path(zip_name).unlink()


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


class Dataset7Seg(DatasetBase):
    def __init__(self, options: DatasetOptions):
        options.cache_filename = f"{type(self).__name__}-len-{options.data_len}-seed-{options.seed}-raster_size-{'-'.join(map(str, options.raster_size))}-background_color-{'-'.join(map(str, options.background_color.tolist()))}.pickle"

        super().__init__(options)

    def get_images(self):
        self.set_seed()

        font_size = min(*self.options.raster_size) // 2
        font = ImageFont.truetype(font_file, font_size)

        images = [
            random_image_7seg(tuple((self.options.background_color[0:3].cpu().numpy() * 255).astype(np.uint8).tolist()), 1, self.options.raster_size, font)
            for _ in range(self.options.data_len)
        ]

        return images
