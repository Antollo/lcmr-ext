import torch
import numpy as np
from torchtyping import TensorType
from skimage.transform import downscale_local_mean, rotate
from skimage.draw import disk
from skimage.feature import blob_log
from skimage.color import rgb2gray
from nptyping import NDArray, Shape
from functools import cache
from kornia.geometry.transform import resize


from lcmr.grammar import Scene, Layer
from lcmr.renderer import OpenGLRenderer2D
from lcmr.utils.guards import typechecked, ImageBHWC3, height_dim, width_dim


@cache
def rot_mat2(angle):
    angle = angle / 180 * np.pi
    M = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    return M


def rotate_points(point, origin, angle):
    return ((point - origin) @ rot_mat2(angle).astype(np.float32)) + origin


# for now it's ok
# if more speed needed rewrite using torch
@typechecked
class SceneFromBlobs:
    def __init__(
        self, raster_size: tuple[int, int], background_color: TensorType[4, torch.float32] = torch.zeros(4), device: torch.device = torch.device("cpu")
    ):
        self.center = np.array([[0.5, 0.5]], dtype=np.float32)
        self.raster_size = raster_size
        self.device = device
        self.renderer = OpenGLRenderer2D(raster_size, gamma_rgb=1.2, gamma_confidence=2.5, background_color=background_color, device=device)

    def predict(self, imgs: ImageBHWC3, num_blobs: int = 7):
        # param_groups = [self.single_img(img, num_blobs=num_blobs) for img in imgs]
        # param_groups = [torch.cat(p, dim=0) for p in zip(*param_groups)]
        # return param_groups
        if imgs.shape[1:2] != self.raster_size:
            imgs = resize(imgs.permute(0, 3, 1, 2), self.raster_size, interpolation="area").permute(0, 2, 3, 1)
        scenes = [self.single_img(img, num_blobs=num_blobs) for img in imgs]
        return torch.cat(scenes, dim=0)

    def single_img(self, img: TensorType[height_dim, width_dim, 3, torch.float32], num_blobs: int = 7):
        img_np = img.detach().cpu().numpy()

        # translation, scale, color, confidence, angle
        param_groups = [[], [], [], [], []]

        def append_new(params):
            for group, p in zip(param_groups, params):
                group.append(p)

        # maybe call 'single_img_inner' in loop

        params = self.single_img_inner(img_np, num_blobs=num_blobs * 2)
        if len(params[0]) > 0:
            append_new(params)

        params = self.single_img_inner(downscale_local_mean(img_np, (2, 1, 1)), num_blobs=num_blobs * 2)
        if len(params[0]) > 0:
            append_new(params)

        params = self.single_img_inner(downscale_local_mean(img_np, (1, 2, 1)), num_blobs=num_blobs * 2)
        if len(params[0]) > 0:
            append_new(params)

        params = self.single_img_inner(downscale_local_mean(rotate(img_np, 45), (2, 1, 1)), num_blobs=num_blobs * 2)
        if len(params[0]) > 0:
            params[0] = rotate_points(np.array(params[0]), self.center, -45)
            params[4] = np.array(params[4]) + 45 / 180 * np.pi
            append_new(params)

        params = self.single_img_inner(downscale_local_mean(rotate(img_np, -45), (2, 1, 1)), num_blobs=num_blobs * 2)
        if len(params[0]) > 0:
            params[0] = rotate_points(np.array(params[0]), self.center, 45)
            params[4] = np.array(params[4]) + -45 / 180 * np.pi
            append_new(params)

        for i, group in enumerate(param_groups):
            param_groups[i] = torch.from_numpy(np.concatenate(group))[None, None, ...]

        count = param_groups[0].shape[2]
        mask = torch.zeros((count,), dtype=torch.bool)

        translation, scale, color, confidence, angle = param_groups
        scene = Scene.from_tensors_sparse(translation=translation, scale=scale, color=color, confidence=confidence, angle=angle)

        for _ in range(num_blobs * 2):
            best_idx = -1
            best_mse = np.inf
            for idx in range(count):
                if mask[idx]:
                    continue
                mask[idx] = True
                scene_masked = Scene(
                    batch_size=[1, 1],
                    layer=Layer(batch_size=[1, 1], object=scene.layer.object[:, :, mask], scale=scene.layer.scale, composition=scene.layer.composition),
                )
                mask[idx] = False
                y = self.renderer.render(scene_masked)[0, ..., :3]
                mse = ((img - y) ** 2).mean()
                if best_mse > mse:
                    best_mse = mse
                    best_idx = idx
            if best_idx != -1:
                mask[best_idx] = True

        for _ in range(num_blobs):
            best_idx = -1
            best_mse = np.inf
            for idx in range(count):
                if not mask[idx]:
                    continue
                mask[idx] = False
                scene_masked = Scene(
                    batch_size=[1, 1],
                    layer=Layer(batch_size=[1, 1], object=scene.layer.object[:, :, mask], scale=scene.layer.scale, composition=scene.layer.composition),
                )
                mask[idx] = True
                x = self.renderer.render(scene_masked)[0, ..., :3]
                mse = ((img - x) ** 2).mean()
                if best_mse > mse:
                    best_mse = mse
                    best_idx = idx
            if best_idx != -1:
                mask[best_idx] = False

        translation, scale, color, confidence, angle = [group[:, :, mask] for group in param_groups]
        return Scene.from_tensors_sparse(translation=translation, scale=scale, color=color, confidence=confidence, angle=angle)

    def single_img_inner(self, img: NDArray[Shape["W, H, 3"], np.float32], num_blobs: int):
        img_gray = rgb2gray(img)
        width, height = img_gray.shape

        blobs_log = blob_log(img_gray, max_sigma=8, threshold=0.1, overlap=0.2)
        blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
        blobs_log = sorted(blobs_log, key=lambda tup: tup[2], reverse=True)[:num_blobs]  # sort by radius

        translation = []
        scale = []
        color = []
        confidence = []
        angle = []

        for blob in blobs_log:
            x, y, r = blob
            # r += 0.05
            rr, cc = disk((x, y), r, shape=img_gray.shape)
            c = img[rr, cc].mean(axis=0)
            translation.append((y / height, x / width))
            scale.append((r / height, r / width))
            angle.append((0.0,))
            color.append(c)
            confidence.append((1.0,))

        return [
            np.array(translation, dtype=np.float32),
            np.array(scale, dtype=np.float32),
            np.array(color, dtype=np.float32),
            np.array(confidence, dtype=np.float32),
            np.array(angle, dtype=np.float32),
        ]
