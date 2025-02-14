from typing import Sequence, Type

import torch
from lcmr.grammar.scene_data import SceneData
from lcmr.renderer.renderer2d import Renderer2D
from lcmr.utils.guards import typechecked
from lcmr.utils.presentation import display_img, make_img_grid
from tqdm import tqdm

from lcmr_ext.loss import BaseLoss, ImageMseLoss


@typechecked
def optimize_params(
    scene: SceneData,
    target: SceneData,
    renderer: Renderer2D,
    params: Sequence[torch.Tensor],
    epochs: int = 100,
    lr: float = 0.01,
    show_progress: bool = False,
    show_interval: int = 20,
    Optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
    loss_func: BaseLoss = ImageMseLoss(),
    clip_grad_func=lambda *_: None,
    img_collect_func=lambda *_: None,
):
    requires_grad_list = []
    for param in params:
        requires_grad_list.append(param.requires_grad)
        param.requires_grad = True

    optimizer = Optimizer(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5, cooldown=10, verbose=False)

    it = range(epochs)
    if show_progress:
        it = tqdm(it)

    for epoch in it:
        optimizer.zero_grad()
        pred = renderer.render(scene.scene)
        loss = loss_func(target, pred)
        loss.backward()
        clip_grad_func()
        optimizer.step()

        scheduler.step(loss.detach())

        if show_progress:
            with torch.no_grad():
                it.set_description(f"loss: {loss.detach().cpu().item():.4f}")
                if show_interval > 0 and epoch % show_interval == 0:
                    img_grid = make_img_grid((pred[:8].image, target[:8].image), nrow=max(len(pred[:8]), len(target[:8])))
                    display_img(img_grid)
                    img_collect_func(img_grid)

    for param, requires_grad in zip(params, requires_grad_list):
        param.requires_grad = requires_grad
