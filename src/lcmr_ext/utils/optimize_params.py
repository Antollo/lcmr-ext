import torch
from typing import Sequence, Type
from tqdm import tqdm

from lcmr.grammar import Scene
from lcmr.renderer.renderer2d import Renderer2D
from lcmr.utils.guards import typechecked, ImageBHWC3
from lcmr.utils.presentation import display_img, make_img_grid

# TODO allow any loss


@typechecked
def optimize_params(
    scene: Scene,
    target: ImageBHWC3,
    renderer: Renderer2D,
    params: Sequence[torch.Tensor],
    epochs: int = 241,
    lr: float = 0.005,
    show_progress: bool = False,
    show_interval: int = 20,
    Optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
    Loss: Type[torch.nn.Module] = torch.nn.MSELoss
):
    requires_grad_list = []
    for param in params:
        requires_grad_list.append(param.requires_grad)
        param.requires_grad = True

    optimizer = Optimizer(params, lr=lr)
    loss_func = Loss().to(scene.device)

    it = range(epochs)
    if show_progress:
        it = tqdm(it)

    for epoch in it:
        optimizer.zero_grad()
        pred = renderer.render(scene)[..., :3]
        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()

        if show_progress:
            with torch.no_grad():
                it.set_description(f"loss: {loss.detach().cpu().item():.4f}")
                if show_interval > 0 and epoch % show_interval == 0:
                    img_grid = make_img_grid((pred[:8], target[:8]), nrow=max(len(pred[:8]), len(target[:8])))
                    display_img(img_grid)

    for param, requires_grad in zip(params, requires_grad_list):
        param.requires_grad = requires_grad
