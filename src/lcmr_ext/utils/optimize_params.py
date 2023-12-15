import torch
from typing import Sequence
from torchtyping import TensorType
from tqdm import tqdm

from lcmr.grammar import Scene
from lcmr.renderer.renderer2d import Renderer2D
from lcmr.utils.guards import typechecked, batch_dim, height_dim, width_dim
from lcmr.utils.presentation import display_img

# TODO allow any loss


@typechecked
def optimize_params(
    scene: Scene,
    target: TensorType[batch_dim, height_dim, width_dim, 3, torch.float32],
    renderer: Renderer2D,
    params: Sequence[torch.Tensor],
    epochs: int = 241,
    lr: int = 0.005,
    show_progress: bool = False,
    show_interval: int = 20,
):
    requires_grad_list = []
    for param in params:
        requires_grad_list.append(param.requires_grad)
        param.requires_grad = True

    optimizer = torch.optim.Adam(params, lr=lr)

    it = range(epochs)
    if show_progress:
        it = tqdm(it)

    for epoch in it:
        optimizer.zero_grad()
        pred = renderer.render(scene)[..., :3]
        loss = (pred - target).pow(2).mean()
        loss.backward()
        optimizer.step()

        if show_progress:
            with torch.no_grad():
                it.set_description(f"loss: {loss.detach().cpu().item():.4f}")
                if show_interval > 0 and epoch % show_interval == 0:
                    display_img(pred[0])

    for param, requires_grad in zip(params, requires_grad_list):
        param.requires_grad = requires_grad
