import torch
import numpy as np
from lcmr.utils.guards import typechecked

# Adapted from https://stackoverflow.com/a/78761664
@typechecked
class EarlyStopping:
    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.state_dict = None

    def __call__(self, validation_loss: float, module: torch.nn.Module) -> bool:
        if (validation_loss + self.min_delta) < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.state_dict = module.state_dict()
        elif (validation_loss + self.min_delta) > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
