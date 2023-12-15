import torch


def collate_fn(batch):
    if type(batch[0]) is tuple:
        batch = [list(x) for x in zip(*batch)]
    return [torch.cat(x, dim=0).pin_memory() for x in batch]
