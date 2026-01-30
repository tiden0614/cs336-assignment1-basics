import torch
from torch import Tensor
import numpy as np
from numpy import typing as npt
import random
import os
import typing


def get_batch(
    dataset: npt.NDArray, device: str, batch_size: int, context_length: int
) -> tuple[Tensor, Tensor]:
    assert (
        dataset.size >= context_length + 1
    ), f"dataset too small ({dataset.size}) vs context_length ({context_length})"

    sequences = np.empty((batch_size, context_length), dtype=int)
    targets = np.empty((batch_size, context_length), dtype=int)
    for batch in range(batch_size):
        starting = random.randint(0, dataset.size - context_length - 1)
        sequences[batch][:] = dataset[starting : starting + context_length]
        targets[batch][:] = dataset[starting + 1 : starting + context_length + 1]

    sequences_tensor = torch.from_numpy(sequences)
    targets_tensor = torch.from_numpy(targets)

    return (sequences_tensor.to(device=device), targets_tensor.to(device=device))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    obj = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "iter": iteration,
    }
    torch.save(obj, f=out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(f=src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optim"])
    return obj["iter"]
