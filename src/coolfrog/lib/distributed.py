"""Data-parallel training over the GPUs of one host, one process per GPU, started by `torchrun`.

- Without `torchrun` (no `WORLD_SIZE`) everything runs as a single process: rank 0 of 1, collectives are no-ops
- Rank 0 is the main process: it alone writes run files, logs and shows the dashboard

Examples
--------
>>> # torchrun --standalone --nproc_per_node 4 -m coolfrog.experiments.finetune_autoencoder
>>> init_distributed()
>>> loss.backward()
>>> average_gradients(model.trainable_parameters().values())
"""

from collections.abc import Iterable
from datetime import timedelta
import os

import torch
from torch import distributed, nn

# rank 0 may hold the others at a collective for long: the uncommitted-changes prompt, saving a stage, waiting on `build.lock`
COLLECTIVE_TIMEOUT = timedelta(hours=6)


def init_distributed() -> None:
    """Join the process group of `torchrun` on this process's GPU, nothing without `torchrun`."""
    if int(os.environ.get("WORLD_SIZE", "1")) == 1 or distributed.is_initialized():
        return
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    distributed.init_process_group("nccl", timeout=COLLECTIVE_TIMEOUT, device_id=torch.cuda.current_device())


def rank() -> int:
    return distributed.get_rank() if distributed.is_initialized() else 0


def world_size() -> int:
    return distributed.get_world_size() if distributed.is_initialized() else 1


def is_main_process() -> bool:
    return rank() == 0


def barrier() -> None:
    """Wait for every process, nothing in a single process."""
    if distributed.is_initialized():
        distributed.barrier()


def broadcast_parameters(parameters: Iterable[nn.Parameter]) -> None:
    """Copy rank 0's values into every process, e.g. the randomly initialized LoRA matrices."""
    if distributed.is_initialized():
        for parameter in parameters:
            distributed.broadcast(parameter.data, src=0)


def average_gradients(parameters: Iterable[nn.Parameter]) -> None:
    """Replace every gradient by its mean over the processes, in one flat all-reduce."""
    if not distributed.is_initialized():
        return
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    flat = torch.cat([gradient.flatten() for gradient in gradients])
    distributed.all_reduce(flat)
    flat /= world_size()
    for gradient, reduced in zip(gradients, flat.split([gradient.numel() for gradient in gradients]), strict=True):
        gradient.copy_(reduced.view_as(gradient))


def all_gather_objects[T](local: T) -> list[T]:
    """Every process's `local`, in rank order, picklable objects such as polars tables."""
    if not distributed.is_initialized():
        return [local]
    gathered: list[T] = [local] * world_size()
    distributed.all_gather_object(gathered, local)
    return gathered
