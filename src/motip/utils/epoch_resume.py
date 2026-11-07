"""Optional per-rank RNG snapshots for epoch-boundary SLURM continuation.

Optimizer/scheduler/model remain in the normal MOTIP epoch checkpoint. These
sidecars are written *after* evaluation and scheduler.step(), before creating
the next epoch's DataLoader iterator. Default training behaviour is unchanged.
"""
from pathlib import Path
import random

import numpy as np
import torch


def save_epoch_rng(root, rank, world_size, states):
    target = Path(root) / f"rng_epoch_{states['start_epoch'] - 1}_rank_{rank}.pt"
    blob = dict(rank=rank, world_size=world_size, states=dict(states),
                python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
    temporary = target.with_suffix(".pt.tmp")
    torch.save(blob, temporary)
    temporary.replace(target)


def load_epoch_rng(root, rank, world_size, states):
    target = Path(root) / f"rng_epoch_{states['start_epoch'] - 1}_rank_{rank}.pt"
    blob = torch.load(target, map_location="cpu", weights_only=False)
    if (blob["rank"], blob["world_size"], blob["states"]) != (rank, world_size, states):
        raise ValueError("RNG sidecar does not match checkpoint epoch, step or DDP topology")
    random.setstate(blob["python"])
    np.random.set_state(blob["numpy"])
    torch.set_rng_state(blob["torch"])
    if blob["cuda"] is not None:
        torch.cuda.set_rng_state(blob["cuda"])
