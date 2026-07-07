import os
import random
from functools import wraps

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """
    Set global seed for Python, NumPy, PyTorch (CPU + CUDA).
    When *deterministic* is True (default), also disables CuDNN autotune
    so that convolution algorithms are fully repeatable.

    Returns the seed so callers can store it in experiment metadata.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

    return seed


def seed_worker(worker_id: int) -> None:
    """
    Worker initialisation function for ``torch.utils.data.DataLoader``.

    Each worker gets an independent seed derived from the main process
    seed so that data shuffling / augmentations are deterministic across
    runs while remaining different across workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def reproducible_dataloader(
    dataset,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    generator: torch.Generator = None,
    **kwargs,
) -> DataLoader:
    """
    Build a ``DataLoader`` whose workers are seeded deterministically.

    Usage
    -----
    >>> from utils import set_seed, reproducible_dataloader
    >>> set_seed(42)
    >>> loader = reproducible_dataloader(train_dataset, batch_size=32)
    >>> for epoch in range(10):
    ...     for x, y in loader:
    ...         ...

    The worker seed is derived from the generator state so that shuffling
    order is identical every time ``set_seed(42)`` is called.
    """
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(torch.initial_seed())

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Decorator  — wrap any training function with global seed + deterministic
# DataLoader behaviour.
# ---------------------------------------------------------------------------

def with_seed(seed: int = 42):
    """
    Decorator that applies :func:`set_seed` before the wrapped function
    runs and ensures all ``DataLoader`` instances created inside use the
    deterministic worker init.

    Example
    -------
    >>> @with_seed(seed=42)
    ... def train(config):
    ...     loader = DataLoader(ds, ...)   # worker seeding automatic
    ...     for epoch in range(10):
    ...         ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            set_seed(seed)
            _patch_dataloader()
            try:
                return fn(*args, **kwargs)
            finally:
                _unpatch_dataloader()
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Internal helpers  — monkey-patch DataLoader so vanilla calls get the
# deterministic worker_init_fn automatically.
# ---------------------------------------------------------------------------

_ORIGINAL_DATALOADER_INIT = None


def _patch_dataloader():
    global _ORIGINAL_DATALOADER_INIT
    if _ORIGINAL_DATALOADER_INIT is not None:
        return  # already patched

    _ORIGINAL_DATALOADER_INIT = DataLoader.__init__

    def _patched_init(self, dataset, **kwargs):
        kwargs.setdefault("worker_init_fn", seed_worker)
        if "generator" not in kwargs:
            gen = torch.Generator()
            gen.manual_seed(torch.initial_seed())
            kwargs["generator"] = gen
        _ORIGINAL_DATALOADER_INIT(self, dataset, **kwargs)

    DataLoader.__init__ = _patched_init


def _unpatch_dataloader():
    global _ORIGINAL_DATALOADER_INIT
    if _ORIGINAL_DATALOADER_INIT is not None:
        DataLoader.__init__ = _ORIGINAL_DATALOADER_INIT
        _ORIGINAL_DATALOADER_INIT = None
