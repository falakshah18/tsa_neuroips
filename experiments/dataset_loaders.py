# experiments/dataset_loaders.py
"""
Centralized dataset loaders for all neuromorphic benchmarks.

Each get_*_loaders() function returns (train_loader, val_loader, test_loader)
and follows the same conventions as BaselineComparison._get_data_loaders():
  - 90/10 train/val split
  - optional quick-mode subsampling (320/128/128)
  - DataLoader(num_workers=4, pin_memory=True)
"""

from typing import Dict, Optional, Tuple

from torch.utils.data import DataLoader, random_split, Subset
from tonic import datasets, transforms


# ---------------------------------------------------------------------------
# Dataset metadata consumed by model builders and config loaders
# ---------------------------------------------------------------------------
DATASET_CONFIGS: Dict[str, dict] = {
    'nmnist': {
        'sensor_size': (34, 34, 2),
        'num_classes': 10,
        'img_size': 34,
        'patch_size': 2,
        'in_channels': 2,
        'T': 20,
        'n_time_bins': 20,
        'batch_size': 32,
    },
    'shd': {
        'sensor_size': (700, 1, 1),
        'num_classes': 20,
        'img_size': 700,
        'patch_size': 10,
        'in_channels': 2,
        'T': 100,
        'n_time_bins': 100,
        'batch_size': 32,
    },
    'dvs_gesture': {
        'sensor_size': (128, 128, 2),
        'num_classes': 11,
        'img_size': 34,          # downsampled from 128 for TSA compatibility
        'patch_size': 2,
        'in_channels': 2,
        'T': 20,
        'n_time_bins': 20,
        'batch_size': 32,
    },
    'cifar10_dvs': {
        'sensor_size': (128, 128, 2),
        'num_classes': 10,
        'img_size': 34,          # downsampled from 128 for TSA compatibility
        'patch_size': 2,
        'in_channels': 2,
        'T': 20,
        'n_time_bins': 20,
        'batch_size': 32,
    },
}


def _subsample(dataset, n: int):
    """Truncate a dataset to at most *n* samples (fixed prefix, deterministic)."""
    n = min(n, len(dataset))
    return Subset(dataset, list(range(n)))


def _make_loaders(
    train_ds,
    test_ds,
    batch_size: int,
    quick: bool,
    train_quick: int = 320,
    val_quick: int = 128,
    test_quick: int = 128,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Shared train/val split + DataLoader creation."""
    train_size = int(0.9 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    if quick:
        train_ds = _subsample(train_ds, train_quick)
        val_ds = _subsample(val_ds, val_quick)
        test_ds = _subsample(test_ds, test_quick)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# DVS-Gesture
# ---------------------------------------------------------------------------
def get_dvs_gesture_loaders(
    batch_size: int = 32,
    n_time_bins: int = 20,
    quick: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the DVSGesture dataset (128×128, 11 classes).

    Events are spatially downsampled from 128×128 → 34×34 so the resulting
    frames match the default TSA / N-MNIST spatial dimensions.
    """
    sensor_size = (128, 128, 2)
    target_size = (34, 34, 2)

    transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.Downsample(spatial_factor=34 / 128),
        transforms.ToFrame(
            sensor_size=target_size,
            n_time_bins=n_time_bins,
        ),
    ])

    train_ds = datasets.DVSGesture(
        save_to='./data', train=True, transform=transform,
    )
    test_ds = datasets.DVSGesture(
        save_to='./data', train=False, transform=transform,
    )

    return _make_loaders(train_ds, test_ds, batch_size, quick)


# ---------------------------------------------------------------------------
# CIFAR10-DVS
# ---------------------------------------------------------------------------
def get_cifar10_dvs_loaders(
    batch_size: int = 32,
    n_time_bins: int = 20,
    quick: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the CIFAR10-DVS dataset (128×128, 10 classes).

    Events are spatially downsampled from 128×128 → 34×34 so the resulting
    frames match the default TSA / N-MNIST spatial dimensions.
    """
    sensor_size = (128, 128, 2)
    target_size = (34, 34, 2)

    transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.Downsample(spatial_factor=34 / 128),
        transforms.ToFrame(
            sensor_size=target_size,
            n_time_bins=n_time_bins,
        ),
    ])

    train_ds = datasets.CIFAR10DVS(
        save_to='./data', train=True, transform=transform,
    )
    test_ds = datasets.CIFAR10DVS(
        save_to='./data', train=False, transform=transform,
    )

    return _make_loaders(train_ds, test_ds, batch_size, quick)


# ---------------------------------------------------------------------------
# N-MNIST (re-exported for convenience / unified API)
# ---------------------------------------------------------------------------
def get_nmnist_loaders(
    batch_size: int = 32,
    n_time_bins: int = 20,
    quick: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load the N-MNIST dataset (34×34, 10 classes)."""
    sensor_size = (34, 34, 2)

    transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(
            sensor_size=sensor_size,
            n_time_bins=n_time_bins,
        ),
    ])

    train_ds = datasets.NMNIST(
        save_to='./data', train=True, transform=transform,
    )
    test_ds = datasets.NMNIST(
        save_to='./data', train=False, transform=transform,
    )

    return _make_loaders(train_ds, test_ds, batch_size, quick)


# ---------------------------------------------------------------------------
# SHD (re-exported for convenience / unified API)
# ---------------------------------------------------------------------------
def get_shd_loaders(
    batch_size: int = 32,
    n_time_bins: int = 100,
    quick: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load the Spiking Heidelberg Digits dataset (700 bins, 20 classes)."""
    sensor_size = (700, 1, 1)

    transform = transforms.Compose([
        transforms.ToFrame(
            sensor_size=sensor_size,
            n_time_bins=n_time_bins,
        ),
        lambda frame: frame.reshape(frame.shape[0], -1),
    ])

    train_ds = datasets.SHD(
        save_to='./data', train=True, transform=transform,
    )
    test_ds = datasets.SHD(
        save_to='./data', train=False, transform=transform,
    )

    return _make_loaders(train_ds, test_ds, batch_size, quick)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
_LOADER_FNS = {
    'nmnist': get_nmnist_loaders,
    'shd': get_shd_loaders,
    'dvs_gesture': get_dvs_gesture_loaders,
    'cifar10_dvs': get_cifar10_dvs_loaders,
}


def get_all_dataset_loaders(
    dataset_name: str,
    batch_size: int = 32,
    n_time_bins: Optional[int] = None,
    quick: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Dispatch to the correct loader based on *dataset_name*.

    Parameters
    ----------
    dataset_name : str
        One of 'nmnist', 'shd', 'dvs_gesture', 'cifar10_dvs'.
    batch_size : int
    n_time_bins : int, optional
        If None, uses the default from DATASET_CONFIGS.
    quick : bool
        If True, subsample to 320/128/128 for smoke tests.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    if dataset_name not in _LOADER_FNS:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Valid options: {sorted(_LOADER_FNS.keys())}"
        )

    if n_time_bins is None:
        n_time_bins = DATASET_CONFIGS[dataset_name]['n_time_bins']

    loader_fn = _LOADER_FNS[dataset_name]
    return loader_fn(
        batch_size=batch_size,
        n_time_bins=n_time_bins,
        quick=quick,
    )
