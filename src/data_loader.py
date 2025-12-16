"""MNIST data loading utilities for ES experiments."""

from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

__all__ = ["get_mnist_dataloaders"]


def get_mnist_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare train and test loaders for MNIST with fast input pipelines.

    Args:
        data_dir: Directory to store/download MNIST.
        batch_size: Number of samples per batch.
        num_workers: Number of dataloader workers (0 uses main process).
        download: Whether to download the dataset if missing.
        normalize: Apply standard MNIST normalization.
    """
    transform_steps = [transforms.ToTensor()]
    if normalize:
        transform_steps.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(transform_steps)

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=download, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=download, transform=transform
    )

    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, test_loader

