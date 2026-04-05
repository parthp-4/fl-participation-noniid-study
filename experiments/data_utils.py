"""
Data utilities for federated learning experiments.
Implements Dirichlet-based non-IID data partitioning across clients.
"""

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def dirichlet_partition(dataset, num_clients, alpha, seed=42):
    """
    Partition dataset across clients using Dirichlet(alpha) distribution.

    Lower alpha values produce more extreme non-IID splits:
      - alpha=0.05 : most clients see only 1-2 digit classes
      - alpha=0.1  : moderate skew, clients dominated by a few classes
      - alpha=0.5  : mild heterogeneity
      - alpha=1.0  : roughly uniform, approaching IID

    Parameters
    ----------
    dataset : torchvision.datasets
        Full training dataset
    num_clients : int
        Number of federated clients
    alpha : float
        Dirichlet concentration parameter
    seed : int
        Random seed for reproducibility

    Returns
    -------
    list[list[int]]
        Per-client lists of dataset indices
    """
    np.random.seed(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))

    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]

    for c_idx in class_indices:
        np.random.shuffle(c_idx)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(c_idx)).astype(int)
        proportions[-1] = len(c_idx) - proportions[:-1].sum()

        start = 0
        for client_id, count in enumerate(proportions):
            client_indices[client_id].extend(c_idx[start:start + count].tolist())
            start += count

    return client_indices


def load_mnist():
    """
    Load MNIST train and test sets with standard normalization.

    Returns
    -------
    trainset, testset, testloader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=256, shuffle=False)
    return trainset, testset, testloader


def build_client_loaders(trainset, client_indices, batch_size=64):
    """
    Build a DataLoader for each client's data partition.

    Parameters
    ----------
    trainset : Dataset
        Full training dataset
    client_indices : list[list[int]]
        Per-client index lists from dirichlet_partition
    batch_size : int

    Returns
    -------
    list[DataLoader]
    """
    return [
        DataLoader(Subset(trainset, indices), batch_size=batch_size, shuffle=True)
        for indices in client_indices
    ]
