"""
Federated learning algorithm implementations:
  - FedAvg  (McMahan et al., 2017)
  - FedProx (Li et al., 2020)
  - SCAFFOLD (Karimireddy et al., 2020)

Each function performs local training on a single client and returns
updated weights (plus control variate deltas for SCAFFOLD).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import get_weights


# ──────────────────────────────────────────────────────────────────────────
# FedAvg
# ──────────────────────────────────────────────────────────────────────────

def local_train_fedavg(model, loader, epochs, lr, device):
    """
    Standard FedAvg local training.
    Each client runs SGD on its local data for `epochs` passes.
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()
    return get_weights(model)


# ──────────────────────────────────────────────────────────────────────────
# FedProx
# ──────────────────────────────────────────────────────────────────────────

def local_train_fedprox(model, loader, epochs, lr, global_weights,
                        device, mu=0.01):
    """
    FedProx local training with proximal regularization.
    Adds a penalty term mu/2 * ||w - w_global||^2 to the local loss,
    which limits how far local models can drift from the global model.
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    global_params = [torch.tensor(w).to(device) for w in global_weights]

    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            # Proximal term: penalize deviation from global model
            prox = sum(
                ((p - gp) ** 2).sum()
                for p, gp in zip(model.parameters(), global_params)
            )
            (loss + (mu / 2) * prox).backward()
            optimizer.step()
    return get_weights(model)


# ──────────────────────────────────────────────────────────────────────────
# SCAFFOLD
# ──────────────────────────────────────────────────────────────────────────

def local_train_scaffold(model, loader, epochs, lr,
                         global_weights, c_global, c_local, device):
    """
    SCAFFOLD local training with variance reduction via control variates.

    The gradient update is corrected by (c_global - c_local) to compensate
    for client drift caused by heterogeneous data distributions.

    Parameters
    ----------
    c_global : list[np.ndarray]
        Server-level control variate
    c_local : list[np.ndarray]
        This client's control variate from previous round

    Returns
    -------
    new_weights : list[np.ndarray]
    new_c_local : list[np.ndarray]
        Updated client control variate
    delta_c : list[np.ndarray]
        Change in control variate (for server aggregation)
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    c_g = [torch.tensor(c).to(device) for c in c_global]
    c_l = [torch.tensor(c).to(device) for c in c_local]

    K = epochs * len(loader)  # total local gradient steps

    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            # SCAFFOLD correction: shift gradient by (c_global - c_local)
            with torch.no_grad():
                for param, cg, cl in zip(model.parameters(), c_g, c_l):
                    if param.grad is not None:
                        param.grad += cg - cl
            optimizer.step()

    new_weights = get_weights(model)

    # Option II control variate update from Karimireddy et al.
    new_c_local = [
        cl.cpu().numpy() - cg.cpu().numpy() + (gw - nw) / (K * lr)
        for cl, cg, gw, nw in zip(c_l, c_g, global_weights, new_weights)
    ]
    delta_c = [nc - oc.cpu().numpy() for nc, oc in zip(new_c_local, c_l)]

    return new_weights, new_c_local, delta_c
