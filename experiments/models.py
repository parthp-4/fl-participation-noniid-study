"""
SimpleCNN model for MNIST classification.
Legacy architecture — not used in the published experiments. See LiteCNN in run_fast.py.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A lightweight CNN for MNIST digit classification.
    Architecture: Conv(1->32) -> Conv(32->64) -> MaxPool -> FC(9216->128) -> FC(128->10)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def get_weights(model):
    """Extract model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, weights):
    """Load a list of numpy arrays into model parameters."""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def evaluate_model(model, loader, device):
    """
    Evaluate model on a DataLoader.
    Returns (accuracy, average_loss).
    """
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss_sum += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total, loss_sum / len(loader)
