#!/usr/bin/env python3
"""
Local experiment runner for FL participation × non-IID study.
Runs all 48 configurations and saves results incrementally.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
import os, json, csv, time

# ── Configuration ────────────────────────────────────────────────────────
ALPHAS         = [0.05, 0.1, 0.5, 1.0]
PART_RATES     = [0.2, 0.5, 0.8, 1.0]
ALGORITHMS     = ['fedavg', 'fedprox', 'scaffold']
NUM_CLIENTS    = 10
NUM_ROUNDS     = 30
LOCAL_EPOCHS   = 3
BATCH_SIZE     = 64
LR             = 0.01
DEVICE         = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR    = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR    = os.path.join(SCRIPT_DIR, '..', 'paper', 'figures')
RESULTS_FILE   = os.path.join(RESULTS_DIR, 'all_results.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Total experiments: {len(ALPHAS)*len(PART_RATES)*len(ALGORITHMS)}")

# ── Model ────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
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
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, weights):
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def evaluate_model(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss_sum += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total, loss_sum / len(loader)

# ── Data Partitioning ────────────────────────────────────────────────────
def dirichlet_partition(dataset, num_clients, alpha, seed=42):
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

# ── FL Algorithms ────────────────────────────────────────────────────────
def local_train_fedavg(model, loader, epochs, lr):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()
    return get_weights(model)

def local_train_fedprox(model, loader, epochs, lr, global_weights, mu=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    global_params = [torch.tensor(w).to(DEVICE) for w in global_weights]
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            prox = sum(((p - gp) ** 2).sum()
                       for p, gp in zip(model.parameters(), global_params))
            (loss + (mu / 2) * prox).backward()
            optimizer.step()
    return get_weights(model)

def local_train_scaffold(model, loader, epochs, lr,
                         global_weights, c_global, c_local):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    c_g = [torch.tensor(c).to(DEVICE) for c in c_global]
    c_l = [torch.tensor(c).to(DEVICE) for c in c_local]
    K = epochs * len(loader)
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            with torch.no_grad():
                for param, cg, cl in zip(model.parameters(), c_g, c_l):
                    if param.grad is not None:
                        param.grad += cg - cl
            optimizer.step()
    new_weights = get_weights(model)
    new_c_local = [
        cl.cpu().numpy() - cg.cpu().numpy() + (gw - nw) / (K * lr)
        for cl, cg, gw, nw in zip(c_l, c_g, global_weights, new_weights)
    ]
    delta_c = [nc - oc.cpu().numpy() for nc, oc in zip(new_c_local, c_l)]
    return new_weights, new_c_local, delta_c

# ── Load Data ────────────────────────────────────────────────────────────
print("Loading MNIST...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
data_dir = os.path.join(SCRIPT_DIR, 'data')
trainset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=256, shuffle=False)
print("Data loaded.")

# ── Experiment Runner ────────────────────────────────────────────────────
def run_experiment(algorithm, alpha, participation_rate):
    exp_id = f"{algorithm}_alpha{alpha}_part{participation_rate}"

    if os.path.exists(RESULTS_FILE):
        existing = pd.read_csv(RESULTS_FILE)
        if exp_id in existing['exp_id'].values:
            print(f"  SKIP (done): {exp_id}")
            return None

    t0 = time.time()
    print(f"  RUNNING: {exp_id}")

    client_data_indices = dirichlet_partition(trainset, NUM_CLIENTS, alpha)
    client_loaders = [
        DataLoader(Subset(trainset, indices), batch_size=BATCH_SIZE, shuffle=True)
        for indices in client_data_indices
    ]

    global_model = SimpleCNN().to(DEVICE)
    global_weights = get_weights(global_model)
    c_global = [np.zeros_like(w) for w in global_weights]
    c_locals = [[np.zeros_like(w) for w in global_weights] for _ in range(NUM_CLIENTS)]
    num_selected = max(1, int(NUM_CLIENTS * participation_rate))

    round_results = []
    rounds_to_80 = None

    for rnd in range(NUM_ROUNDS):
        selected = np.random.choice(NUM_CLIENTS, num_selected, replace=False)
        local_weights_list = []

        for cid in selected:
            local_model = SimpleCNN().to(DEVICE)
            set_weights(local_model, global_weights)

            if algorithm == 'fedavg':
                lw = local_train_fedavg(local_model, client_loaders[cid],
                                        LOCAL_EPOCHS, LR)
                local_weights_list.append(lw)
            elif algorithm == 'fedprox':
                lw = local_train_fedprox(local_model, client_loaders[cid],
                                         LOCAL_EPOCHS, LR, global_weights)
                local_weights_list.append(lw)
            elif algorithm == 'scaffold':
                lw, new_cl, delta_c = local_train_scaffold(
                    local_model, client_loaders[cid], LOCAL_EPOCHS, LR,
                    global_weights, c_global, c_locals[cid])
                local_weights_list.append(lw)
                c_locals[cid] = new_cl
                c_global = [cg + (dc / NUM_CLIENTS)
                            for cg, dc in zip(c_global, delta_c)]

        sizes = [len(client_data_indices[i]) for i in selected]
        total_size = sum(sizes)
        new_weights = [
            sum(w[layer] * s / total_size
                for w, s in zip(local_weights_list, sizes))
            for layer in range(len(global_weights))
        ]
        global_weights = new_weights
        set_weights(global_model, global_weights)

        if (rnd + 1) % 3 == 0 or rnd == 0:
            acc, loss = evaluate_model(global_model, testloader)
            if acc >= 0.80 and rounds_to_80 is None:
                rounds_to_80 = rnd + 1
            round_results.append({
                'round': rnd + 1, 'accuracy': acc, 'loss': loss
            })
            print(f"    Round {rnd+1:3d}: acc={acc:.4f}, loss={loss:.4f}")

    per_client_accs = []
    for cid in range(NUM_CLIENTS):
        local_model = SimpleCNN().to(DEVICE)
        set_weights(local_model, global_weights)
        acc, _ = evaluate_model(local_model,
                                DataLoader(Subset(trainset,
                                                  client_data_indices[cid]),
                                           batch_size=256))
        per_client_accs.append(acc)

    final_acc = round_results[-1]['accuracy']
    elapsed = time.time() - t0

    row = {
        'exp_id': exp_id,
        'algorithm': algorithm,
        'alpha': alpha,
        'participation_rate': participation_rate,
        'final_accuracy': final_acc,
        'rounds_to_80pct': rounds_to_80 if rounds_to_80 else 999,
        'per_client_acc_mean': np.mean(per_client_accs),
        'per_client_acc_std': np.std(per_client_accs),
        'per_client_acc_min': np.min(per_client_accs),
        'round_results': json.dumps(round_results)
    }

    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"  DONE: {exp_id} | acc={final_acc:.4f} | {elapsed:.0f}s")
    return row

# ── Main Loop ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    total_start = time.time()
    completed = 0

    for alg in ALGORITHMS:
        for alpha in ALPHAS:
            for part in PART_RATES:
                result = run_experiment(alg, alpha, part)
                if result:
                    completed += 1

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Completed {completed} new experiments in {total_elapsed/60:.1f} minutes")
    print(f"Results: {RESULTS_FILE}")
