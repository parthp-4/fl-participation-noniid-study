#!/usr/bin/env python3
"""
Multi-seed validation for the model collapse point (α=0.1, p=0.2).
Runs all 3 algorithms at seeds 43, 44, 45 to confirm collapse is not
an artifact of seed 42.
"""
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as transforms
import numpy as np, json, time, sys, os

sys.stdout.reconfigure(line_buffering=True)

# ── Config ───────────────────────────────────────────────────────────────
ALPHA        = 0.1
PART_RATE    = 0.2
SEEDS        = [43, 44, 45]
ALGORITHMS   = ['fedavg', 'fedprox', 'scaffold']
N_CLIENTS    = 10
N_ROUNDS     = 30
LOCAL_EPOCHS = 1
BATCH_SIZE   = 256
LR           = 0.01

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))

# ── LiteCNN (~106K params) ───────────────────────────────────────────────
class LiteCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ── Helpers ──────────────────────────────────────────────────────────────
def get_state(model):
    return {k: v.clone() for k, v in model.state_dict().items()}

def set_state(model, s):
    model.load_state_dict(s)

@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    out = model(X)
    return (out.argmax(1)==y).float().mean().item(), nn.functional.cross_entropy(out,y).item()

# ── Load data ────────────────────────────────────────────────────────────
print("Loading MNIST...", flush=True)
tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
ddir = os.path.join(SCRIPT_DIR, 'data')
trs = torchvision.datasets.MNIST(ddir, train=True,  download=True, transform=tfm)
tes = torchvision.datasets.MNIST(ddir, train=False, download=True, transform=tfm)
X_test = torch.stack([tes[i][0] for i in range(len(tes))])
y_test = torch.tensor([tes[i][1] for i in range(len(tes))])
X_all  = torch.stack([trs[i][0] for i in range(len(trs))])
y_all  = torch.tensor([trs[i][1] for i in range(len(trs))])
labels = y_all.numpy()
print(f"Loaded. Train={len(trs)}, Test={len(tes)}", flush=True)

# ── Partition ────────────────────────────────────────────────────────────
def dirichlet_partition(alpha, seed):
    rng = np.random.RandomState(seed)
    cidx = [[] for _ in range(N_CLIENTS)]
    for c in range(10):
        idx = np.where(labels==c)[0]; rng.shuffle(idx)
        p = rng.dirichlet([alpha]*N_CLIENTS)
        p = (p*len(idx)).astype(int); p[-1]=len(idx)-p[:-1].sum()
        s=0
        for i,n in enumerate(p):
            cidx[i].extend(idx[s:s+n].tolist()); s+=n
    return cidx

# ── Local training ───────────────────────────────────────────────────────
def train_local(model, X, y, lr, g_state=None, mu=0.0, c_g=None, c_l=None):
    model.train()
    use_sc = c_g is not None
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.0 if use_sc else 0.9)
    crit = nn.CrossEntropyLoss()
    K = 0
    for _ in range(LOCAL_EPOCHS):
        perm = torch.randperm(len(y))
        for s in range(0, len(y), BATCH_SIZE):
            idx = perm[s:s+BATCH_SIZE]
            opt.zero_grad()
            loss = crit(model(X[idx]), y[idx])
            if mu > 0 and g_state:
                prox = sum(((p-g_state[k])**2).sum() for k,p in model.named_parameters() if k in g_state)
                loss = loss + (mu/2)*prox
            loss.backward()
            if use_sc:
                with torch.no_grad():
                    for n,p in model.named_parameters():
                        if p.grad is not None and n in c_g:
                            p.grad += c_g[n] - c_l[n]
            opt.step(); K+=1

    ns = get_state(model)
    if use_sc:
        ncl = {n: c_l[n]-c_g[n]+(g_state[n]-ns[n])/(K*lr) for n in c_g}
        dc  = {n: ncl[n]-c_l[n] for n in c_g}
        return ns, ncl, dc
    return ns

# ── Run one experiment ───────────────────────────────────────────────────
def run_one(alg, seed):
    t0 = time.time()
    print(f"  {alg} seed={seed}", end="", flush=True)

    cidx = dirichlet_partition(ALPHA, seed)
    cdata = [(X_all[i], y_all[i]) for i in cidx]

    model = LiteCNN()
    gs = get_state(model)
    cg = {k:torch.zeros_like(v) for k,v in gs.items()} if alg=='scaffold' else None
    cls = [{k:torch.zeros_like(v) for k,v in gs.items()} for _ in range(N_CLIENTS)] if alg=='scaffold' else None
    nsamp = max(1, int(N_CLIENTS*PART_RATE))
    rng = np.random.RandomState(seed)

    rr = []
    for rd in range(N_ROUNDS):
        sel = rng.choice(N_CLIENTS, nsamp, replace=False)
        lss = []; szs = []
        for ci in sel:
            lm = LiteCNN(); set_state(lm, gs)
            cx, cy = cdata[ci]
            if alg=='fedavg':
                ls = train_local(lm, cx, cy, LR)
            elif alg=='fedprox':
                ls = train_local(lm, cx, cy, LR, g_state=gs, mu=0.01)
            elif alg=='scaffold':
                ls, ncl, dc = train_local(lm, cx, cy, LR, g_state=gs, c_g=cg, c_l=cls[ci])
                cls[ci] = ncl
                for n in cg: cg[n] = cg[n] + dc[n]/N_CLIENTS
            lss.append(ls if not isinstance(ls, tuple) else ls)
            szs.append(len(cdata[ci][1]))

        ts = sum(szs)
        gs = {k: sum(l[k]*(s/ts) for l,s in zip(lss,szs)) for k in gs}
        set_state(model, gs)

        if (rd+1)%3==0 or rd==0:
            a, lo = evaluate(model, X_test, y_test)
            rr.append({'round':rd+1,'accuracy':round(a*100,2),'loss':round(lo,4)})

    fa = rr[-1]['accuracy']
    el = time.time()-t0
    print(f" → {fa:.2f}% ({el:.0f}s)", flush=True)
    return {'algorithm': alg, 'seed': seed, 'final_accuracy': fa, 'trajectory': rr}

# ── Main ─────────────────────────────────────────────────────────────────
if __name__=='__main__':
    print(f"Multi-seed validation: α={ALPHA}, p={PART_RATE}")
    print(f"Seeds: {SEEDS}, Algorithms: {ALGORITHMS}")
    print("="*60)

    all_results = []
    for alg in ALGORITHMS:
        print(f"\n=== {alg.upper()} ===")
        for seed in SEEDS:
            r = run_one(alg, seed)
            all_results.append(r)

    print("\n" + "="*60)
    print("SUMMARY: α=0.1, p=0.2 across seeds")
    print("="*60)
    for alg in ALGORITHMS:
        accs = [r['final_accuracy'] for r in all_results if r['algorithm']==alg]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"{alg:>10}: {', '.join(f'{a:.2f}%' for a in accs)} | mean={mean_acc:.2f}% ± {std_acc:.2f}%")

    # Also include original seed=42 results for context
    print("\nOriginal seed=42 results (from all_results.csv):")
    print("  FedAvg:   11.95%")
    print("  FedProx:  14.41%")
    print("  SCAFFOLD: 67.74%")

    print(f"\nAll {len(all_results)} experiments complete.")
