#!/usr/bin/env python3
"""
Ultra-fast FL experiment runner. Lightweight CNN, 1 local epoch, large batches.
Target: all 48 experiments in <60 minutes on M-series Mac CPU.
"""
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as transforms
import numpy as np, pandas as pd
import os, json, csv, time, sys

sys.stdout.reconfigure(line_buffering=True)

# ── Config ───────────────────────────────────────────────────────────────
ALPHAS       = [0.05, 0.1, 0.5, 1.0]
PART_RATES   = [0.2, 0.5, 0.8, 1.0]
ALGORITHMS   = ['fedavg', 'fedprox', 'scaffold']
N_CLIENTS    = 10
N_ROUNDS     = 30
LOCAL_EPOCHS = 1
BATCH_SIZE   = 256
LR           = 0.01

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR  = os.path.join(SCRIPT_DIR, '..', 'paper', 'figures')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'all_results.csv')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Lightweight CNN (no huge FC layer) ───────────────────────────────────
class LiteCNN(nn.Module):
    """Small CNN: 2 conv + 2 FC → ~106K params (majority in first FC layer)."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28x28→28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                  # →14x14
            nn.Conv2d(16, 32, 3, padding=1), # →14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                  # →7x7
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
def dirichlet_partition(alpha, seed=42):
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
def run_one(alg, alpha, pr):
    eid = f"{alg}_alpha{alpha}_part{pr}"
    if os.path.exists(RESULTS_FILE):
        try:
            if eid in pd.read_csv(RESULTS_FILE)['exp_id'].values: 
                print(f"  SKIP {eid}", flush=True); return None
        except: pass
    
    t0 = time.time()
    print(f"  RUN  {eid}", end="", flush=True)
    
    cidx = dirichlet_partition(alpha)
    cdata = [(X_all[i], y_all[i]) for i in cidx]
    
    model = LiteCNN()
    gs = get_state(model)
    cg = {k:torch.zeros_like(v) for k,v in gs.items()} if alg=='scaffold' else None
    cls = [{k:torch.zeros_like(v) for k,v in gs.items()} for _ in range(N_CLIENTS)] if alg=='scaffold' else None
    nsamp = max(1, int(N_CLIENTS*pr))
    rng = np.random.RandomState(42)
    
    rr = []; r80 = None
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
            if a>=0.80 and r80 is None: r80=rd+1
            rr.append({'round':rd+1,'accuracy':a,'loss':lo})
    
    pca = []
    for ci in range(N_CLIENTS):
        set_state(model, gs)
        a,_ = evaluate(model, *cdata[ci])
        pca.append(a)
    
    fa = rr[-1]['accuracy']
    el = time.time()-t0
    
    row = {'exp_id':eid,'algorithm':alg,'alpha':alpha,'participation_rate':pr,
           'final_accuracy':fa,'rounds_to_80pct':r80 if r80 else 999,
           'per_client_acc_mean':np.mean(pca),'per_client_acc_std':np.std(pca),
           'per_client_acc_min':np.min(pca),'round_results':json.dumps(rr)}
    
    fe = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE,'a',newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not fe: w.writeheader()
        w.writerow(row)
    
    print(f" → acc={fa:.4f}  ({el:.0f}s)", flush=True)
    return row

# ── Main ─────────────────────────────────────────────────────────────────
if __name__=='__main__':
    # Fresh start — delete old results with different settings
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
        print("Cleared old results (different settings).", flush=True)
    
    t0=time.time(); done=0
    for a in ALGORITHMS:
        print(f"\n=== {a.upper()} ===", flush=True)
        for al in ALPHAS:
            for pr in PART_RATES:
                r=run_one(a,al,pr)
                if r: done+=1
    print(f"\n{'='*60}")
    print(f"Completed {done} experiments in {(time.time()-t0)/60:.1f} min")
    print(f"Results: {RESULTS_FILE}")
