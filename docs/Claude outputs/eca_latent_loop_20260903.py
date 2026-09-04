"""Latent-loop ECA micro-experiment (epiplexity testbed transposed to a weight-tied latent core).

Task: predict Y = F^tau(X), tau iterations of an elementary cellular automaton (ECA) on a ring of n cells,
from X alone (no intermediate states emitted).  Model: embed -> tied residual block applied K times -> readout.
The block has a conv kernel of width 17 (radius 8 = tau), so even K = 1 has the receptive field to solve tau = 8
exactly; what K changes is COMPUTE, not visibility.  Prequential area = sum over steps of (loss - final loss).
Probe: a linear probe from the hidden state at visit j to the true intermediate state F^j(X).
"""
import torch, math, time, sys
torch.manual_seed(0)
n, tau, N_train, N_test = 32, 8, 20000, 2000
torch.set_num_threads(1)

def eca_step(x, rule):
    l, r = torch.roll(x, 1, -1), torch.roll(x, -1, -1)
    idx = (l * 4 + x * 2 + r).long()
    table = torch.tensor([(rule >> i) & 1 for i in range(8)], dtype=x.dtype)
    return table[idx]

def make(rule, N):
    X = torch.randint(0, 2, (N, n)).float(); states = [X]
    for _ in range(tau): states.append(eca_step(states[-1], rule))
    return states  # states[j] = F^j(X)

class Core(torch.nn.Module):
    def __init__(s, h=64, k=17):
        super().__init__()
        s.emb = torch.nn.Linear(1, h)
        s.conv = torch.nn.Conv1d(h, h, k, padding=k // 2, padding_mode='circular')
        s.mlp = torch.nn.Sequential(torch.nn.Linear(h, 2 * h), torch.nn.GELU(), torch.nn.Linear(2 * h, h))
        s.norm1, s.norm2 = torch.nn.LayerNorm(h), torch.nn.LayerNorm(h)
        s.out = torch.nn.Linear(h, 1)
    def block(s, z):
        z = z + s.conv(s.norm1(z).transpose(1, 2)).transpose(1, 2)
        z = z + s.mlp(s.norm2(z))
        return z
    def forward(s, x, K):
        z = s.emb(x.unsqueeze(-1)); hs = []
        for _ in range(K): z = s.block(z); hs.append(z)
        return s.out(z).squeeze(-1), hs

def probe_acc(h, target):
    # linear probe (ridge) from hidden h [N,n,d] to bits target [N,n]; report accuracy
    Hm = h.reshape(-1, h.shape[-1]); Hm = torch.cat([Hm, torch.ones(Hm.shape[0], 1)], 1); t = target.reshape(-1)
    w = torch.linalg.solve(Hm.T @ Hm + 1e-2 * torch.eye(Hm.shape[1]), Hm.T @ (2 * t - 1))
    return ((Hm @ w > 0).float() == t).float().mean().item()

def run(rule, K, steps=1200, lr=2e-3, bs=256, curriculum=False):
    torch.manual_seed(1)
    tr, te = make(rule, N_train), make(rule, N_test)
    m = Core(); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    bce = torch.nn.BCEWithLogitsLoss(); curve = []
    for t in range(steps):
        idx = torch.randint(0, N_train, (bs,))
        Kt = K if not curriculum else (1 if t < 200 else 2 if t < 400 else 4 if t < 700 else K)  # K curriculum 1->2->4->K, as in WEFT-1
        logits, _ = m(tr[0][idx], Kt); loss = bce(logits, tr[tau][idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if t % 25 == 0:
            with torch.no_grad():
                lg, _ = m(te[0][:1000], K); curve.append(bce(lg, te[tau][:1000]).item() / math.log(2))
    with torch.no_grad():
        lg, hs = m(te[0], K); final = bce(lg, te[tau]).item() / math.log(2)
        acc = ((lg > 0).float() == te[tau]).float().mean().item()
        # probe: does the hidden state at visit j encode F^j(X)?  (only meaningful when K == tau)
        probes = [round(probe_acc(hs[j - 1], te[j]), 3) for j in range(1, min(K, tau) + 1)] if K >= 1 else []
    area = sum(max(c - final, 0.0) for c in curve) * 25  # bits/cell x steps, above terminal
    return final, acc, area, probes

if __name__ == '__main__':
    rules = [int(a) for a in sys.argv[1:] if a.isdigit()] or [30, 54]; Ks = [1, 4, 8]
    print("rule  K   terminal bpc   acc     preq_area(bpc*steps)   probe acc of F^j(X) at visit j")
    cur = '--curriculum' in sys.argv
    rules = [r for r in rules]
    for rule in rules:
        for K in (Ks if not cur else [8]):
            t0 = time.time(); f, a, ar, pr = run(rule, K, curriculum=cur)
            print("%4d %3d%s  %8.4f   %6.4f   %10.1f            %s   (%.0fs)" % (rule, K, 'c' if cur else ' ', f, a, ar, pr, time.time() - t0)); sys.stdout.flush()
