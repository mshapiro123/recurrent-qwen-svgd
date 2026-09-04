"""Symmetry forces in a bicameral pair: does delta collapse, hover, or grow?

Toy: two hemispheres A/B = same architecture (2-layer MLP), parameterized as
W_A = mu + dU dV^T, W_B = mu - dU dV^T (SwapLinear, handoff 5.5), identical input,
outputs combined y = cos(th)*mean + sin(th)*half-diff (S-2), AdamW on (mu, dU, dV, th)
with decoupled weight decay lambda_delta on dU,dV. Target: a nonlinear teacher.
We track ||delta||/||mu||, output correlation rho_hat(A,B) of residuals, theta,
and the loss, under five conditions.
"""
import torch, math, json
torch.manual_seed(0)
dev = 'cpu'
d_in, d_h, d_out, n = 32, 64, 8, 4096
X = torch.randn(n, d_in)
teacher = torch.nn.Sequential(torch.nn.Linear(d_in, 128), torch.nn.Tanh(), torch.nn.Linear(128, d_out))
with torch.no_grad(): Y = teacher(X); Y = (Y - Y.mean(0)) / Y.std(0)

class Swap(torch.nn.Module):
    def __init__(s, i, o, rank=8, sd=0.02):
        super().__init__()
        s.mu = torch.nn.Parameter(torch.randn(o, i) / math.sqrt(i))
        s.dU = torch.nn.Parameter(torch.randn(o, rank) * sd)
        s.dV = torch.nn.Parameter(torch.randn(i, rank) * sd)
    def forward(s, x, hemi): return x @ s.mu.T + hemi * ((x @ s.dV) @ s.dU.T)
    def delta_ratio(s): return ((s.dU @ s.dV.T).norm() / s.mu.norm()).item()

class Pair(torch.nn.Module):
    def __init__(s, theta0, rank=8, sd=0.02):
        super().__init__()
        s.l1 = Swap(d_in, d_h, rank, sd); s.l2 = Swap(d_h, d_out, rank, sd)
        s.theta = torch.nn.Parameter(torch.full((d_out,), theta0))
    def hemi(s, x, h): return s.l2(torch.nn.functional.gelu(s.l1(x, h)), h)
    def forward(s, x, noise=0.0):
        xa = x + noise * torch.randn_like(x); xb = x + noise * torch.randn_like(x)
        oa, ob = s.hemi(xa, +1), s.hemi(xb, -1)
        mu, dl = (oa + ob) / 2, (oa - ob) / 2
        return torch.cos(s.theta) * mu + torch.sin(s.theta) * dl, oa, ob
    def delta_ratio(s): return 0.5 * (s.l1.delta_ratio() + s.l2.delta_ratio())

def rho_hat(oa, ob, y):
    ra, rb = (oa - y).flatten(), (ob - y).flatten()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return ((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12)).item()

def run(name, theta0=0.0, wd_delta=0.1, lam_div=0.0, rho_star=0.5, noise=0.0, steps=3000, lr=3e-3):
    torch.manual_seed(1)
    m = Pair(theta0)
    delta_params = [p for nme, p in m.named_parameters() if 'dU' in nme or 'dV' in nme]
    mu_params = [p for nme, p in m.named_parameters() if '.mu' in nme]
    opt = torch.optim.AdamW([{'params': mu_params, 'weight_decay': 0.1},
                             {'params': delta_params, 'weight_decay': wd_delta},
                             {'params': [m.theta], 'weight_decay': 0.0}], lr=lr)
    traj = []
    for t in range(steps):
        idx = torch.randint(0, n, (256,))
        y, oa, ob = m(X[idx], noise)
        loss = ((y - Y[idx]) ** 2).mean()
        if lam_div > 0:
            ra, rb = (oa - Y[idx]).flatten(), (ob - Y[idx]).flatten()
            ra, rb = ra - ra.mean(), rb - rb.mean()
            rho = (ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12)
            loss = loss + lam_div * (rho - rho_star) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
        if t % 500 == 0 or t == steps - 1:
            with torch.no_grad():
                y, oa, ob = m(X, 0.0)
                traj.append((t, round(((y - Y) ** 2).mean().item(), 4), round(m.delta_ratio(), 4),
                             round(rho_hat(oa, ob, Y), 3), round(m.theta.abs().mean().item(), 3)))
    print(f"\n{name}\n  step   loss   |delta|/|mu|  rho_hat(A,B)  |theta|")
    for r in traj: print("  %5d  %.4f  %8.4f  %8.3f  %6.3f" % r)
    return traj

# gradient identities at theta = 0 (S-2 init)
m = Pair(0.0); idx = torch.arange(256); y, oa, ob = m(X[idx]); loss = ((y - Y[idx])**2).mean()
g = torch.autograd.grad(loss, [m.theta, m.l2.dU, m.l2.mu], retain_graph=True)
print("theta=0 init: |dL/dtheta| = %.3e   |dL/d(dU_out)| = %.3e   |dL/d(mu_out)| = %.3e" % (g[0].norm(), g[1].norm(), g[2].norm()))
print("  -> delta gets gradient only via mu's second-order (Jensen) term; theta gets <g, delta_out> which is O(|delta|).")

run("C1  baseline: theta0=0, wd_delta=0.1, no L_div, no noise  (as ratified)")
run("C2  wd_delta=0 (delta un-decayed)", wd_delta=0.0)
run("C3  theta0=0.05 (delta channel loss-visible from step 1)", theta0=0.05)
run("C4  L_div interior target rho*=0.5, lam=1.0", lam_div=1.0, rho_star=0.5)
run("C5  per-hemisphere entry noise sigma=0.3 (parallax by noise)", noise=0.3)
run("C6  C3 + C4 + C5 together", theta0=0.05, lam_div=1.0, rho_star=0.5, noise=0.3)

# ---- structural parallax: fixed, different, deterministic views per hemisphere
class PairView(Pair):
    def __init__(s, theta0, keep=0.75):
        super().__init__(theta0)
        g = torch.Generator().manual_seed(7)
        k = int(d_in * keep)
        s.register_buffer('mask_a', (torch.rand(d_in, generator=g) < keep).float())
        s.register_buffer('mask_b', (torch.rand(d_in, generator=g) < keep).float())
    def forward(s, x, noise=0.0):
        oa, ob = s.hemi(x * s.mask_a, +1), s.hemi(x * s.mask_b, -1)
        mu, dl = (oa + ob) / 2, (oa - ob) / 2
        return torch.cos(s.theta) * mu + torch.sin(s.theta) * dl, oa, ob

def run2(name, cls, steps=3000, lr=3e-3, **kw):
    torch.manual_seed(1); m = cls(0.0, **kw)
    dp = [p for nme, p in m.named_parameters() if 'dU' in nme or 'dV' in nme]
    mp = [p for nme, p in m.named_parameters() if '.mu' in nme]
    opt = torch.optim.AdamW([{'params': mp, 'weight_decay': 0.1}, {'params': dp, 'weight_decay': 0.1}, {'params': [m.theta], 'weight_decay': 0.0}], lr=lr)
    traj = []
    for t in range(steps):
        idx = torch.randint(0, n, (256,)); y, oa, ob = m(X[idx]); loss = ((y - Y[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if t % 1000 == 0 or t == steps - 1:
            with torch.no_grad():
                y, oa, ob = m(X); traj.append((t, round(((y - Y) ** 2).mean().item(), 4), round(m.delta_ratio(), 4), round(rho_hat(oa, ob, Y), 3), round(m.theta.abs().mean().item(), 3)))
    print(f"\n{name}\n  step   loss   |delta|/|mu|  rho_hat(A,B)  |theta|")
    for r in traj: print("  %5d  %.4f  %8.4f  %8.3f  %6.3f" % r)

run2("C7  structural parallax: each hemisphere sees a fixed 75% subset of input dims (different subsets)", PairView, keep=0.75)
run("C4b L_div rho*=0.0 (independent errors), lam=1.0", lam_div=1.0, rho_star=0.0)
run("C1L baseline, 9000 steps (does the null-space drift saturate?)", steps=9000)
