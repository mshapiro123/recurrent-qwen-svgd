"""WEFT-1 math check, 2026-09-03. Four claims from the architecture walkthrough, verified numerically.
Run: python3 math_check_20260903.py
"""
import math, torch, numpy as np
torch.manual_seed(0); np.random.seed(0)
torch.set_default_dtype(torch.float64)

print("=" * 78)
print("CHECK 1 — bicameral symmetry: delta = 0 at init is a gradient-dead fixed point")
print("=" * 78)
# Toy two-hemisphere loop: shared K/V from h0, paired weights W_A = mu + dU dV^T, W_B = mu - dU dV^T,
# callosum A(rho) on the disagreement, combine at theta = 0 (consensus). Loss on combined output.
d, r, T, K = 32, 4, 6, 3
def run(delta_scale, rho=0.3, seed=1):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(d, d, generator=g) / math.sqrt(d)
    dU = torch.randn(d, r, generator=g) * delta_scale
    dV = torch.randn(d, r, generator=g) * delta_scale
    Wq_mu = torch.randn(d, d, generator=g) / math.sqrt(d)
    Wk = torch.randn(d, d, generator=g) / math.sqrt(d)   # shared (mu-only) K/V
    Wv = torch.randn(d, d, generator=g) / math.sqrt(d)
    for p in (mu, dU, dV, Wq_mu, Wk, Wv): p.requires_grad_(True)
    h0 = torch.randn(T, d, generator=g)
    Kc, Vc = h0 @ Wk.T, h0 @ Wv.T                          # cached once from h0
    hA = hB = h0
    delta = dU @ dV.T
    for k in range(K):
        WA, WB = mu + delta, mu - delta
        for h_side, W, hemi in ((hA, WA, +1), (hB, WB, -1)):
            pass
        qA = hA @ (Wq_mu + delta).T; qB = hB @ (Wq_mu - delta).T
        attA = torch.softmax(qA @ Kc.T / math.sqrt(d), -1) @ Vc
        attB = torch.softmax(qB @ Kc.T / math.sqrt(d), -1) @ Vc
        hA = hA + torch.tanh((hA + attA) @ WA.T)
        hB = hB + torch.tanh((hB + attB) @ WB.T)
        # callosum on the disagreement: mu untouched, delta_state scaled by (1 - 2 rho)
        m, dlt = (hA + hB) / 2, (hA - hB) / 2
        hA, hB = m + (1 - 2 * rho) * dlt, m - (1 - 2 * rho) * dlt
    y = (hA + hB) / 2                                        # combine at theta = 0
    loss = (y ** 2).sum()
    loss.backward()
    return dict(state_gap=(hA - hB).norm().item(), g_dU=dU.grad.norm().item(), g_dV=dV.grad.norm().item(),
                g_mu=mu.grad.norm().item())
for s in (0.0, 1e-3, 2e-2):
    out = run(s)
    print(f"delta init scale {s:>6}: |hA-hB| after K={K}: {out['state_gap']:.3e}   "
          f"|grad dU| {out['g_dU']:.3e}  |grad dV| {out['g_dV']:.3e}  |grad mu| {out['g_mu']:.3e}")
print("-> identity dL/d(delta) = dL/dW_A - dL/dW_B vanishes when the hemispheres coincide; dU=dV=0 never wakes.")

print("\nCHECK 1b — the callosum cannot re-symmetrize for rho < 1/2: disagreement regenerated every visit")
rho_list = [0.0, 0.25, 0.45, 0.5]
for rho in rho_list:
    out = run(2e-2, rho=rho)
    print(f"rho={rho:<5}: terminal state disagreement |hA-hB| = {out['state_gap']:.3e}")
print("-> rho=1/2 annihilates the STATE disagreement per visit, but the WEIGHT delta is untouched and regenerates it;"
      " only weights being equal (delta=0) is absorbing.")

print("\n" + "=" * 78)
print("CHECK 2 — engram gate scale: g = sigma(<RMSNorm(h), RMSNorm(W_K e)> / sqrt(d_m))")
print("=" * 78)
def rmsnorm(x): return x / x.pow(2).mean(-1, keepdim=True).sqrt()
d_model, d_m, n = 1024, 64, 20000
h = rmsnorm(torch.randn(n, d_model)); e = rmsnorm(torch.randn(n, d_model))   # independent at init
logit_d = (h * e).sum(-1)
for name, div in (("/ sqrt(d_m)=8 (as written, dot in R^d)", math.sqrt(d_m)), ("/ sqrt(d)=32 (dot dimension)", math.sqrt(d_model))):
    z = logit_d / div; g = torch.sigmoid(z)
    sat = ((g < 0.05) | (g > 0.95)).float().mean().item()
    print(f"  {name:<42} logit std {z.std():.3f}   gate mean {g.mean():.3f}   saturated fraction {sat:.3f}")
hm = rmsnorm(torch.randn(n, d_m)); em = rmsnorm(torch.randn(n, d_m))
z = (hm * em).sum(-1) / math.sqrt(d_m)
print(f"  dot taken in R^{d_m} then / sqrt(d_m)             logit std {z.std():.3f}   gate mean {torch.sigmoid(z).mean():.3f}")
print("-> the divisor must be sqrt(dimension of the inner product); sqrt(64) over a 1024-dim dot gives logit std 4 and ~"
      f"{((torch.sigmoid(logit_d/8)<0.05)|(torch.sigmoid(logit_d/8)>0.95)).float().mean().item():.0%} of gates saturated at init.")

print("\n" + "=" * 78)
print("CHECK 3 — L_stage: gradient on the SHARED coda scales with the number of decoded visits; compute multiplier")
print("=" * 78)
dc = 32
def stage_run(K, weights, seed=2):
    g = torch.Generator().manual_seed(seed)
    Wc = (torch.randn(dc, dc, generator=g) / math.sqrt(dc)).requires_grad_(True)   # 'coda'
    states = [torch.randn(8, dc, generator=g) for _ in range(K)]                    # per-visit combined states
    loss = 0.0
    for k, w in enumerate(weights):
        y = torch.tanh(states[k] @ Wc.T)
        loss = loss + w * (y ** 2).mean()
    loss.backward()
    return Wc.grad.norm().item()
for K in (1, 2, 4, 8):
    g_unnorm = stage_run(K, [1.0] * K)              # every visit weight 1 (+ final counted in the K)
    g_norm = stage_run(K, [1.0 / K] * K)            # weights sum to 1
    print(f"  K={K}: |grad coda| unnormalized {g_unnorm:.4f}   normalized (sum w_k = 1) {g_norm:.4f}")
print("-> unnormalized per-visit decoding grows the coda's gradient ~linearly in K; normalize sum_k w_k = 1 (or scale lambda_stage).")
N_pre, N_core, N_coda = 105e6, 57.5e6, 105e6
for K in (2, 4, 6):
    base = N_pre + K * N_core + N_coda
    full = N_pre + K * N_core + (K + 1) * N_coda
    samp = N_pre + K * N_core + 2 * N_coda
    print(f"  K={K}: AE as accounted {base/1e6:6.1f}M | full per-visit coda decode {full/1e6:6.1f}M ({full/base:.2f}x) | "
          f"one sampled visit + final {samp/1e6:6.1f}M ({samp/base:.2f}x)")
print("-> the ratified per-visit step logits through the FULL shared coda roughly double training compute at K=4;"
      " the compute plan was built on the left column.")

print("\n" + "=" * 78)
print("CHECK 4 — WHT: involution W W = d I, sequency ordering (row k has k sign changes), 2^-p exactness")
print("=" * 78)
def wht(x):
    d = x.shape[-1]; orig = x.shape; x = x.reshape(-1, d).clone(); h = 1
    while h < d:
        x = x.view(-1, d // (2 * h), 2, h); a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = torch.stack((a + b, a - b), dim=2).reshape(-1, d); h *= 2
    return x.reshape(orig)
def sequency_perm(d):
    p = int(math.log2(d)); idx = np.arange(d)
    gray = idx ^ (idx >> 1)                                              # Gray code of k ...
    rev = np.array([int(f"{i:0{p}b}"[::-1], 2) for i in gray])          # ... then bit-reversed: perm[k] = bitrev(gray(k))
    return rev
for d in (8, 16, 1024):
    I = torch.eye(d); W = wht(I)                                         # rows of the natural-order Hadamard
    assert torch.equal(W @ W, d * I), "involution failed"
    perm = sequency_perm(d); Ws = W[perm]
    changes = (torch.sign(Ws[:, 1:]) != torch.sign(Ws[:, :-1])).sum(1).numpy()
    ok = np.array_equal(changes, np.arange(d))
    print(f"  d={d}: W W = d I ✓   perm[k] = bitrev(gray(k)) gives row k exactly k sign changes: {ok}")
x = torch.randn(4096, 1024, dtype=torch.float32); p = 10
rt_pow2 = wht(wht(x)) * (2.0 ** -p)
rt_sqrt = wht(wht(x) / 1024 ** 0.5) / 1024 ** 0.5
print(f"  fp32 round-trip max |err|: * 2^-p  {(rt_pow2 - x).abs().max():.3e}   |  / sqrt(d) twice  {(rt_sqrt - x).abs().max():.3e}")
print(f"  exact bit-equality of the 2^-p round trip in fp32: {torch.equal(rt_pow2, x)}  (butterfly sums round; the SCALING is exact, the sums are not)")
# band structure at d=1024, E=8
print("  bands at d=1024, E=8: band b = sequencies [128b, 128b+127] in Walsh order; band 0 = lowest 128 sequencies")
