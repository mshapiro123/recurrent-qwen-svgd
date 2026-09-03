import numpy as np, math
np.random.seed(0)
# 1. LoopSplit execution order arithmetic (verbatim logic from modeling_nanbeige.py)
def loopsplit(L, M=None):
    if M is None: M = L//2
    assert L % M == 0
    first = (L-M)//2; ms, me = first, first+M
    reps = (L+M)//M
    order = [(i,None) for i in range(0,ms)] + [(i,r) for r in range(reps) for i in range(ms,me)] + [(i,None) for i in range(me,L)]
    return first, M, reps, L-me, len(order)
for L,M in [(22,None),(22,2),(22,22),(32,8)]:
    print("LoopSplit L=%d M=%s -> prefix %d / middle %d x %d repeats / suffix %d = effective depth %d (2L=%d)" % ((L,M)+loopsplit(L,M)+(2*L,)))
# 2. Nanbeige n-gram fusion gate: sigmoid(sign(z)*sqrt|z|) vs sigmoid(z), z~N(0,1) (RMSNorm both sides, /sqrt(d_f))
z = np.random.randn(200000)
g_plain = 1/(1+np.exp(-z)); g_sqrt = 1/(1+np.exp(-np.sign(z)*np.sqrt(np.abs(z))))
sat = lambda g: np.mean((g<0.05)|(g>0.95))
print("gate logit std 1: saturated frac plain %.4f  signed-sqrt %.4f ; std of gate plain %.3f sqrt %.3f" % (sat(g_plain), sat(g_sqrt), g_plain.std(), g_sqrt.std()))
z4 = 4*z; g4p = 1/(1+np.exp(-z4)); g4s = 1/(1+np.exp(-np.sign(z4)*np.sqrt(np.abs(z4))))
print("gate logit std 4 (catch-37-like): saturated plain %.4f  signed-sqrt %.4f" % (sat(g4p), sat(g4s)))
# gradient of sigmoid(sign z sqrt|z|) wrt z at small z: d/dz = sig' * 1/(2 sqrt|z|) -> unbounded at 0
print("d gate/dz at z=1e-4 (signed-sqrt): %.1f" % (0.25/ (2*math.sqrt(1e-4))))
# 3. Sinkhorn at mHC init: logits = bias with +20 diag, -20 off; 20 iterations
n=4; logits = np.full((n,n),-20.0); np.fill_diagonal(logits,20.0)
base = np.exp(logits - logits.max(-1,keepdims=True))
M = base.copy()
for _ in range(20):
    M = M/np.clip(M.sum(-1,keepdims=True),1e-6,None); M = M/np.clip(M.sum(-2,keepdims=True),1e-6,None)
print("mHC init h_res: max|M-I| = %.2e ; row sums %s ; spectral norm %.6f" % (np.abs(M-np.eye(n)).max(), np.round(M.sum(-1),6), np.linalg.norm(M,2)))
# random-logit Sinkhorn, 20 iters: doubly stochastic? spectral norm <= 1?
worst=0; worst_sn=0
for _ in range(2000):
    lg = np.random.randn(n,n)*3; B=np.exp(lg-lg.max(-1,keepdims=True)); Mx=B.copy()
    for _ in range(20):
        Mx = Mx/np.clip(Mx.sum(-1,keepdims=True),1e-6,None); Mx = Mx/np.clip(Mx.sum(-2,keepdims=True),1e-6,None)
    worst=max(worst, max(np.abs(Mx.sum(-1)-1).max(), np.abs(Mx.sum(-2)-1).max())); worst_sn=max(worst_sn,np.linalg.norm(Mx,2))
print("Sinkhorn-20 on random logits (std 3): worst marginal error %.2e, max spectral norm %.6f" % (worst, worst_sn))
# 4. two-stream doubly stochastic == our A(rho): check eigenvalues
for rho in [0.0,0.25,0.5]:
    A=(1-rho)*np.eye(2)+rho*np.array([[0,1],[1,0]]); print("rho=%.2f eig %s" % (rho, np.round(np.linalg.eigvals(A),3)))
