# Latent Microstep Reasoning for Qwen2.5-0.5B
## Teacher Canonicalization, Flow-Hitch Distillation, Recurrent Architecture, Training Plan, and Strategy-Agent Handoff

**Version:** 0.6.1  
**Date:** August 3, 2026  
**Status:** Research design; mathematically reviewed, empirically unvalidated  
**Backbone:** Qwen2.5-0.5B / Qwen2.5-0.5B-Instruct  
**Working name:** Latent Microstep Bridge (LMB)  
**Training method:** Canonical Flow-Hitch Distillation with Progressive Latent Residual Refinement (CFH-PLRR)
**v0.6.1 revision:** retains the v0.6 whitening analysis and binds one effective-eigenvalue rule, a shared PCA orientation across partial-whitening arms, affine flow geometry without persistent-state renormalization, explicit bridge initialization, and staged adaptation contracts.

---

# 1. Executive conclusion

The proposed system is mathematically coherent if it is framed narrowly:

> A recurrent latent scratchpad is trained to perform a small amount of additional nonlinear computation between emitted tokens, so that successive latent loops progressively improve a bounded set of future-token distributions and the final answer distribution.

The proposal is **not** mathematically justified if framed as:

> A few latent vectors faithfully encode or reproduce a long chain-of-thought trace.

The strongest defensible design combines seven ideas:

1. **Localized recurrence:** split Qwen2.5-0.5B into a lower prefix and upper suffix, and recurrently process only a small scratchpad at the insertion point.
2. **Anchored residual reentry:** preserve the original pretrained hidden state and add only a small, normalized scratchpad-derived correction.
3. **Two-dimensional supervision:** distinguish latent refinement depth \(k\) from speculative token horizon \(j\). The model predicts a grid \(q_{k,j}\), not one undifferentiated future.
4. **Self-conditioned rollout:** train the same shared recurrent module on the state distributions it creates, and train speculative horizons on student-generated prefixes where deployment will encounter them.
5. **Acceptance-aligned residual refinement:** train cumulative predictions toward verified target distributions, while optionally shaping each loop as a bounded partial correction rather than forcing every loop to independently solve the full task.
6. **Canonical teacher-state distillation:** project selected teacher trajectories into a small, fixed, well-conditioned latent space that preserves short-horizon predictive information rather than raw activation variance alone.
7. **Training-only flow hitch:** connect the frozen teacher canonical endpoint and functional probe to the student only through stopped-gradient targets and differentiable losses; remove the teacher branch entirely for inference.

AngelSpec materially strengthens the case for shared-parameter multi-depth training, self-generated rollout, target-distribution distillation, and genuine online acceptance measurement. It does **not** prove that model-size differences form a reasoning trace, nor that a DeltaNet update should be copied literally into a latent scratchpad. AngelSpec’s MTP depth is primarily **future-token position**; the proposed scratchpad loop is **computation depth at the same token boundary**. Those axes must remain separate.

The recommended canonicalizer is a **regularized, optionally partial-whitened predictive reduced-rank linear map**, optionally preceded by a small fixed or lightly learned structured pooling step. This is the best first implementation because it provides deterministic endpoints, linear increment consistency, Euclidean flow geometry, low storage cost, and tractable gradients. PCA, Tucker, attention pooling, deterministic autoencoders, and VAEs remain explicit ablations rather than assumptions.

The central experimental question is:

\[
\boxed{
\text{Can a small shared recurrent state produce cost-effective, causally necessary reductions in future predictive error?}
}
\]

The system should proceed only if recurrent loops improve actual verified acceptance, final task quality, or explicit-reasoning displacement beyond compute-matched feed-forward and pause-token controls.

---

# 2. Claims, proofs, and open hypotheses

## 2.1 Claims supported by elementary mathematics

### Claim A — Per-loop supervision shortens gradient paths

If a recurrent state evolves as

\[
z_{k+1}=\Phi_\theta(z_k,h),
\]

and only the terminal state receives loss \(\mathcal L_K\), then

\[
\nabla_{z_t}\mathcal L_K
=
\left(
\prod_{i=t}^{K-1}
J_i^\top
\right)
\nabla_{z_K}\mathcal L_K,
\qquad
J_i=\frac{\partial z_{i+1}}{\partial z_i}.
\]

With losses at every loop,

\[
\mathcal L=\sum_{k=1}^{K}w_k\mathcal L_k,
\]

the gradient becomes

\[
\nabla_{z_t}\mathcal L
=
\sum_{k=t}^{K}
w_k
\left(
\prod_{i=t}^{k-1}
J_i^\top
\right)
\nabla_{z_k}\mathcal L_k.
\]

The \(k=t\) term has no recurrent Jacobian product. Deep supervision therefore supplies a direct learning signal to early loops.

### Claim B — Small residual gates preserve the pretrained path

For an anchored bridge

\[
h_{k+1}
=
h_0+a_k(h_k-h_0)+g_kF_\theta(h_k,z_k),
\]

the loop Jacobian is

\[
J_k=a_kI+g_kJ_{F,k}.
\]

When \(a_k\) is near one and \(g_k\) is initially small, the mapping begins near the identity. The pretrained forward path is exactly retained when zero loops are used and approximately retained for early small-gate training.

### Claim C — KL distillation supplies a prediction-error gradient

For a teacher distribution \(p\), student logits \(\ell\), and

\[
q=\operatorname{softmax}(\ell),
\]

the forward KL loss

\[
\mathcal L_{\mathrm{KL}}=\operatorname{KL}(p\|q)
\]

has gradient

\[
\frac{\partial\mathcal L_{\mathrm{KL}}}{\partial \ell}=q-p.
\]

A gradient step moves logits in the direction \(p-q\). This is a true categorical prediction-error correction.

### Claim D — Single-position speculative acceptance equals distributional overlap

Under standard speculative rejection sampling, a proposal \(v\sim q\) is accepted with probability

\[
\min\left(1,\frac{p(v)}{q(v)}\right).
\]

The expected acceptance probability is

\[
\alpha(p,q)
=
\sum_v q(v)
\min\left(1,\frac{p(v)}{q(v)}\right)
=
\sum_v\min(p(v),q(v))
=
1-\operatorname{TV}(p,q).
\]

Therefore TV overlap is directly aligned with single-position acceptance.

### Claim E — A deterministic latent loop cannot create information absent from its inputs

If \(Z=f(H)\) is deterministic, then for a future variable \(Y\),

\[
I(Z;Y)\leq I(H;Y)
\]

by the data-processing inequality.

The loop can reorganize information, implement additional computation, improve linear accessibility, and allocate capacity toward the future. It cannot create information that is not inferable from context.

### Claim F — Logit interpolation gives a precise partial correction

Given the current distribution \(q\), teacher distribution \(p\), and step fraction \(\beta\in[0,1]\), define

\[
\ell^\star_\beta
=
(1-\beta)\log q+\beta\log p.
\]

After normalization,

\[
r_\beta(v)
=
\frac{
q(v)^{1-\beta}p(v)^\beta
}{
\sum_u q(u)^{1-\beta}p(u)^\beta
}.
\]

This is an exact exponential-family interpolation between \(q\) and \(p\). It provides a mathematically clean bounded residual target for one latent loop.


### Claim G — Linear canonicalizers preserve increments exactly

For an affine canonicalizer

\[
C(h)=Wh+b,
\]

the difference between two projected states is

\[
C(h_2)-C(h_1)=W(h_2-h_1).
\]

Therefore teacher hidden-state increments map to canonical increments without curvature or path ambiguity. This does not imply that the selected subspace is semantically optimal; it establishes why linear canonicalizers are unusually compatible with residual and rectified-flow targets.

### Claim H — Whitening improves Euclidean conditioning

Let a projected teacher state have covariance

\[
\Sigma_z=U\Lambda U^\top.
\]

Define the truncated PCA-whitened state

\[
\widetilde z
=
\Lambda_{\mathrm{eff},r}^{-1/2}U_r^\top(z-\mu).
\]

This is the PCA-coordinate form used by the implementation. All whitening-strength arms share the same frozen basis \(U_r\), so their only difference is variance equalization.

Then, on the retained subspace,

\[
\operatorname{Cov}(\widetilde z)\approx I.
\]

Consequently, an isotropic squared or Huber flow loss does not automatically overweight high-variance coordinates. Whitening is not information preserving when low-eigenvalue dimensions are truncated, and it can amplify noise if eigenvalues are not floored.


#### What “whitening” means in this design

Whitening is a **fixed population-level affine change of coordinates** applied after the teacher state has already been compressed into the predictive canonical subspace. It is more specific than ordinary normalization and is not, by itself, a shrinkage or dimension-reduction method.

Let the predictive projector produce

\[
z=A^\top(h-\mu_h)\in\mathbb R^r.
\]

Across a calibration corpus, estimate

\[
\mu_z=\mathbb E[z]
\]

and

\[
\Sigma_z
=
\mathbb E\left[(z-\mu_z)(z-\mu_z)^\top\right].
\]

The implementation uses PCA coordinates. If

\[
\Sigma_z=U\Lambda U^\top,
\]

define one effective spectrum

\[
\lambda_{\mathrm{eff},i}
=
\max(\lambda_i,10^{-4}\lambda_{\max},10^{-8}).
\]

For whitening strength \(\alpha\in[0,1]\), the canonical state is

\[
z_{\alpha}
=
\Lambda_{\mathrm{eff}}^{-\alpha/2}U^\top(z-\mu_z).
\]

The effective eigenvalues are computed once, stored, and used directly. The forward pass adds no second epsilon. In the original projected coordinates the induced metric is

\[
U\Lambda_{\mathrm{eff}}^{-\alpha}U^\top,
\]

which is regularized Mahalanobis at \(\alpha=1\) and fractional Mahalanobis at \(\alpha=0.5\).

Ignoring finite-sample error and the ridge term,

\[
\mathbb E[z_{\mathrm w}]=0,
\qquad
\operatorname{Cov}(z_{\mathrm w})=I.
\]

Whitening therefore performs three operations:

1. subtract the population mean;
2. rotate into uncorrelated covariance directions;
3. scale each retained direction to comparable population variance.

It is the multivariate analogue of a z-score, with the additional step of removing cross-coordinate correlation.

#### Whitening is not the same as RMSNorm

RMSNorm is computed separately for each example:

\[
\operatorname{RMSNorm}(z)
=
\frac{z}{\sqrt{r^{-1}\sum_i z_i^2+\epsilon}}.
\]

It controls the total magnitude of one state but does not estimate or remove population means, feature correlations, or direction-specific variances.

Whitening is estimated across the teacher calibration distribution and then frozen. The intended sequence is:

```text
teacher hidden state
        -> predictive low-rank projection
        -> one frozen PCA basis plus fixed population equalization
        -> unnormalized canonical flow endpoint
        -> RMSNorm on recurrent-module inputs and innovations only
```

The two mechanisms solve different problems:

- whitening defines a globally well-conditioned canonical geometry;
- RMSNorm stabilizes recurrent-module inputs and innovations without changing the persistent canonical state.

#### Whitening is not inherently shrinkage

Along covariance eigen-direction \(i\), whitening applies

\[
z_{\mathrm w,i}
=
\frac{z_i-\mu_i}{\sqrt{\lambda_{\mathrm{eff},i}}}.
\]

Thus it scales high-variance directions down and low-variance directions up. Pure whitening can therefore **amplify**, rather than shrink, weak directions.

Three distinct operations should not be conflated:

1. **Subspace selection or truncation** removes dimensions and is the actual compression step.
2. **Ridge regularization** \(\epsilon I\) limits amplification of tiny eigenvalues and has a shrinkage-like stabilizing effect.
3. **Whitening** equalizes and decorrelates the retained coordinates.

The recommended implementation first selects a reliable predictive rank, then applies regularized whitening only within that retained subspace.

#### Why whitening is useful for flow matching

Suppose the student and teacher canonical states differ by

\[
\Delta z=z_T-z_S.
\]

An ordinary Euclidean flow loss is

\[
\|\Delta z\|_2^2.
\]

After PCA-form whitening,

\[
\|\Lambda_{\mathrm{eff}}^{-1/2}U^\top\Delta z\|_2^2
=
\Delta z^\top
U\Lambda_{\mathrm{eff}}^{-1}U^\top
\Delta z.
\]

This is a regularized Mahalanobis distance in the original projected space. Errors are measured relative to the natural variability of each teacher direction:

- errors in normally low-variance directions receive more weight;
- errors in naturally high-variance directions receive less weight;
- correlated error components are not counted redundantly.

This improves optimization conditioning when a squared or Huber velocity/state loss is used. Without whitening, a few high-variance coordinates may dominate the recurrent-flow gradient even when they are not the most useful predictive coordinates.

Because both the predictive projection and frozen whitening transform are affine,

\[
C(h)=W_{\mathrm w}\left[A^\top(h-\mu_h)-\mu_z\right],
\]

and teacher increments remain exact linear increments:

\[
C(h_2)-C(h_1)
=
W_{\mathrm w}A^\top(h_2-h_1).
\]

This property is one reason the linear canonicalizer is particularly compatible with rectified or residual flow targets.

#### Risks and safeguards

Full whitening can be harmful if small eigenvalues reflect sampling noise or irrelevant directions. The strategy agent should treat whitening strength as an ablation, not a fixed truth.

Required safeguards:

1. **Predictive rank selection:** retain only dimensions that improve held-out future-probe performance.
2. **Eigenvalue floor:** compute once and store
   \[
   \lambda_{\mathrm{eff},i}=\max(\lambda_i,10^{-4}\lambda_{\max},10^{-8}).
   \]
   Log raw and effective condition numbers separately; do not add another epsilon in the forward pass.
3. **Calibration holdout:** estimate whitening statistics on a representative training/calibration split and evaluate on a separate split.
4. **Frozen transform:** do not recompute whitening per minibatch during student training.
5. **Condition-number logging:** track \(\lambda_{\max}/\lambda_{\min}\) before and after flooring.
6. **Gradient audit:** compare per-coordinate and total flow-gradient norms with and without whitening.
7. **Probe audit:** verify that whitening does not reduce held-out functional-probe fidelity.
8. **Sparse-domain audit:** compute statistics by workload or use a mixture only if code, math, chat, and retrieval distributions are materially different.

#### Partial whitening

Full equalization is not always optimal. Define

\[
T_\alpha
=
\Lambda_{\mathrm{eff}}^{-\alpha/2}U^\top,
\qquad
0\leq\alpha\leq1.
\]

Then:

- \(\alpha=0\): no variance equalization; the shared frozen PCA rotation remains, so this is not literally centering only;
- \(\alpha=1\): full whitening;
- \(0<\alpha<1\): partial whitening.

In eigen-coordinates, the scaling is

\[
\lambda_{\mathrm{eff},i}^{-\alpha/2}.
\]

Partial whitening reduces dominance by high-variance axes without fully amplifying every low-variance direction. Recommended first ablation:

\[
\alpha\in\{0,0.5,1.0\}.
\]

Experiments 0A and 0B may screen out unsuitable values, but final selection occurs only after matched DEV-only pilots on the built module. The selection criterion is downstream flow convergence, final upper-model quality, and verified speculative acceptance—not covariance identity alone.

#### PCA versus ZCA orientation

Two standard forms are:

PCA whitening:

\[
z_{\mathrm{PCA}}
=
\Lambda_r^{-1/2}U_r^\top(z-\mu),
\]

which rotates into principal coordinates, and ZCA whitening:

\[
z_{\mathrm{ZCA}}
=
U_r\Lambda_r^{-1/2}U_r^\top(z-\mu),
\]

which preserves an orientation as close as possible to the input coordinates.

For a newly defined canonical space, orientation is arbitrary; PCA-form whitening is simpler and naturally combines truncation with conditioning. Because the Huber term is applied per coordinate and is not rotation-invariant, every \(\alpha\) arm must use the same frozen PCA basis. This shared-basis contract makes the ablation isolate scaling rather than orientation. ZCA is useful only when preserving the pre-whitened projector orientation has demonstrated interpretive or probe value.

#### Recommended implementation contract

The v1 canonicalizer should expose fixed buffers:

```python
projector_weight      # predictive low-rank map
teacher_mean          # mean before projection, if needed
canonical_mean        # projected population mean
whiten_basis          # retained covariance eigenvectors
whiten_eigenvalues    # floored retained eigenvalues
whiten_alpha          # 0, 0.5, or 1.0
```

The forward transform is:

```python
z = predictive_projector(h)
z = z - canonical_mean
scale = whiten_eigenvalues.pow(-0.5 * whiten_alpha)
z = (z @ whiten_basis) * scale
```

If the basis is already expressed in canonical coordinates, the final reconstruction rotation is unnecessary. The student is trained directly in the resulting canonical coordinates.

The default hypothesis is **regularized partial whitening** with \(\alpha=0.5\), compared against \(\alpha\in\{0,0.5,1.0\}\) on one shared frozen PCA basis. Final selection follows matched trained pilots; full whitening becomes the default only if it improves held-out predictive fidelity, flow conditioning, upper-model quality, verified acceptance, and deployment metrics without amplifying noisy directions.

### Claim I — Predictive reduced-rank projection optimizes a different objective from PCA

For teacher state matrix \(X\) and future target matrix \(Y\), reduced-rank regression solves

\[
\min_{W:\operatorname{rank}(W)\le r}
\|Y-XW\|_F^2.
\]

A factorization \(W=AB^\top\) gives a compact canonical state

\[
z=A^\top x,
\qquad
\widehat y=Bz.
\]

PCA instead minimizes reconstruction error of \(X\). Predictive reduced-rank projection therefore selects teacher directions that explain the defined future target, even when those directions are not the highest-variance components of the teacher representation.

### Claim J — A deterministic teacher endpoint permits a clean training-only hitch

Let

\[
z_T=C_T(H_T)
\]

be computed by a frozen teacher and canonicalizer under `no_grad`, and let the student state be

\[
z_{k+1}=\Phi_\theta(z_k,h_0).
\]

For a loss

\[
\mathcal L(z_{k+1},\operatorname{sg}[z_T]),
\]

gradients satisfy

\[
\nabla_{\theta_T}\mathcal L=0,
\qquad
\nabla_{\theta}\mathcal L
=
\frac{\partial z_{k+1}}{\partial\theta}^\top
\nabla_{z_{k+1}}\mathcal L.
\]

The teacher supplies privileged targets but is not part of the student inference graph.

## 2.2 Claims supported by AngelSpec’s empirical evidence

AngelSpec supports the following engineering conclusions:

1. A shared MTP module trained only for one teacher-forced application suffers a train–inference mismatch when recurrently reused for deeper speculative positions.
2. Autoregressively unrolling the same shared module during training, feeding each depth its own previous prediction, substantially improves later-depth acceptance.
3. Each prediction depth should receive a distinct shifted teacher distribution.
4. KL performs better than hard-label CE for drafter distillation in AngelSpec’s ablation.
5. A negative-log overlap cold start followed by an end-to-end TV objective outperformed direct TV optimization from the initial checkpoint.
6. Genuine online speculative evaluation is necessary because proxy losses do not reliably determine serving speed.
7. Workload entropy and continuation structure affect which drafting architecture is best.
8. Draft depth has a hardware-dependent break-even committed length; longer drafts are useful only when additional accepted progress exceeds their proposal and verification cost.

These findings are directly relevant to the speculative-horizon axis of the proposed system.

## 2.3 Claims that remain hypotheses

The following are plausible but unproved:

1. A latent recurrent state will learn an internal operation analogous to reasoning rather than a task-specific predictive shortcut.
2. Cross-scale differences between Qwen2.5-7B, 14B, and 32B correspond to progressively more useful computational abstractions.
3. Explicitly supervising residual logit directions improves over cumulative KL alone.
4. Different recurrent loops will specialize into sequential microsteps without explicit span-boundary supervision.
5. Improved draft acceptance will transfer to better final answer quality.
6. A scratchpad that helps short-horizon prediction will eventually displace meaningful amounts of explicit reasoning text.
7. Teacher agreement implies correctness. Same-family models may share errors and biases.
8. A larger teacher’s correction is necessarily teachable by a 0.5B student.

The experiment plan treats each as falsifiable.

---

# 3. The most important conceptual correction: two depth axes

AngelSpec and the latent scratchpad both involve repeated computation, but they repeat along different axes.

## 3.1 Latent refinement depth

Let

\[
k\in\{0,\ldots,K\}
\]

index recurrent scratchpad loops before the next emitted token. At fixed text context \(x_{\leq t}\),

\[
z_{k+1}
=
\Phi_\theta(z_k,h_0).
\]

This is **compute depth**.

## 3.2 Speculative horizon

Let

\[
j\in\{1,\ldots,J\}
\]

index future token positions:

\[
y_{t+1},y_{t+2},\ldots,y_{t+J}.
\]

This is **autoregressive future depth**.

AngelSpec’s shared-parameter MTP unrolling primarily moves along \(j\): the drafter’s previous proposed token becomes input to the next speculative position.

The proposed latent scratchpad moves along \(k\): the same text boundary receives additional latent computation before output.

## 3.3 The correct prediction object is a grid

Define

\[
q_{k,j}
=
q_\theta
\left(
y_{t+j}
\mid
x_{\leq t},
z_k,
\widehat y_{t+1:t+j-1}
\right).
\]

The full training object is:

\[
\mathbf Q
=
\{q_{k,j}:k=0,\ldots,K,\ j=1,\ldots,J\}.
\]

This separation prevents three common errors:

- treating loop 2 as though it automatically predicts token \(t+2\);
- assigning 7B/14B/32B to loops and calling them reasoning steps;
- measuring only a single aggregate KL while ignoring whether improvement comes from more computation or merely from a different future position.

The architecture may use latent loops to improve every horizon, while student-generated token rollout is separately used to expose future horizons to realistic prefixes.

---

# 4. Model architecture

## 4.1 Backbone split

Let the 24-layer Qwen2.5-0.5B model be split at layer \(m\):

\[
h_0=F_{\leq m}(x_{\leq t}),
\]

\[
p_{\mathrm{final}}
=
F_{>m}(h_K).
\]

Initial insertion sweep:

- layer 8;
- layer 12;
- layer 16.

Do not assume the exact midpoint is optimal.

## 4.2 Scratchpad state

Use fixed latent slots:

\[
S_k\in\mathbb R^{B\times M\times r}.
\]

Recommended v1:

- slots \(M=8\);
- scratch width \(r=256\);
- loops \(K\leq4\);
- model width \(d=896\).

Initialize scratch slots from learned anchors conditioned on the cached lower representation:

\[
S_0
=
\operatorname{Init}_\theta(A,h_0).
\]

The lower Transformer executes once. Recurrent loops attend to cached prompt-side context.

## 4.3 Shared recurrent operator

Use one shared block across loops:

\[
U_k
=
R_\theta
\left(
\operatorname{RMSNorm}(S_k),
\operatorname{Cache}(h_0),
e_k
\right),
\]

where \(e_k\) is a small loop-index embedding.

Normalize the innovation:

\[
\widehat U_k
=
\frac{U_k}
{\operatorname{RMS}(U_k)+\epsilon}
\cdot
\operatorname{sg}
\left[
\operatorname{RMS}(S_k)
\right].
\]

Update the persistent canonical state without an outer normalization:

\[
S_{k+1}
=
a_k S_k+\eta_k\widehat U_k.
\]

RMSNorm is used only to construct module inputs and normalized innovations. It is never applied to the persistent state after the residual update. Endpoints, serial interpolation targets, and flow losses remain in unnormalized partially whitened coordinates; state-scale drift is measured by telemetry rather than projected onto a sphere.

Recommended initialization:

\[
a_k\in[0.95,1.0],
\qquad
\eta_k\approx0.02-0.04.
\]

## 4.4 Anchored bridge

Read the scratchpad back into the Qwen residual stream:

\[
\Delta h_k
=
P_{\mathrm{out}}
\operatorname{CrossAttn}
\left(
Q=\operatorname{RMSNorm}(h_0),
K=\operatorname{RMSNorm}(S_k),
V=\operatorname{RMSNorm}(S_k)
\right).
\]

Scale the correction relative to the pretrained state:

\[
\widehat{\Delta h}_k
=
\frac{\Delta h_k}
{\operatorname{RMS}(\Delta h_k)+\epsilon}
\cdot
\operatorname{sg}
\left[
\operatorname{RMS}(h_0)
\right].
\]

Use:

\[
h_k
=
h_0
+
\rho_k(h_{k-1}-h_0)
+
g_k\widehat{\Delta h}_k.
\]

Initialize:

\[
\rho_k\approx0.9-0.98,
\qquad
g_k\approx0.01-0.03.
\]

Initialize \(P_{\mathrm{out}}\sim\mathcal N(0,(10^{-3})^2)\), never exactly zero. This leaves one near-zero bridge factor rather than multiplying two exact zeros. The zero-loop path remains exactly the pretrained model.

## 4.5 Predictive residual readout

Let \(\ell_{k,j}\) denote logits for horizon \(j\) after loop \(k\). Produce an innovation-specific residual:

\[
\Delta \ell_{k,j}
=
W_{\Delta,j}
\operatorname{Pool}(\widehat U_k).
\]

Update cumulative draft logits:

\[
\ell_{k,j}
=
\ell_{k-1,j}
+
b_{k,j}\Delta\ell_{k,j},
\]

where \(b_{k,j}\in[0,1]\) is a learned write gate.

In v1, use:

- one scalar gate per example, loop, and horizon;
- shared low-rank output projections;
- tied vocabulary head;
- no full per-token, per-channel gating.

The residual head is an auxiliary and deployment drafter. The final answer still passes through the actual upper Qwen layers after the last loop.

## 4.6 Separate control state

Maintain a small control state \(c_k\) containing:

- predicted gain from another loop;
- confidence;
- scratch innovation norm;
- projected acceptance improvement;
- optional halt probability.

Do not rely solely on the normalized scratch vector’s magnitude to encode control.

---

# 5. Progressive Latent Residual Refinement

## 5.1 Why “residual refinement” is preferable to “DeltaNet”

DeltaNet updates a key-addressed associative memory across sequence time. The proposed model updates a latent predictive state across computational loops. It borrows the principle:

\[
\text{retain current estimate}
+
\text{write remaining error}.
\]

It does not implement the same memory object or update algebra. “Progressive Latent Residual Refinement” avoids claiming an architectural identity that is not present.

## 5.2 Cumulative target loss

For teacher target \(p_{k,j}^\star\),

\[
q_{k,j}
=
\operatorname{softmax}(\ell_{k,j}),
\]

use:

\[
\mathcal L_{\mathrm{cum}}
=
\sum_{k,j}
w_kv_j
\operatorname{KL}
\left(
\operatorname{sg}[p_{k,j}^\star]
\|
q_{k,j}
\right).
\]

This is the primary probability-correctness objective.

## 5.3 Bounded partial-step target

For each loop, define the current student \(q_{k-1,j}\), selected teacher \(p_{k,j}^\star\), and desired fractional progress \(\beta_{k,j}\).

Construct:

\[
r_{k,j}^{(\beta)}
\propto
q_{k-1,j}^{1-\beta_{k,j}}
\left(
p_{k,j}^\star
\right)^{\beta_{k,j}}.
\]

Train:

\[
\mathcal L_{\mathrm{bridge}}
=
\sum_{k,j}
w_kv_j
\operatorname{KL}
\left(
\operatorname{sg}
\left[
r_{k,j}^{(\beta)}
\right]
\|
q_{k,j}
\right).
\]

This is mathematically cleaner than directly matching an arbitrary cosine direction. It tells each loop to cover a controlled fraction of the remaining correction.

Choices:

- fixed \(\beta_k\), such as \([0.5,0.6,0.8]\);
- learned \(\beta_{k,j}\) with regularization;
- teacher-informed \(\beta^\star\) during curriculum training.

The final loop still receives the full verified teacher target.

## 5.4 Progress loss

Define:

\[
D_{k,j}
=
\operatorname{KL}
\left(
p^\star_j\|q_{k,j}
\right).
\]

Require a fractional reduction:

\[
\mathcal L_{\mathrm{progress}}
=
\sum_{k,j}
\max
\left(
0,
D_{k,j}
-
(1-\delta_{k,j})
\operatorname{sg}[D_{k-1,j}]
\right).
\]

This directly measures whether an additional loop reduces remaining error.

A progress loss is more interpretable than delta-vector alignment and less sensitive to arbitrary logit coordinates.

## 5.5 Optional logit-delta diagnostic

Centered logits remove additive invariance:

\[
C(\ell)=\ell-\operatorname{mean}(\ell).
\]

The ideal remaining correction is:

\[
d^\star_{k,j}
=
C
\left(
\log p^\star_{k,j}
-
\log\operatorname{sg}[q_{k-1,j}]
\right).
\]

The predicted correction is:

\[
d_{k,j}
=
C
\left(
\ell_{k,j}
-
\operatorname{sg}[\ell_{k-1,j}]
\right).
\]

Track:

\[
\cos(d_{k,j},d^\star_{k,j}).
\]

Use this as:

- a diagnostic in v1;
- a low-weight auxiliary only after cumulative and bridge losses work.

It is redundant with KL in the unconstrained full-logit case and may overconstrain a low-rank recurrent model.

---

# 6. Cross-scale Qwen teacher lattice

## 6.1 Purpose

Use Qwen2.5-7B, 14B, and 32B to estimate:

- stable consensus targets;
- student capability gaps;
- scale-emergent corrections;
- ambiguous or nonmonotonic examples;
- examples likely to benefit from additional latent compute.

The teacher ladder is a curriculum and uncertainty instrument, not a reasoning trace.

## 6.2 Teacher agreement

Let

\[
\bar p_j
=
\frac{1}{3}
\left(
p_{7,j}+p_{14,j}+p_{32,j}
\right).
\]

Define:

\[
\operatorname{JS}_j
=
\frac{1}{3}
\sum_{s\in\{7,14,32\}}
\operatorname{KL}(p_{s,j}\|\bar p_j).
\]

Normalized agreement:

\[
A_j
=
1-\frac{\operatorname{JS}_j}{\log 3}.
\]

Also store argmax agreement, entropy, and calibration.

## 6.3 Student gap

For a selected target \(p_j^\star\),

\[
G_j
=
\operatorname{KL}(p_j^\star\|q_{0,j}).
\]

Agreement and gap define distinct curricula:

| Agreement | Gap | Meaning |
|---|---:|---|
| high | low | consensus easy; teach restraint |
| high | high | stable but challenging; high-value distillation |
| low | low | benign ambiguity |
| low | high | capability gap, teacher conflict, or inaccessible target |

## 6.4 Teachability

Approximate whether the teacher correction lies near student support:

\[
T_j(K)
=
\sum_{v\in\operatorname{TopK}(q_{0,j})}
p_j^\star(v).
\]

A better symmetric candidate-set version uses the union of student and teacher top-\(K\) candidates and measures teacher mass covered by the union.

Low teachability suggests:

- use an intermediate teacher;
- shorten the target horizon;
- retain explicit text;
- exclude from early curriculum;
- increase student capacity rather than loss weight.

## 6.5 Scale coherence

On candidate set \(U\), define:

\[
\delta_{7\rightarrow14}
=
C(\log p_{14})-C(\log p_7),
\]

\[
\delta_{14\rightarrow32}
=
C(\log p_{32})-C(\log p_{14}).
\]

Then:

\[
M
=
\cos
\left(
\delta_{7\rightarrow14},
\delta_{14\rightarrow32}
\right).
\]

Interpretation:

- \(M>0\): coherent directional refinement;
- \(M\approx0\): unrelated or weak changes;
- \(M<0\): reversal or nonmonotonicity.

Scale-specific loop targets should be used only for coherent, verified examples.

## 6.6 Verification

Agreement is not correctness. Use:

- exact arithmetic verification;
- unit tests for code;
- symbolic checks;
- answer-key matching;
- environment/tool outcomes;
- consistency with target-model rollout;
- human or judge review for open-ended cases.

Define verifier confidence:

\[
V\in[0,1].
\]

All scale-ladder weighting should be multiplied by \(V\) or restricted to verified subsets when possible.

## 6.7 Target assignment

### Bucket A — Consensus easy

All teachers and student agree.

- use zero or one loop;
- apply no-op loss to later loops;
- use for calibration and preservation.

### Bucket B — Consensus challenging

Teachers agree; student differs.

- use consensus mixture or 32B target;
- high-priority cumulative KL;
- moderate bounded residual steps.

### Bucket C — 14B/32B convergence

7B differs; 14B and 32B agree.

- strongest scale-bridge examples;
- loop 1 may use 7B or 14B depending teachability;
- final loops use 14B/32B consensus.

### Bucket D — 32B-only emergence

7B and 14B agree; 32B differs.

- require verification;
- treat as advanced curriculum;
- do not assume 32B is correct solely because it is larger.

### Bucket E — All disagree

- exclude initially;
- use soft mixture only if calibrated;
- route to explicit reasoning or stronger verification.

### Initial bucket-by-loop target-assignment table

This table fixes curriculum roles while leaving exact mixture weights and per-horizon target tensors to the E1 lock. The final loop of every included row uses the strongest verified target available.

| Bucket | Loop 1 | Loop 2 | Loops 3–4 | Training role |
|---|---|---|---|---|
| A — consensus easy | consensus target or zero-loop preservation | no-op/preservation target | no-op/preservation target | restraint and calibration |
| B — consensus challenging | verified consensus target | same target with bounded remaining-error step | full verified consensus target | stable distillation |
| C — 14B/32B convergence | teachable 7B or 14B bridge target | 14B target | verified 14B/32B consensus | coherent scale transition |
| D — 32B-only emergence | intermediate verified target when teachable | bounded step toward verified 32B | verified 32B target | advanced curriculum only |
| E — all disagree | excluded | excluded | excluded | explicit reasoning or stronger verification |


---

# 7. AngelSpec integration

## 7.1 What transfers directly

### Shared parameters across depth

Use the same recurrent scratchpad block at every loop. This trains an iterative algorithm rather than a fixed stack of unrelated adapters.

### Depth-specific supervision

Every \((k,j)\) location receives an explicit target. Avoid one terminal loss for all recurrent computation.

### Self-conditioned rollout

At speculative horizon \(j>1\), condition training on the student’s own previous draft tokens for some or all examples:

\[
\widehat y_{t+j-1}\sim q_{k,j-1}.
\]

This reproduces deployment exposure.

### Frozen target

Keep teacher distributions and hidden states detached. The target model must not move toward the drafter.

### Acceptance-aligned schedule

Use a stable cold start before direct overlap optimization.

### Online evaluation

Run genuine speculative decoding against the current checkpoint during training.

## 7.2 What does not transfer directly

### AngelSpec TTT is not latent-loop training

AngelSpec unrolls across future proposed tokens. The scratchpad loops before token emission. Use AngelSpec’s principle of matching deployment-state distributions, but preserve the distinction between \(k\) and \(j\).

### Argmax token feedback is not required inside latent loops

The scratchpad can remain continuous across \(k\). Token rollout is required across \(j\), not necessarily across \(k\).

### DFly is not the scratchpad architecture

DFly’s predecessor-conditioned autoregressive head improves dependencies inside a parallel token block. It suggests adding conditional structure among horizons, not copying block diffusion into the latent loop.

## 7.3 Proposed two-dimensional rollout schedule

For each batch:

1. Compute lower hidden state \(h_0\).
2. Run latent loop \(k=1\).
3. Produce draft distributions \(q_{1,1:J}\).
4. For horizon rollout, feed selected or sampled student tokens into subsequent horizon states.
5. Run latent loop \(k=2\) from the student-created latent state.
6. Recompute or refine \(q_{2,1:J}\).
7. Repeat to \(K\).
8. Run the full upper Qwen suffix from \(h_K\) for final prediction.

To control cost, do not always materialize all \(K\times J\) full-vocabulary logits. Use sparse candidate logits and sample subsets of the grid.

## 7.4 Scheduled rollout mixture

Use a curriculum:

\[
\pi_{\mathrm{TF}}(e)\downarrow
\]

over training epoch \(e\), where \(\pi_{\mathrm{TF}}\) is teacher-forcing probability.

Possible stages:

- 100% teacher-forced horizons;
- 50% teacher / 50% student rollout;
- 100% student greedy rollout;
- stochastic student rollout at deployment temperature.

For latent recurrence, always use the student’s own prior latent state after the initial bridge-alignment stage.

---

# 8. Acceptance-aligned objective

## 8.1 Per-position overlap

For every \(k,j\),

\[
\alpha_{k,j}
=
\sum_v
\min
\left(
p_j^\star(v),
q_{k,j}(v)
\right)
=
1-\operatorname{TV}
\left(
p_j^\star,q_{k,j}
\right).
\]

## 8.2 Negative log-overlap cold start

Define:

\[
\mathcal L_{\mathrm{LK}}
=
-\sum_{k,j}
w_kv_j
\log(\alpha_{k,j}+\epsilon).
\]

The gradient has the same local direction as TV, rescaled by approximately \(1/\alpha\). When overlap is low, it supplies a larger signal than raw TV. This matches AngelSpec’s rationale for cold-start alignment.

Recommended schedule:

1. KL plus CE warm start.
2. KL plus negative log-overlap.
3. End-to-end accepted-length surrogate plus a smaller KL anchor.

Do not begin with pure TV from a poorly aligned checkpoint.

## 8.3 Expected accepted length surrogate

For a fixed loop \(k\), approximate prefix survival:

\[
P_{k,j}
=
\prod_{r=1}^{j}\alpha_{k,r}.
\]

A differentiable expected accepted-token surrogate is:

\[
\widehat A_k
=
\sum_{j=1}^{J}P_{k,j}.
\]

If the target contributes a bonus token after verification, include the corresponding term according to the exact decoder protocol.

Optimize:

\[
\mathcal L_{\mathrm{EAL}}
=
-\widehat A_K.
\]

This multiplicative structure correctly gives early horizons more importance: rejection at position 1 prevents positions 2 through \(J\) from being committed.

Caveat: the product is exact only when each \(\alpha_{k,j}\) is evaluated under the correct conditional prefix distribution. Student-generated on-policy rollout is therefore essential.

## 8.4 Actual serving metric

The real objective is not acceptance alone. Define:

\[
\operatorname{Utility}(K,J;c)
=
\frac{
\mathbb E[\text{committed tokens}\mid K,J,c]
}{
C_{\mathrm{draft}}(K,J,c)
+
C_{\mathrm{verify}}(J,c)
},
\]

where \(c\) includes:

- hardware;
- batch size;
- concurrency;
- context length;
- cache behavior;
- target-model latency.

Report:

- mean accepted length;
- committed length;
- target passes per generated token;
- latency;
- throughput;
- memory;
- break-even committed length.

An additional loop is useful only when:

\[
\Delta\mathbb E[\text{committed tokens}]
>
\Delta C_{\mathrm{loop}}
\times
\text{hardware-specific conversion factor}.
\]

---

# 9. Workload-aware routing

AngelSpec shows that open-ended chat and predictable code/math continuations favor different drafter paradigms. The latent model should not assume one loop policy fits every domain.

## 9.1 Initial workload classes

- open-ended chat;
- mathematical reasoning;
- code generation;
- tool calls;
- structured extraction;
- short factual completion;
- long-context retrieval.

## 9.2 Routing variables

Predict:

- teacher entropy proxy;
- student entropy;
- expected loop gain;
- expected accepted length;
- workload/domain;
- continuation regularity;
- verification cost.

## 9.3 v1 policy

Do not build a complex learned router initially.

Use:

- zero loops for consensus/easy tokens;
- one loop for high-agreement student-gap tokens;
- two to three loops for verified coherent scale-transition tokens;
- explicit reasoning or fallback for low-teachability disagreements.

Train a gain head only after fixed-loop experiments establish real marginal improvements.

---

# 10. Teacher-state and span supervision

Probability distillation alone may produce predictive shortcuts. Use short explicit reasoning spans to supply sequential structure.

## 10.1 Span segmentation

Segment teacher reasoning into short spans \(C_1,\ldots,C_K\), initially:

- 2–4 tokens;
- then 4–8 tokens;
- later 8–16 tokens if successful.

Each span should correspond to a local operation where possible.

## 10.2 Boundary-state target

At insertion layer \(m\), collect teacher state after each span:

\[
T_k
=
h_m^{\mathrm{teacher}}
(x,C_{1:k}).
\]

Project scratch state:

\[
\widehat T_k=P_T(S_k).
\]

Use:

\[
\mathcal L_{\mathrm{state}}
=
\sum_k
\left[
1-\cos(
\widehat T_k,
\operatorname{sg}[T_k]
)
\right]
+
\lambda_{\mathrm{scale}}
\left(
\log\operatorname{RMS}(\widehat T_k)
-
\log\operatorname{RMS}(T_k)
\right)^2.
\]

This anchors trajectory geometry but does not prove reasoning equivalence.

## 10.3 Next-span target

Train each loop to predict the next short span:

\[
\mathcal L_{\mathrm{span}}
=
-\sum_k
\log p_\theta(C_k\mid S_k,h_0).
\]

This provides sequential microstep supervision that the model-size ladder cannot provide.

## 10.4 Text-displacement test

Progressively remove explicit spans and replace them with latent loops.

The strongest evidence of migration is:

\[
\text{same accuracy}
+
\text{less explicit reasoning text}
+
\text{causally necessary scratchpad}.
\]

---


# 11. Teacher canonicalization and the training-only flow hitch

## 11.1 Design goal

The teacher projector should construct a compact state

\[
Z_T=C_T(H_T)\in\mathbb R^{B\times M\times r}
\]

that is:

- predictive of a short future window;
- stable and inexpensive to cache;
- compatible with the student scratchpad shape;
- geometrically well-conditioned for flow and residual losses;
- independent of arbitrary differences in teacher and student width;
- removable with the teacher after training.

The projector should not attempt to preserve every teacher activation. The objective is a **minimal sufficient predictive state**, not high-fidelity teacher reconstruction.

## 11.2 Shared notation

Let selected teacher states be

\[
H_T\in\mathbb R^{B\times N\times L\times d_T},
\]

where:

- \(N\): short future positions or reasoning-span boundaries;
- \(L\): selected teacher layers;
- \(d_T\): teacher hidden width.

The canonical teacher state is

\[
Z_T\in\mathbb R^{B\times M\times r},
\]

and the student scratch trajectory is

\[
Z_{S,0},Z_{S,1},\ldots,Z_{S,K}
\in\mathbb R^{B\times M\times r}.
\]

Recommended v1:

\[
M=8,
\qquad
r=128\text{ or }256,
\qquad
K\le4.
\]

## 11.3 Projector option A — PCA or SVD

Flatten or pool teacher states into \(x\in\mathbb R^D\), center them, and compute

\[
X_c=U\Sigma V^\top.
\]

Retain \(V_r\):

\[
z=(x-\mu)V_r.
\]

PCA solves

\[
V_r
=
\arg\min_{V^\top V=I}
\|X_c-X_cVV^\top\|_F^2.
\]

### Strengths

- closed-form and deterministic;
- highly stable;
- cheap projection and caching;
- clear rank and explained-variance diagnostics;
- linear increments are preserved;
- excellent baseline and initialization.

### Weaknesses

- preserves variance rather than future utility;
- can devote rank to format, style, token identity, or length;
- cannot unfold nonlinear manifolds;
- requires pooling or reshaping choices before decomposition.

### Recommended role

Always include whitened PCA as the first flow baseline and as an initializer or audit for more supervised projectors.

## 11.4 Projector option B — Predictive reduced-rank linear projection

This is the recommended default.

Let \(x\) be a teacher summary and \(y\) a future target, such as:

- compressed future teacher logits;
- future hidden-state deltas;
- answer or operation labels;
- concatenated multi-horizon probe targets.

Learn

\[
z=A^\top x,
\qquad
\widehat y=Bz,
\]

by minimizing

\[
\min_{A,B}
\mathbb E
\left[
\|y-B A^\top x\|_2^2
\right].
\]

Equivalently:

\[
\min_{W:\operatorname{rank}(W)\le r}
\|Y-XW\|_F^2.
\]

A whitened cross-covariance implementation is closely related to supervised PCA, PLS, or CCA depending on normalization:

\[
\widetilde X
=
(X-\mu_X)\Sigma_X^{-1/2},
\]

\[
\widetilde Y
=
(Y-\mu_Y)\Sigma_Y^{-1/2},
\]

\[
C_{XY}
=
\frac{1}{n}\widetilde X^\top\widetilde Y,
\]

\[
C_{XY}=U\Lambda V^\top,
\qquad
z=U_r^\top\widetilde x.
\]

### Strengths

- deterministic and linear;
- directly optimized for future-relevant information;
- preserves increment consistency;
- naturally whitened or easily whitened;
- cheap enough to precompute endpoints;
- supports closed-form or small two-stage optimization;
- provides a clear predictive-rank curve.

### Weaknesses

- target definition strongly shapes retained information;
- remains linear;
- cross-scale target alignment can inherit teacher errors;
- a narrow target can overspecialize the canonical space.

### Recommended role

Use a broad multi-target \(Y\) containing several future horizons and optionally one compressed future-state target. Validate against PCA at matched rank.

## 11.5 Projector option C — Tucker-style multilinear compression

Tucker preserves layer, position, and hidden axes separately:

\[
\mathcal H_T
\approx
\mathcal Q
\times_1 U_N
\times_2 U_L
\times_3 U_H.
\]

A practical projection is

\[
G_T
=
H_T
\times_{\mathrm{layer}}U_L^\top
\times_{\mathrm{hidden}}U_H^\top.
\]

Then resample positions:

\[
Z_T=R_NG_T.
\]

For ranks \(r_L,r_H\), the slot width is

\[
r=r_Lr_H.
\]

LoRi uses a related low-rank tensor subspace and aligns first- and second-order trajectory statistics rather than requiring token-by-token equality.

### Strengths

- preserves layer and hidden structure separately;
- remains linear and flow-friendly;
- handles several teacher layers efficiently;
- supports trajectories with different lengths through moment matching;
- useful global relational regularizer.

### Weaknesses

- unsupervised Tucker still prioritizes reconstruction;
- rank selection occurs across multiple axes;
- covariance matching is coarse and can be expensive at large core rank;
- student and teacher need separate maps into shared core dimensions.

### Recommended role

Use Tucker-initialized layer compression when one teacher layer is insufficient. Follow it with the predictive reduced-rank hidden projection and whitening. Do not use moment matching as the sole objective.

## 11.6 Projector option D — Attention pooling plus predictive bottleneck

Use learned queries to map variable-length teacher states to fixed slots:

\[
Z_T
=
\operatorname{softmax}
\left(
\frac{QK(H_T)^\top}{\sqrt r}
\right)V(H_T).
\]

Then pass through a small deterministic bottleneck and frozen future probe.

### Strengths

- content-dependent selection;
- naturally creates \(M\) student-shaped slots;
- handles variable-length spans and multi-layer tokens;
- can specialize slots to distinct information;
- expressive at moderate parameter cost.

### Weaknesses

- nonlinear and input-dependent;
- projected increments are no longer additive;
- attention assignments can change discontinuously across neighboring states;
- straight-line latent interpolation may leave the learned manifold;
- slot collapse and probe shortcuts require regularization.

### Recommended role

Use only after the predictive linear projector demonstrates insufficient probe fidelity at acceptable rank. If used, retain a linear final bottleneck, whiten outputs, and supervise functional probes along intermediate flow states.

## 11.7 Projector option E — Deterministic autoencoder

Use

\[
z=E_\phi(H_T),
\qquad
\widehat H_T=D_\psi(z),
\]

with a predictive and reconstruction objective:

\[
\mathcal L
=
\mathcal L_{\mathrm{future}}
+
\lambda_{\mathrm{rec}}
\|D(z)-H_T\|^2
+
\lambda_{\mathrm{var}}
\mathcal L_{\mathrm{var}}.
\]

### Strengths

- nonlinear manifold compression;
- deterministic endpoints;
- more flexible than linear maps;
- easier than a VAE.

### Weaknesses

- reconstruction may preserve irrelevant details;
- latent geometry is not automatically Euclidean;
- straight flow paths may be semantically invalid;
- encoder and decoder can hide complexity from the student;
- additional pretraining and regularization.

### Recommended role

Only after linear predictive rank must become too large. Prefer functional decoding over full-state reconstruction.

## 11.8 Projector option F — VAE or conditional VAE

A VAE defines

\[
q_\phi(z\mid H_T)
=
\mathcal N
\left(
\mu_\phi(H_T),
\operatorname{diag}\sigma_\phi^2(H_T)
\right),
\]

with

\[
z=\mu+\sigma\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
\]

The standard objective is

\[
\mathcal L_{\mathrm{VAE}}
=
\mathbb E_q[-\log p_\psi(H_T\mid z)]
+
\beta
\operatorname{KL}(q_\phi(z\mid H_T)\|p(z)).
\]

A conditional predictive VAE decodes future behavior instead of the full teacher state.

### Strengths

- nonlinear and probabilistic;
- explicit uncertainty;
- supports sampling and potentially multimodal extensions;
- can provide a distributional flow endpoint.

### Weaknesses

- posterior collapse;
- KL prior pressure deliberately discards information;
- stochastic gradient variance;
- significantly more hyperparameters;
- a standard Gaussian posterior is not truly multimodal;
- point-to-distribution or distribution-to-distribution flow is more complex;
- using only the posterior mean negates much of the VAE benefit.

### Recommended role

Defer until deterministic targets demonstrably blend distinct valid futures. A mixture or hierarchical latent may ultimately be more relevant than a basic VAE, but only after the simpler system has a proven limitation.

## 11.9 Projector comparison

| Projector | Predictive targeting | Linear increment consistency | Variable-length input | Uncertainty | Flow compatibility | Complexity |
|---|---:|---:|---:|---:|---:|---:|
| whitened PCA | no | exact | requires pooling | no | excellent | very low |
| predictive reduced rank | yes | exact | requires structured pooling | no | excellent | low |
| Tucker + predictive map | yes after augmentation | exact | yes | no | excellent | moderate |
| attention + bottleneck | yes | no | excellent | no | good with controls | moderate |
| deterministic autoencoder | yes | no | yes | no | moderate | moderate-high |
| VAE/CVAE | yes | no | yes | yes | weakest initially | high |

## 11.10 Recommended v1 canonicalizer

Use:

```text
Selected teacher layers and short future boundaries
                    │
                    ▼
           per-layer RMS normalization
                    │
                    ▼
    small learned or fixed linear layer mixture
                    │
                    ▼
 fixed positional resampler to M canonical slots
                    │
                    ▼
 predictive factorized linear projection
            d_T → r_c → M×r
                    │
                    ▼
       fixed truncated whitening transform
                    │
                    ▼
              teacher endpoint Z_T
              ┌─────┴─────┐
              ▼           ▼
        frozen probe   flow target cache
```

Recommended starting values:

\[
r_c=256,
\qquad
M=8,
\qquad
r=128.
\]

The layer mixture can begin as a fixed weighted average of three selected layers and later become a learned simplex weight. Keep the entire canonicalizer affine after RMS normalization in v1.

## 11.11 Canonicalizer training target

Let \(Y\) combine multiple future objectives:

\[
Y
=
\left[
R_{\mathrm{logit}}(p_{T,1:J});
R_{\mathrm{state}}(\Delta h_{T,+J});
R_{\mathrm{answer}}(a_T)
\right].
\]

Here each \(R\) is a fixed low-rank compression or task label. Fit the predictive projection to explain this combined target.

After learning the projection, compute canonical covariance and whiten. Freeze:

- teacher model;
- teacher canonicalizer;
- whitening statistics;
- functional future probe.

## 11.12 Functional probe

A canonical state should preserve function, not merely geometry. Train a modest frozen probe

\[
D_j:Z\rightarrow p_j
\]

for each future horizon or use a shared low-rank probe with horizon embeddings.

The student functional loss is

\[
\mathcal L_{\mathrm{func}}
=
\sum_{k,j}w_kv_j
\operatorname{KL}
\left(
\operatorname{sg}[D_j(Z_T)]
\|D_j(Z_{S,k})
\right).
\]

The probe is frozen but differentiable, so its Jacobian sends gradients into the student. Keep it deliberately smaller than the student recurrent module to prevent a powerful decoder from compensating for a weak latent state.

## 11.13 Training-only hitch

### Teacher branch

The teacher sees privileged short-future spans or verified reasoning boundaries:

\[
H_T
=
F_T(x_{\le t},y_{t+1:t+J}).
\]

Compute

\[
Z_T=C_T(H_T)
\]

under `torch.no_grad()` or from a cache.

### Student branch

The student sees only the inference-available context:

\[
h_0=F_{S,\le m}(x_{\le t}),
\]

\[
Z_{S,0}=I_S(h_0),
\]

\[
Z_{S,k+1}=\Phi_\theta(Z_{S,k},h_0,k).
\]

### Connection

The branches are connected only by losses:

\[
\mathcal L_{\mathrm{flow}},
\quad
\mathcal L_{\mathrm{func}},
\quad
\mathcal L_{\mathrm{rel}},
\quad
\mathcal L_{\mathrm{out}}.
\]

At inference, delete:

- teacher;
- teacher canonicalizer;
- teacher-state cache;
- teacher target probe path.

Retain:

- student lower model;
- scratch initializer;
- recurrent refiner;
- bridge;
- upper model;
- optional student speculative head.

# 12. Flow-consistent recurrent distillation

All teacher endpoints, detached student endpoints, interpolation targets, and flow losses in this section live in the unnormalized partially whitened coordinates. Do not renormalize an interpolated target or the persistent recurrent state per example: that projects the straight affine path onto a sphere and changes the governing geometry. RMSNorm remains permitted only on recurrent-module inputs and innovations.

## 12.1 Why not train only on teacher-interpolated states

A textbook rectified-flow path is

\[
Z_\tau=(1-\tau)Z_{S,0}+\tau Z_T,
\]

with target velocity

\[
v^\star=Z_T-Z_{S,0}.
\]

Training only on \(Z_\tau\) leaks part of the teacher endpoint into the model input. The recurrent student may not learn to recover that direction from its own states at inference.

Use teacher-interpolated flow matching as an optional pretraining or diagnostic, not the main recurrent objective.

## 12.2 Serial self-conditioned residual flow

Let

\[
0=t_0<t_1<\cdots<t_K=1.
\]

Define the fraction of remaining distance:

\[
\beta_k
=
\frac{t_{k+1}-t_k}{1-t_k}.
\]

For example:

\[
t=[0,0.50,0.80,1.0]
\]

gives

\[
\beta=[0.50,0.60,1.0].
\]

At each student-created state, define a detached partial endpoint:

\[
\widetilde Z_{k+1}
=
\mathcal N
\left(
(1-\beta_k)\operatorname{sg}[Z_{S,k}]
+
\beta_k\operatorname{sg}[Z_T]
\right).
\]

Train:

\[
\mathcal L_{\mathrm{flow},k}
=
\operatorname{Huber}
\left(
Z_{S,k+1},
\widetilde Z_{k+1}
\right)
+
\lambda_{\cos}
\left[
1-\cos
\left(
Z_{S,k+1},
\widetilde Z_{k+1}
\right)
\right].
\]

The student is always conditioned on its own prior latent state. The teacher remains only in the target.

## 12.3 Delta-style gated innovation

Use

\[
U_k
=
R_\theta
\left(
\mathcal N(Z_{S,k}),
h_0,e_k
\right),
\]

\[
g_k
=
\sigma
\left(
G_\theta(Z_{S,k},h_0,e_k)
\right),
\]

\[
Z_{S,k+1}
=
\mathcal N
\left(
Z_{S,k}+g_k\odot U_k
\right).
\]

This retains the DeltaNet principle of prediction-error correction without introducing a matrix-valued associative state. A full DeltaNet-style matrix state costs approximately \(O(r^2)\) per mode and adds key-addressing semantics that are not yet justified for \(K\le4\).

## 12.4 Piecewise teacher trajectory

When trustworthy short-span teacher boundaries exist, compute

\[
Z_T^{(0)},Z_T^{(1)},\ldots,Z_T^{(K)}.
\]

A piecewise target is

\[
\widetilde Z_{k+1}
=
(1-\beta_k)\operatorname{sg}[Z_{S,k}]
+
\beta_k\operatorname{sg}[Z_T^{(k+1)}].
\]

This is stronger sequential supervision than one final endpoint, but only valid when span boundaries correspond to meaningful local computation. Compare final-endpoint and piecewise-flow variants explicitly.

## 12.5 Relational trajectory loss

Pool and normalize loop states into rows of \(R_S\) and teacher boundaries into \(R_T\). Define

\[
G_S=R_SR_S^\top,
\qquad
G_T=R_TR_T^\top.
\]

Then

\[
\mathcal L_{\mathrm{rel}}
=
\|G_S-\operatorname{sg}[G_T]\|_F^2.
\]

Alternatively match low-rank mean and covariance as in LoRi. Keep the weight low because relational geometry is non-identifying and can match trajectories with different computations.

## 12.6 Complete state-distillation objective

Use

\[
\begin{aligned}
\mathcal L_{\mathrm{stateKD}}
={}&
\lambda_{\mathrm{flow}}\sum_k\mathcal L_{\mathrm{flow},k}
+
\lambda_{\mathrm{func}}\sum_{k,j}\mathcal L_{\mathrm{func},k,j}\\
&+
\lambda_{\mathrm{rel}}\mathcal L_{\mathrm{rel}}
+
\lambda_{\mathrm{out}}\mathcal L_{\mathrm{output}}
+
\lambda_{\mathrm{preserve}}\mathcal L_{\mathrm{preserve}}.
\end{aligned}
\]

Recommended initial weights:

\[
\lambda_{\mathrm{flow}}=1.0,
\qquad
\lambda_{\mathrm{func}}=0.5,
\qquad
\lambda_{\mathrm{rel}}=0.01,
\qquad
\lambda_{\mathrm{out}}=1.0,
\qquad
\lambda_{\mathrm{preserve}}=0.1.
\]

Do not activate every later speculative and acceptance loss at full strength during the canonical-flow bootstrap.

## 12.7 Gradient flow

Ignoring normalization, the recurrent Jacobian is

\[
J_k
=
I
+
g_k\frac{\partial U_k}{\partial Z_k}
+
U_k\frac{\partial g_k}{\partial Z_k}.
\]

Initialize the gate bias near \(-4\):

\[
g_k\approx0.018.
\]

Then \(J_k\) starts close to identity. With per-loop losses, early states receive both direct and indirect gradients.

Default gradient rules:

- full BPTT through \(K\le4\) student loops;
- no gradient through teacher or canonical endpoint;
- no gradient through target construction from \(Z_{S,k}\);
- frozen but differentiable functional probe on student states;
- no gradient through sampled discrete horizon tokens;
- recurrent and bridge gradient clipping;
- JVP monitoring.

## 12.8 Computational budget

Assume:

- teacher width \(d_T=5120\);
- three selected layers;
- eight future boundaries;
- factor rank \(r_c=256\);
- canonical shape \(8\times128=1024\).

A factorized endpoint projection costs approximately

\[
5120\times256+256\times1024
\approx1.57\text{ million MACs}
\]

per pooled endpoint, negligible relative to teacher inference.

Raw bf16 states for eight positions, three layers, and width 5120 require approximately

\[
8\times3\times5120\times2
=245{,}760\text{ bytes}
\]

per example. The canonical endpoint requires

\[
8\times128\times2=2{,}048\text{ bytes}.
\]

This is roughly a 120-fold storage reduction.

The recurrent student cost is dominated by the low-rank context and bridge projections, not by the elementwise gated flow update.

## 12.9 Core PyTorch implementation

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSUnit(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(
            x.square().mean(dim=-1, keepdim=True) + self.eps
        )


class PredictiveLinearCanonicalizer(nn.Module):
    """
    Affine teacher canonicalizer after selected-state pooling.

    pooled_teacher: [B, d_teacher]
    output: [B, n_slots, latent_dim]
    """

    def __init__(
        self,
        d_teacher: int,
        rank: int = 256,
        n_slots: int = 8,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.latent_dim = latent_dim
        self.register_buffer(
            "projector_weight",
            torch.zeros(d_teacher, n_slots * latent_dim),
        )
        self.register_buffer(
            "teacher_mean",
            torch.zeros(d_teacher),
        )
        self.register_buffer(
            "canonical_mean",
            torch.zeros(n_slots, latent_dim),
        )
        self.register_buffer(
            "whiten_basis",
            torch.eye(latent_dim),
        )
        self.register_buffer(
            "whiten_eigenvalues",
            torch.ones(latent_dim),
        )
        self.register_buffer(
            "whiten_alpha",
            torch.tensor(0.5),
        )

    def forward(self, pooled_teacher: torch.Tensor) -> torch.Tensor:
        z = (pooled_teacher - self.teacher_mean) @ self.projector_weight
        z = z.view(-1, self.n_slots, self.latent_dim)
        z = z - self.canonical_mean
        scale = self.whiten_eigenvalues.pow(-0.5 * self.whiten_alpha)
        return (z @ self.whiten_basis) * scale


class SharedResidualFlow(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        max_steps: int = 4,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.norm = RMSUnit()
        self.context_proj = nn.Linear(
            context_dim, latent_dim, bias=False
        )
        self.step_embedding = nn.Embedding(
            max_steps, latent_dim
        )
        input_dim = latent_dim * 3
        hidden_dim = latent_dim * expansion
        self.delta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.gate_net = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.gate_net.weight)
        nn.init.constant_(self.gate_net.bias, -4.0)
        nn.init.normal_(self.delta_net[-1].weight, std=1e-3)
        nn.init.zeros_(self.delta_net[-1].bias)

    def forward(
        self,
        z0: torch.Tensor,
        context: torch.Tensor,
        n_steps: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if not 1 <= n_steps <= self.max_steps:
            raise ValueError("invalid n_steps")
        batch, slots, latent_dim = z0.shape
        context_z = self.context_proj(context)
        context_z = context_z[:, None, :].expand(
            batch, slots, latent_dim
        )
        z = z0
        states = [z]
        updates = []
        gates = []
        for step in range(n_steps):
            step_ids = torch.full(
                (batch,), step, dtype=torch.long, device=z.device
            )
            step_z = self.step_embedding(step_ids)
            step_z = step_z[:, None, :].expand_as(z)
            features = torch.cat(
                [self.norm(z), context_z, step_z], dim=-1
            )
            raw_delta = self.delta_net(features)
            gate = torch.sigmoid(self.gate_net(features))
            update = gate * raw_delta
            z = z + update
            states.append(z)
            updates.append(update)
            gates.append(gate)
        return states, updates, gates


def serial_flow_loss(
    states: Sequence[torch.Tensor],
    teacher_endpoint: torch.Tensor,
    betas: Sequence[float],
    cosine_weight: float = 0.1,
) -> torch.Tensor:
    teacher = teacher_endpoint.detach()
    losses = []
    for k, beta in enumerate(betas):
        previous = states[k].detach()
        target = (1.0 - beta) * previous + beta * teacher
        huber = F.smooth_l1_loss(states[k + 1], target)
        cosine = (
            1.0
            - F.cosine_similarity(
                states[k + 1], target, dim=-1
            ).mean()
        )
        losses.append(huber + cosine_weight * cosine)
    return torch.stack(losses).mean()


def trajectory_relation_loss(
    student_states: Sequence[torch.Tensor],
    teacher_states: Sequence[torch.Tensor],
) -> torch.Tensor:
    student = torch.stack(
        [s.mean(dim=1) for s in student_states], dim=1
    )
    teacher = torch.stack(
        [s.mean(dim=1) for s in teacher_states], dim=1
    ).detach()
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher, dim=-1)
    return F.mse_loss(
        student @ student.transpose(-1, -2),
        teacher @ teacher.transpose(-1, -2),
    )
```

## 12.10 Training-hitch pseudocode

```python
# Teacher branch: privileged training information only.
with torch.no_grad():
    teacher_hidden = teacher.forward_selected_states(
        teacher_input_ids,
        future_span_ids,
    )
    pooled_teacher = pool_selected_states(teacher_hidden)
    z_teacher = teacher_canonicalizer(pooled_teacher)
    teacher_probe_logits = frozen_probe(z_teacher)

# Student branch: identical to inference path.
lower_hidden = student.forward_lower(student_input_ids)
context = lower_hidden.mean(dim=1)
z0 = student.initialize_scratch(lower_hidden)
states, updates, gates = refiner(z0, context, n_steps)

flow_loss = serial_flow_loss(
    states,
    z_teacher,
    betas=[0.5, 0.6, 1.0][:n_steps],
)

functional_losses = []
for z_student in states[1:]:
    student_probe_logits = frozen_probe(z_student)
    functional_losses.append(
        F.kl_div(
            F.log_softmax(student_probe_logits, dim=-1),
            F.softmax(teacher_probe_logits, dim=-1),
            reduction="batchmean",
        )
    )
functional_loss = torch.stack(functional_losses).mean()

bridged_hidden = student.bridge_scratch(
    lower_hidden, states[-1]
)
student_logits = student.forward_upper(bridged_hidden)
output_loss = causal_lm_loss(student_logits, labels)

total_loss = (
    output_loss
    + 1.0 * flow_loss
    + 0.5 * functional_loss
)
```

## 12.11 Inference separation

```text
TRAINING
Teacher + future span ─► canonical endpoint ─┐
                                             ├─ losses ─► student gradients
Student context ─► recurrent scratchpad ─────┘

INFERENCE
Student context ─► recurrent scratchpad ─► bridge ─► upper Qwen ─► output
```

The student never receives the teacher endpoint as an input. This is essential to prevent target leakage and guarantee deployability.

---

# 13. Stability and gradient control

## 13.1 Hidden-scale blind spot

RMSNorm can make cross-entropy relatively insensitive to radial state drift. Add:

\[
\mathcal L_{\mathrm{stateScale}}
=
\sum_k
\left[
\log
\frac{
\operatorname{RMS}(h_k)
}{
\operatorname{RMS}(h_0)
}
\right]^2.
\]

Add update-ratio control:

\[
\mathcal L_{\mathrm{updateScale}}
=
\sum_k
\left[
\frac{
\operatorname{RMS}(g_k\Delta h_k)
}{
\operatorname{RMS}(h_0)
}
-
\mu
\right]^2,
\]

with \(\mu\approx0.01-0.05\) initially.

## 13.2 Jacobian monitoring

Track JVP gain:

\[
G_k(v)
=
\frac{\|J_kv\|}{\|v\|}.
\]

Also track finite-horizon gain:

\[
G_{1:K}(v)
=
\frac{
\left\|
\frac{\partial z_K}{\partial z_0}v
\right\|
}{
\|v\|
}.
\]

Do not rely only on spectral radius. Non-normal Jacobians can show large transient amplification even if all eigenvalues lie inside the unit circle.

## 13.3 Gradient monitoring

Log gradients by:

- recurrent block;
- bridge;
- delta readout;
- scratch initializer;
- nearby LoRA;
- each loss component;
- each loop.

Keep major gradient contributions within roughly one order of magnitude during initial training.

## 13.4 Truncation and detach ablations

Compare:

1. full backpropagation through all loops;
2. stop-gradient between loops;
3. truncated BPTT after two loops;
4. detach previous logits only for residual target construction.

Expected default:

- full gradient through scratch recurrence;
- detached teacher;
- detached previous logits inside the auxiliary residual target;
- no gradient through discrete sampled tokens.

---

# 14. Sparse teacher-logit representation

Full-vocabulary storage is prohibitive.

For vocabulary \(V\approx152{,}000\), three fp16 teacher vectors require roughly:

\[
152{,}000\times3\times2
\approx912{,}000
\text{ bytes per token}.
\]

Use:

1. union of top-\(K\) teacher/student token IDs;
2. \(K=64\) or \(128\);
3. stored log probabilities for each model;
4. one aggregated tail bucket;
5. optional exact full-logit audit subset.

Validate:

- KL approximation error;
- TV approximation error;
- acceptance prediction error;
- delta-direction cosine;
- teacher-mass coverage.

Top-\(K\) truncation must not silently renormalize away large tail mass.

---

# 15. Recommended total objective

Use:

\[
\begin{aligned}
\mathcal L
={}&
\lambda_{\mathrm{final}}\mathcal L_{\mathrm{final}}
+
\lambda_{\mathrm{canonFlow}}\mathcal L_{\mathrm{stateKD}}
+
\lambda_{\mathrm{cum}}\mathcal L_{\mathrm{cum}}
+
\lambda_{\mathrm{bridge}}\mathcal L_{\mathrm{bridge}}
+
\lambda_{\mathrm{progress}}\mathcal L_{\mathrm{progress}}\\
&+
\lambda_{\mathrm{LK}}\mathcal L_{\mathrm{LK}}
+
\lambda_{\mathrm{EAL}}\mathcal L_{\mathrm{EAL}}
+
\lambda_{\mathrm{local}}\mathcal L_{\mathrm{local}}
+
\lambda_{\mathrm{span}}\mathcal L_{\mathrm{span}}\\
&+
\lambda_{\mathrm{state}}\mathcal L_{\mathrm{state}}
+
\lambda_{\mathrm{scale}}\mathcal L_{\mathrm{stateScale}}
+
\lambda_{\mathrm{update}}\mathcal L_{\mathrm{updateScale}}
+
\lambda_{\mathrm{preserve}}\mathcal L_{\mathrm{preserve}}\\
&+
\lambda_{\mathrm{noop}}\mathcal L_{\mathrm{noop}}
+
\lambda_{\mathrm{deltaDiag}}\mathcal L_{\mathrm{deltaDiag}}.
\end{aligned}
\]

Do not activate all terms at full weight simultaneously.

## 15.1 Suggested phases

Staged adaptation is binding. Train the new initializer, canonical flow, bridge, control, and residual heads first while the pretrained backbone and upper layers remain frozen. Upper-layer unfreezing or LoRA is trigger-held for E4 evidence and requires a separate lock; it is not part of the initial A–E path.


### Phase 0 — Canonicalizer and probe preparation

- collect selected teacher states and future targets;
- fit whitened PCA and predictive reduced-rank projectors;
- evaluate rank–fidelity and interpolation curves;
- train a modest future functional probe;
- freeze the selected canonicalizer, whitening transform, and probe;
- cache canonical endpoints for the bootstrap dataset.

Do not begin recurrent flow training until the canonical space passes conditioning, predictive-fidelity, and intermediate-path audits.

### Phase A — Bridge alignment

- final CE;
- preservation KL;
- state alignment;
- cumulative KL;
- one loop.

### Phase B — Stable drafter

- cumulative KL;
- local CE;
- negative log-overlap;
- loops 1–2;
- teacher-forced horizons.

### Phase C — Recurrent refinement

- bounded bridge targets;
- progress loss;
- loops 1–4;
- mixed teacher/student horizon rollout;
- consensus no-op.

### Phase D — Acceptance optimization

- end-to-end accepted-length surrogate;
- lower-weight KL anchor;
- fully student-generated horizon rollout;
- genuine online verification.

### Phase E — Span migration

- next-span and boundary-state targets;
- progressively removed explicit spans;
- causal scratchpad evaluation.

---

# 16. PyTorch-oriented design

## 16.1 Module boundaries

```text
QwenSplitBackbone
    lower_layers
    upper_layers

ScratchpadInitializer
    learned latent slots
    prompt-conditioned initialization

SharedRecurrentScratchpad
    RMSNorm
    loop embedding
    cached-context attention
    normalized gated update

AnchoredBridge
    scratch-to-token cross-attention
    update normalization
    pretrained anchor path

ResidualDraftHead
    horizon embeddings
    low-rank innovation projection
    cumulative logits
    write gates

HorizonRollout
    teacher/student token mixture
    depth-indexed cache
    validity masks
    document isolation

TeacherLattice
    agreement
    student gap
    teachability
    scale coherence
    verifier confidence
    curriculum bucket

TeacherCanonicalizer
    selected-layer normalization and pooling
    predictive reduced-rank projection
    fixed whitening transform
    endpoint caching

FrozenFunctionalProbe
    multi-horizon future prediction
    small fixed decoder
    functional student gradients

CanonicalFlowHitch
    teacher endpoint target construction
    serial partial-flow targets
    relational trajectory supervision
    training-only lifecycle


AcceptanceObjective
    KL
    TV overlap
    negative log-overlap
    accepted-prefix surrogate

DynamicsMonitor
    state norms
    update ratios
    loop gradients
    JVP gain
    causal interventions

LoopGainController
    predicted marginal acceptance gain
    optional halt policy
```

## 16.2 Forward interface

```python
out = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    n_loops=n_loops,
    n_horizons=n_horizons,
    rollout_mode=rollout_mode,
    teacher_scale_logits=teacher_scale_logits,
    teacher_candidate_ids=teacher_candidate_ids,
    teacher_tail_mass=teacher_tail_mass,
    teacher_states=teacher_states,
    teacher_spans=teacher_spans,
    verifier_confidence=verifier_confidence,
    intervention=intervention,
    return_loop_states=True,
    return_draft_grid=True,
)
```

Outputs:

```text
final_logits
draft_logits[K, J]
cumulative_logits[K, J]
delta_logits[K, J]
write_gates[K, J]
scratch_states[K]
bridge_states[K]
teacher_lattice_metrics
acceptance_surrogates[K]
dynamics_metrics
loss_components
```

## 16.3 Core residual target pseudocode

```python
def geometric_bridge_target(
    prev_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    target_logits = (
        (1.0 - beta) * prev_log_probs.detach()
        + beta * teacher_log_probs.detach()
    )
    return target_logits - torch.logsumexp(
        target_logits,
        dim=-1,
        keepdim=True,
    )


bridge_log_target = geometric_bridge_target(
    prev_log_q,
    teacher_log_p,
    beta,
)

bridge_kl = F.kl_div(
    curr_log_q,
    bridge_log_target.exp(),
    reduction="batchmean",
)

full_teacher_kl = F.kl_div(
    curr_log_q,
    teacher_log_p.exp(),
    reduction="batchmean",
)
```

## 16.4 Acceptance surrogate pseudocode

```python
def overlap(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.minimum(p, q).sum(dim=-1)

alpha = overlap(teacher_probs, student_probs)  # [B, K, J]

prefix_survival = torch.cumprod(alpha, dim=-1)
expected_accepted = prefix_survival.sum(dim=-1)

lk_loss = -torch.log(alpha.clamp_min(1e-6)).mean()
eal_loss = -expected_accepted.mean()
```

For correctness, teacher and student distributions must be conditional on the corresponding realized or rolled-out prefixes.

---

# 17. Training data strategy

## 17.1 Teacher cascade

To limit labeling cost:

1. run 7B broadly;
2. query 14B when student gap, 7B entropy, or task importance exceeds threshold;
3. query 32B when 7B/14B disagree, verification is available, or the example is selected for advanced curriculum;
4. run target rollouts in the actual serving format;
5. periodically relabel student-generated prefixes.

## 17.2 Dataset mixture

Maintain separate strata:

- consensus easy;
- consensus challenging;
- coherent scale transition;
- verified 32B-only emergence;
- ambiguous;
- short explicit reasoning spans;
- long-context examples;
- code/math predictable continuations;
- open-ended high-entropy continuations.

Do not train one uniform mixture. Sample according to current failure modes and deployment workload.

## 17.3 Document isolation

All packing, horizon shifts, and rollouts must respect document boundaries. No speculative or teacher target may cross from one packed sequence into another.

---

# 18. Causal-use tests

A useful scratchpad must matter at inference.

Run:

1. zero scratchpad;
2. random norm-matched scratchpad;
3. cross-example permutation;
4. stale previous-loop state;
5. loop-order permutation;
6. skip recurrent block;
7. zero bridge only;
8. zero predictive residual head only;
9. remove one slot at a time;
10. replace recurrent block with compute-matched feed-forward adapter;
11. replace removed reasoning span with pause tokens;
12. retain text but remove latent loop.

Interpretation:

- If gains remain when scratch content is permuted, the benefit is probably in updated weights or extra depth.
- If the auxiliary draft improves but final upper-Qwen output does not, the scratchpad may be an isolated prediction head.
- If pause tokens match recurrence, extra depth rather than latent content may explain the gain.
- If removed explicit text can be restored only by the correct scratch state, that is strong evidence of migration.

---

# 19. Metrics

## 19.1 Task quality

- exact match;
- pass@1;
- code unit-test success;
- calibrated answer probability;
- zero-loop versus multi-loop performance;
- OOD performance;
- explicit-text displacement curve.

## 19.2 Cross-scale metrics

- teacher JS divergence;
- argmax agreement;
- student gap;
- teachability;
- scale coherence;
- verifier-conditioned teacher accuracy;
- target bucket proportions.

## 19.3 Loop metrics

- KL to final target by loop;
- fractional KL reduction;
- bridge-target KL;
- progress-loss violation;
- delta-direction cosine;
- write gate;
- no-op update norm;
- final upper-tail improvement by loop.

## 19.4 Speculative metrics

- per-horizon overlap;
- per-horizon actual acceptance;
- mean accepted length;
- committed length;
- target passes per generated token;
- throughput;
- latency;
- break-even committed length;
- acceptance under deployment temperature.

## 19.5 Dynamics

- scratch RMS;
- bridge RMS;
- update/base ratio;
- JVP gain;
- finite-loop perturbation gain;
- gradient norm by loop and loss;
- inter-loop cosine;
- slot specialization.

---

# 20. Ablation matrix

| ID | Comparison | Question |
|---|---|---|
| A0 | frozen Qwen baseline | What is the starting point? |
| A1 | draft heads only | Does MTP explain the gains? |
| A2 | one recurrent loop | Is the bridge trainable? |
| A3 | terminal loss vs per-loop loss | Does deep supervision improve gradients? |
| A4 | independent loop heads vs cumulative residual logits | Does residual accumulation help? |
| A5 | cumulative KL vs geometric bridge target | Does bounded refinement help? |
| A6 | bridge target vs explicit logit-delta cosine | Is delta-vector supervision necessary? |
| A7 | teacher-forced vs mixed vs student rollout | Does exposure matching improve deeper horizons? |
| A8 | KL vs CE vs LK cold start | Which stable objective best initializes overlap? |
| A9 | LK only vs LK then EAL/TV | Does acceptance-aligned staging help? |
| A10 | final 32B at every loop vs scale ladder | Does scale curriculum specialize loops? |
| A11 | raw disagreement vs teachability filtering | Are corrections accessible? |
| A12 | agreement vs verifier-filtered agreement | Is shared family bias material? |
| A13 | consensus no-op on/off | Does the model learn restraint? |
| A14 | recurrence vs FLOP-matched feed-forward | Does recurrence matter? |
| A15 | recurrence vs pause tokens | Does latent content matter beyond extra positions? |
| A16 | layer 8 vs 12 vs 16 | Where is reentry most learnable? |
| A17 | 4 vs 8 vs 16 slots | How much state is needed? |
| A18 | shared recurrent block vs untied blocks | Is iterative parameter sharing useful? |
| A19 | one-dimensional \(k\) supervision vs full \(k\times j\) grid | Is the two-axis formulation necessary? |
| A20 | auxiliary draft improvement vs full upper-tail rerun | Does improvement transfer to the actual model? |
| A21 | span state alignment on/off | Does sequential geometry reduce shortcuts? |
| A22 | 2/4/8/16-token span removal | How much explicit computation can migrate? |
| A23 | fixed loops vs adaptive loops | Is marginal-gain routing useful? |
| A24 | top-64 vs top-128 vs full logits | Is sparse storage adequate? |
| A25 | full BPTT vs detached loops | How much cross-loop credit assignment is needed? |
| A26 | whitened PCA vs predictive reduced rank | Does supervision improve the canonical subspace? |
| A27 | predictive linear vs Tucker-initialized predictive | Does multi-layer structure add value? |
| A28 | predictive linear vs attention pooling | Is content-dependent pooling necessary? |
| A29 | deterministic bottleneck vs VAE mean | Does probabilistic encoding add value? |
| A30 | teacher endpoint only vs frozen functional probe | Is geometry alone sufficient? |
| A31 | teacher-interpolated FM vs serial self-conditioned flow | Does target leakage impair inference states? |
| A32 | final endpoint vs piecewise span endpoints | Are trustworthy intermediates helpful? |
| A33 | full DeltaNet matrix state vs gated residual slots | Is associative memory worth the added cost? |
| A34 | cached canonical endpoints vs online teacher hitch | What labeling/storage tradeoff is optimal? |

---

# 21. Go/no-go criteria

## 21.1 Continue if

1. The selected canonicalizer improves future-probe fidelity over whitened PCA at a comparable usable rank, or PCA is retained as the simpler winner.
2. Canonical endpoints are numerically conditioned and functional-probe outputs remain smooth along partial interpolation paths.
3. Zero-loop output remains close to pretrained Qwen.
4. Multi-loop predictions improve cumulative target KL beyond draft-head-only and feed-forward controls.
5. Later speculative horizons improve under student-generated rollout.
6. Actual accepted length improves, not only proxy loss.
7. Improvement exceeds loop latency at the deployment operating point.
8. Scratchpad permutation or bridge ablation removes a meaningful portion of the gain.
9. Final upper-Qwen output improves with the scratchpad.
10. Consensus examples learn smaller late-loop updates.
11. At least one short explicit span can be removed at matched performance.
12. OOD performance does not collapse.

## 21.2 Simplify or stop if

1. No canonicalizer improves predictive utility over PCA at a rank small enough to be useful, suggesting the teacher-state hitch is unnecessary or too lossy.
2. Linear interpolation in the chosen canonical space produces unstable or functionally invalid intermediates that the serial flow cannot correct cheaply.
3. Auxiliary heads improve while final output does not.
4. Recurrent gains disappear against compute-matched feed-forward depth.
5. Scratchpad content interventions have little effect.
6. Teacher disagreement weighting adds no value beyond simple final-teacher KL.
7. Scale-ladder targets are strongly nonmonotonic or unverified.
8. Additional loops reduce acceptance or final quality.
9. Hardware break-even is not reached.
10. Norm or Jacobian instability persists after anchored updates.
11. Explicit span removal fails even on short structured traces.
12. Sparse teacher targets materially distort acceptance estimates.

---

# 22. Phased experiment plan


## Experiment 0A — Canonicalizer audit

Using frozen teacher states, compare at matched canonical rank:

1. whitened PCA;
2. predictive reduced-rank linear projection;
3. Tucker-initialized predictive projection;
4. attention pooling plus deterministic bottleneck;
5. deterministic autoencoder.

Do not include a VAE unless deterministic models demonstrate multimodal averaging failure.

Measure:

- future-probe KL;
- future-state reconstruction or direction;
- held-out task prediction;
- canonical covariance conditioning;
- interpolation probe fidelity;
- storage and projection cost;
- rank required to reach a fixed predictive threshold.

Pass criterion: predictive reduced rank should improve future-probe fidelity over PCA at similar rank without materially worse conditioning or cost.

## Experiment 0B — Flow-path validity

For each canonicalizer, evaluate intermediate points

\[
Z_\tau=(1-\tau)Z_0+\tau Z_T
\]

using the frozen functional probe.

Measure:

- smoothness of probe logits over \(\tau\);
- monotonic improvement toward teacher target;
- curvature under local Jacobian estimates;
- off-manifold degradation;
- student serial-flow trainability.

Pass criterion: the chosen space supports stable partial targets and does not require a high-capacity nonlinear flow merely to connect nearby endpoints.



## Experiment 1 — Identity and bridge audit

Build:

- insertion at layer 12;
- 8 slots × 256;
- one loop;
- anchored bridge;
- frozen Qwen;
- final CE and state alignment.

Pass:

- negligible zero-loop drift;
- bridge update 1–5% of hidden RMS;
- stable gradients;
- no task regression.

## Experiment 2 — Two-axis drafter baseline

Add:

- \(K=2\);
- \(J=4\);
- draft grid \(q_{k,j}\);
- teacher-forced horizons;
- cumulative KL.

Compare independent versus cumulative residual logits.

Pass:

- both loops receive gradient;
- loop 2 improves at least one horizon;
- final upper-tail output tracks auxiliary improvement.

## Experiment 3 — AngelSpec-style rollout and acceptance

Add:

- mixed then student-generated horizon rollout;
- KL/LK warm start;
- online speculative evaluator.

Pass:

- later-horizon actual acceptance improves;
- no major first-horizon regression;
- accepted length exceeds draft-head-only baseline.

## Experiment 4 — Bounded residual refinement

Compare:

- final target at every loop;
- geometric bridge targets;
- progress loss;
- explicit delta-direction auxiliary.

Pass:

- geometric bridge or progress objective improves cumulative refinement or stability;
- delta-direction auxiliary retained only if it improves real metrics.

## Experiment 5 — Cross-scale teacher lattice

Generate 7B/14B/32B labels and compute:

- agreement;
- student gap;
- teachability;
- coherence;
- verification.

First test on a nonrecurrent distillation adapter.

Pass:

- selected examples improve held-out target KL or task quality more efficiently than random or raw-KL selection.

## Experiment 6 — Scale-conditioned loops

Compare:

- 32B every loop;
- 7B/14B/32B ladder;
- verified mixture;
- teachability-filtered ladder;
- consensus no-op.

Pass:

- scale conditioning improves real cumulative refinement, not merely teacher-specific auxiliary alignment.

## Experiment 7 — Explicit-to-latent span migration

Train short teacher spans and progressively remove text.

Pass:

- a latent loop replaces at least one short span at matched task quality;
- correct scratch content is causally necessary.

## Experiment 8 — Adaptive compute

Train marginal-gain controller using measured benefit of loop \(k+1\).

Pass:

- equal or better quality at lower average loop count;
- realized hardware utility improves.

---

# 23. Strategy-agent handoff

## 23.1 Mission

The strategy agent chooses the smallest experiment that distinguishes competing explanations. It must not maximize architectural novelty.

The agent should determine whether observed improvement comes from:

- recurrent latent computation;
- ordinary extra depth;
- speculative-head distillation;
- teacher selection;
- altered backbone weights;
- explicit scratchpad content;
- workload-specific continuation predictability.

## 23.2 Required response for every research cycle

The strategy agent must provide:

1. current diagnosis;
2. mathematical mechanism;
3. evidence;
4. competing hypotheses;
5. smallest discriminating experiment;
6. exact control;
7. expected result under each hypothesis;
8. success threshold;
9. falsification threshold;
10. compute and memory estimate;
11. implementation scope;
12. rollback plan.

## 23.3 Operating prompt

> You are the strategy agent for a Qwen2.5-0.5B latent-microstep research program.
>
> The architecture contains a fixed-size recurrent scratchpad inserted into a middle layer, an anchored residual bridge, cumulative multi-horizon draft heads, and a frozen or slowly adapted upper Qwen suffix.
>
> Treat latent refinement depth \(k\) and speculative token horizon \(j\) as separate axes. Never describe loop \(k\) as token \(t+k\) unless the architecture explicitly makes that mapping.
>
> AngelSpec supports shared-parameter rollout, depth-specific future-token targets, self-generated prefix training, acceptance-aligned objectives, online acceptance evaluation, and workload-aware deployment. It does not establish that model scale is a reasoning trajectory.
>
> Treat Qwen2.5-7B, 14B, and 32B as a capability and uncertainty lattice. Separate agreement, student gap, teachability, scale coherence, and verifier confidence. Do not select examples solely because disagreement is large.
>
> The default teacher canonical space is a whitened predictive reduced-rank linear projection. PCA is the mandatory baseline; Tucker is the structured multi-layer extension; attention pooling and autoencoders require evidence of linear underfitting; VAEs require evidence of harmful deterministic averaging.
>
> The teacher is connected to the student only through stopped-gradient canonical endpoints, frozen probe targets, and loss functions. Never feed the teacher endpoint into the student recurrent forward path.
>
> Rectified-flow interpolation is a geometric target, not proof that intermediate states are valid. Prefer serial self-conditioned flow on student-created states and verify intermediate points with a frozen functional probe.
>
> The primary predictive objective is cumulative teacher-distribution matching. Bounded geometric bridge targets and progress losses may shape successive loops. Explicit logit-delta direction is an auxiliary diagnostic unless an ablation proves incremental value.
>
> Do not infer latent reasoning from improved loss alone. Require:
>
> - improvement in the actual upper-model output;
> - genuine speculative acceptance gains;
> - scratchpad causal-use tests;
> - compute-matched nonrecurrent controls;
> - explicit-span displacement for migration claims.
>
> For every recommendation, state what result would falsify it.
>
> Do not add DeltaNet matrix memory, FFT oscillators, adaptive halting, or complex routing until the basic recurrent bridge passes identity, acceptance, causal-use, and compute-break-even tests.

---

# 24. Final mathematical judgment

The overall program is sound if it is interpreted as **iterative predictive refinement under constrained compute**. The preferred teacher–student interface is a deterministic, whitened, predictive low-rank canonical state. This interface preserves tractable Euclidean geometry and linear increments while filtering the teacher representation toward short-horizon utility.

The most rigorous formulation is:

\[
\boxed{
\begin{aligned}
z_{k+1}
&=
\Phi_\theta(z_k,h_0),\\
q_{k,j}
&=
q_\theta(y_{t+j}\mid z_k,\widehat y_{<t+j}),\\
r_{k,j}^{(\beta)}
&\propto
q_{k-1,j}^{1-\beta}
(p_{k,j}^\star)^\beta,\\
\mathcal L
&=
\mathcal L_{\mathrm{final}}
+
\mathcal L_{\mathrm{cum}}
+
\mathcal L_{\mathrm{bridge}}
+
\mathcal L_{\mathrm{accept}}
+
\mathcal L_{\mathrm{stability}}.
\end{aligned}
}
\]

The cross-scale teacher lattice informs target reliability and curriculum. It should not be mistaken for an internal reasoning trace.

The DeltaNet analogy is useful at the level of **error-correcting residual updates**, but not at the level of memory architecture.

AngelSpec provides strong evidence that the speculative side must be trained on its own rollout distribution and judged by actual accepted progress under serving costs.

The project earns the label “latent microstep reasoning” only when the recurrent state is:

- predictively useful;
- sequentially refining;
- causally necessary;
- stable;
- able to displace explicit computation;
- more efficient than simpler alternatives.

---

# 25. Primary references

1. Liu H, Cen R, Shi J, et al. **AngelSpec: Towards Real-World High Performance Inference with Speculative Decoding.** arXiv:2607.25852, 2026.
2. Tencent. **AngelSpec: A unified, torch-native training framework for speculative-decoding draft models.** GitHub, released July 29, 2026.
3. Yang S, Wang B, Zhang Y, Shen Y, Kim Y. **Parallelizing Linear Transformers with the Delta Rule over Sequence Length.** arXiv:2406.06484, 2024.
4. Yang S, Kautz J, Hatamizadeh A. **Gated Delta Networks: Improving Mamba2 with Delta Rule.** arXiv:2412.06464, 2024.
5. Yang A, Yang B, Zhang B, et al. **Qwen2.5 Technical Report.** arXiv:2412.15115, 2024/2025.
6. Qwen Team. **Qwen2.5: A Party of Foundation Models.** Official release, September 2024.
7. Leviathan Y, Kalman M, Matias Y. **Fast Inference from Transformers via Speculative Decoding.** ICML, 2023.
8. Chen C, Borgeaud S, Irving G, et al. **Accelerating Large Language Model Decoding with Speculative Sampling.** arXiv:2302.01318, 2023.
9. Gloeckle F, Idrissi BY, Rozière B, et al. **Better & Faster Large Language Models via Multi-Token Prediction.** arXiv:2404.19737, 2024.
10. Li Y, Wei F, Zhang C, Zhang H. **EAGLE-3: Scaling Up Inference Acceleration of Large Language Models via Training-Time Test.** arXiv:2503.01840, 2025.
11. Samarin A, Krutikov S, Shevtsov A, et al. **LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding.** arXiv:2602.23881, 2026.

12. Shao S, Shen Z, Gong L, Chen H, Dai X. **Precise Knowledge Transfer via Flow Matching.** arXiv:2402.02012, 2024.
13. Solgi R, Tian J, Zhang Z. **LoRi: Low-Rank Distillation for Implicit Reasoning.** arXiv:2606.05315, 2026.
14. Brown R, Russell C. **Task-Specific Knowledge Distillation via Intermediate Probes.** arXiv:2603.12270, 2026.
15. **KDFlow: A User-Friendly and Efficient Knowledge Distillation Framework for Large Language Models.** arXiv:2603.01875, 2026.
