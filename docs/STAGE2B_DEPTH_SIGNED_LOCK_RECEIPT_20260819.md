# Stage 2B-D Signed Lock Receipt

Date: 2026-08-19

## Authority

- Signature: `Approved by Mark, 2026-08-19, via strategy pre-signature review`
- Executed lock path: `training/paper2_stage2b_depth_executed_lock.json`
- Executed lock SHA-256: `074dc564190a050209925d42caa8f0d8a24b901342df11c2efca54cad655eb17`
- Formal approval Drive ID: `1qUJ-ZaW5W_c1aLRxf4_H70ggsE8lJtWD`
- Formal approval canonical Drive bytes: 7,724
- Formal approval canonical Drive SHA-256: `e203fe92d903615827cfeb58cf185913bdbca7a63aa946e906913a495c24f095`
- Repository mirror bytes: 7,725 (one repository trailing newline)
- Repository mirror SHA-256: `cd64ab6041a462d00b33694a83d8989fc221c6a1f105a59b1bdafe48689b9367`
- Pre-signature review Drive ID: `1m2IRYaNZSp3R5Zq0J_MsglKd5Z0CAxuo`
- Pre-signature review SHA-256: `e53f1e856d4816c4798dd48218f607e2aab2d93a3f8f19dac80ec9c91560e8fd`

The canonical Drive SHA and repository-mirror SHA differ only because the
repository text convention appends one newline. The approval content is
otherwise byte-identical.

## Binding Amendments

1. DEV-1 retains Tier-1 and GSM8K safety-floor duty, both-comparator battery
   decomposition, and the continuity read. DEV-2 has no safety-floor duty.
2. The registered discrete endpoint is pooled DEV-1 plus DEV-2, 3,072 rows.
3. A single 5,000-to-8,000-step deferral is available only when both seeds fail
   separation at step 5,000 and both registered transition slopes are positive.
4. Every look reports loop-1 KL, the finite-horizon gain trajectory, and the R-2
   desk read.

CONFIRM and EVAL-E remain sealed.

## Pooled Endpoint Power

The 1.5-point design target is represented by +46 net rows on 3,072 rows
(1.4973958333 points), with a one-sided exact paired-sign test at alpha 0.05.

| Assumed discordance | Power |
|---:|---:|
| 0.10 | 0.8222548193 |
| 0.20 | 0.5805480451 |
| 0.30 | 0.4347383426 |

The discrete endpoint is descriptive/confirmatory context. The registered
step-5,000 campaign decision instrument remains the row-paired answer-token
margin transition read.

## Deferral Estimator

- Looks: 3, 4, and 5 (steps 3,000, 4,000, and 5,000)
- Transitions: K2 to K3 and K3 to K4
- Estimator: ordinary least-squares slope over equally spaced looks
- Seed weighting: equal
- Bootstrap: 10,000 row-stratified draws, row identity paired across looks,
  seed `20260819 + transition_index`
- Positive trend: point slope and one-sided 95 percent lower bound both exceed
  zero for both transitions
- Deferral count: at most one

## Execution Boundary

The first executable wave must stop after the step-5,000 archives and receipts
are durable. M4 cannot begin until the scripted gate is adjudicated under this
signed lock. A qualifying deferral extends only to step 8,000 and cannot repeat.
