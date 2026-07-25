# COCONUT Composite Integrity Receipt

- Status: `failed_integrity_contract`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Training performed: `False`
- RG-12 authorized or run: `False`

| Check | Passed | Key observation |
|---|---:|---|
| RG1 | `True` | full-block H=0 max diff 0 |
| RG2 | `True` | identity bridge versus raw feedback max diff 0 |
| RG3 | `True` | fed=1.407e-01, prompt=1.632e+02 |
| RG4 | `False` | directional derivative abs error 1.831e-03 |
| RG5 | `False` | cache/recompute grad cosine 1.000000 |
| RG10 | `True` | checkpoint gradient cosine 1.000000 |
| RG9 | `True` | one full forward/backward completed under detect_anomaly |
| RG11 | `False` | bf16/fp32 fed-gradient cosine 0.983584 |
| RG6 | `True` | 84 LoRA modules; frozen base transparent |
| RG7 | `True` | H*L=2 feedback cells; (H+1)*L=3 total cells |
| RG8 | `True` | replaced input-slot gradient zero; parameter-name hashes exact |

The reference recompute path is the authorized future training path. 
Sliced cache is limited to L=1 and is forbidden with active gradient checkpointing.
RG-12 remains unrun pending the T1-lite verdict and a locked null-calibrated KL floor.
