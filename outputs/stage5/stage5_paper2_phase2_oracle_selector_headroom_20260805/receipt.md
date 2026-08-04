# Phase-2 Oracle-Selector Headroom Receipt

CPU-only post-processing of banked DEV row tensors; no model inference or training.

| Alpha | Seed | Always-on delta | Oracle delta | Safe-oracle delta | Selected |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0 | -0.000530 | 0.001657 | 0.001631 | 46.99% |
| 0.0 | 1 | -0.003086 | 0.007954 | 0.007901 | 46.94% |
| 0.5 | 0 | -0.003038 | 0.005457 | 0.005416 | 46.21% |
| 0.5 | 1 | -0.001924 | 0.003778 | 0.003748 | 47.90% |
| 1.0 | 0 | -0.002647 | 0.005319 | 0.005298 | 46.28% |
| 1.0 | 1 | -0.002438 | 0.004826 | 0.004808 | 47.53% |

This is a perfect-hindsight ceiling on cached teacher-forced accepted length, not a deployable selector result.
