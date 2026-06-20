# Stage 4 Particle Noise Sweep - stage4_particle_noise_a100_20260620

Phase1 deterministic baseline: `0.546875`

## Results
- phase1_label: best=0.546875 lift=0.0 raw={'mean': {'correct': 70, 'total': 128, 'accuracy': 0.546875}} helped_hurt=None
- phase1_k4_noise0_rep0: best=0.5390625 lift=-0.0078125 raw={'max': {'correct': 69, 'total': 128, 'accuracy': 0.5390625}, 'mean': {'correct': 69, 'total': 128, 'accuracy': 0.5390625}, 'vote': {'correct': 69, 'total': 128, 'accuracy': 0.5390625}} helped_hurt={'max': {'helped': 2, 'hurt': 3, 'tied': 123}, 'mean': {'helped': 2, 'hurt': 3, 'tied': 123}, 'vote': {'helped': 2, 'hurt': 3, 'tied': 123}}
- phase1_k2_noise005_rep05: best=0.515625 lift=-0.03125 raw={'max': {'correct': 66, 'total': 128, 'accuracy': 0.515625}, 'mean': {'correct': 60, 'total': 128, 'accuracy': 0.46875}, 'vote': {'correct': 61, 'total': 128, 'accuracy': 0.4765625}} helped_hurt={'max': {'helped': 11, 'hurt': 15, 'tied': 102}, 'mean': {'helped': 9, 'hurt': 19, 'tied': 100}, 'vote': {'helped': 9, 'hurt': 18, 'tied': 101}}
- phase1_k2_noise01_rep05: best=0.515625 lift=-0.03125 raw={'max': {'correct': 66, 'total': 128, 'accuracy': 0.515625}, 'mean': {'correct': 64, 'total': 128, 'accuracy': 0.5}, 'vote': {'correct': 64, 'total': 128, 'accuracy': 0.5}} helped_hurt={'max': {'helped': 11, 'hurt': 15, 'tied': 102}, 'mean': {'helped': 9, 'hurt': 15, 'tied': 104}, 'vote': {'helped': 9, 'hurt': 15, 'tied': 104}}
- phase1_k4_noise005_rep05: best=0.4921875 lift=-0.0546875 raw={'max': {'correct': 63, 'total': 128, 'accuracy': 0.4921875}, 'mean': {'correct': 58, 'total': 128, 'accuracy': 0.453125}, 'vote': {'correct': 60, 'total': 128, 'accuracy': 0.46875}} helped_hurt={'max': {'helped': 15, 'hurt': 22, 'tied': 91}, 'mean': {'helped': 12, 'hurt': 24, 'tied': 92}, 'vote': {'helped': 14, 'hurt': 24, 'tied': 90}}
- phase1_k4_noise01_rep05: best=0.5 lift=-0.046875 raw={'max': {'correct': 63, 'total': 128, 'accuracy': 0.4921875}, 'mean': {'correct': 63, 'total': 128, 'accuracy': 0.4921875}, 'vote': {'correct': 64, 'total': 128, 'accuracy': 0.5}} helped_hurt={'max': {'helped': 15, 'hurt': 22, 'tied': 91}, 'mean': {'helped': 15, 'hurt': 22, 'tied': 91}, 'vote': {'helped': 17, 'hurt': 23, 'tied': 88}}
- phase1_k4_noise02_rep05: best=0.5390625 lift=-0.0078125 raw={'max': {'correct': 69, 'total': 128, 'accuracy': 0.5390625}, 'mean': {'correct': 67, 'total': 128, 'accuracy': 0.5234375}, 'vote': {'correct': 68, 'total': 128, 'accuracy': 0.53125}} helped_hurt={'max': {'helped': 21, 'hurt': 22, 'tied': 85}, 'mean': {'helped': 20, 'hurt': 23, 'tied': 85}, 'vote': {'helped': 21, 'hurt': 23, 'tied': 84}}
- phase1_k4_noise005_rep2: best=0.3203125 lift=-0.2265625 raw={'max': {'correct': 41, 'total': 128, 'accuracy': 0.3203125}, 'mean': {'correct': 38, 'total': 128, 'accuracy': 0.296875}, 'vote': {'correct': 39, 'total': 128, 'accuracy': 0.3046875}} helped_hurt={'max': {'helped': 18, 'hurt': 47, 'tied': 63}, 'mean': {'helped': 14, 'hurt': 46, 'tied': 68}, 'vote': {'helped': 12, 'hurt': 43, 'tied': 73}}

## Decision Rule
A particle setting must match or beat Phase1 and have helped >= hurt before we train around it.
