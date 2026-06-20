# Stage 4 Phase2 Rescue Ablation - stage4_phase2_rescue_a100_20260620

## Question
Does inference-time particle/SVGD improve the strong Stage 4 Phase1 recurrent baseline before more Phase2 training?

## ARC Label-Likelihood
- base_label: best=0.5625 lift_over_phase1=0.015625 gap_to_base=0.0 raw={'mean': {'correct': 72, 'total': 128, 'accuracy': 0.5625}}
- phase1_label: best=0.546875 lift_over_phase1=0.0 gap_to_base=-0.015625 raw={'mean': {'correct': 70, 'total': 128, 'accuracy': 0.546875}}
- phase1_particles_rep0_label: best=0.53125 lift_over_phase1=-0.015625 gap_to_base=-0.03125 raw={'max': {'correct': 68, 'total': 128, 'accuracy': 0.53125}, 'mean': {'correct': 67, 'total': 128, 'accuracy': 0.5234375}, 'vote': {'correct': 66, 'total': 128, 'accuracy': 0.515625}}
- phase1_particles_rep05_label: best=0.53125 lift_over_phase1=-0.015625 gap_to_base=-0.03125 raw={'max': {'correct': 68, 'total': 128, 'accuracy': 0.53125}, 'mean': {'correct': 67, 'total': 128, 'accuracy': 0.5234375}, 'vote': {'correct': 68, 'total': 128, 'accuracy': 0.53125}}
- phase1_particles_rep2_label: best=0.5234375 lift_over_phase1=-0.0234375 gap_to_base=-0.0390625 raw={'max': {'correct': 65, 'total': 128, 'accuracy': 0.5078125}, 'mean': {'correct': 67, 'total': 128, 'accuracy': 0.5234375}, 'vote': {'correct': 66, 'total': 128, 'accuracy': 0.515625}}
- phase2_stage4_rep2_label: best=0.5390625 lift_over_phase1=-0.0078125 gap_to_base=-0.0234375 raw={'max': {'correct': 65, 'total': 128, 'accuracy': 0.5078125}, 'mean': {'correct': 69, 'total': 128, 'accuracy': 0.5390625}, 'vote': {'correct': 69, 'total': 128, 'accuracy': 0.5390625}}

## Decision Rule
If no Phase1-particle arm beats `phase1_label`, do not continue Phase2 training with the current particle mechanism.
