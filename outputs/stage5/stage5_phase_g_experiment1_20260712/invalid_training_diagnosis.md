# Invalid Phase G Experiment 1 Attempt

The run `stage5_phase_g_experiment1_20260712` performed 1,000 no-op steps. Every logged step had loss `0`, active loop labels `0`, and trainable gradients `0`. The resulting injective smoke score, `9/128`, is an untrained baseline and must not be read as a model negative.

The prompt/completion boundary has been corrected to `Answer:` plus `" Name"`. The replacement run is `stage5_phase_g_experiment1_fixed_boundary_20260712` and aborts before paid training when active supervision is zero or immediately after backward when all trainable gradients are zero.
