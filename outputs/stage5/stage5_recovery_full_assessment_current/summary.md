# Stage 5 Recovery Full Assessment - stage5_recovery_full_assessment_current

- Status: `needs_competence_recovery`
- Passed: `False`
- Source summary: `outputs/stage5/stage5_balanced_recovery_autopilot_current/summary.json`
- Selected gate: `arc_mix`
- Selected checkpoint: `outputs/stage5/stage5_balanced_recovery_autopilot_current_arc_mix/arc_mix_nodistill_lr3e6/phase1/phase1_step_150.pt`
- Benchmark summary: `outputs/stage5/stage5_recovery_full_assessment_current_balanced_full/summary.json`
- Balanced assessment: `outputs/stage5/stage5_recovery_full_assessment_current/balanced_assessment/summary.json`
- Next step: Use the selected checkpoint as the current balanced baseline, then train with a competence-preserving mixed objective before returning to particles/SVGD.
