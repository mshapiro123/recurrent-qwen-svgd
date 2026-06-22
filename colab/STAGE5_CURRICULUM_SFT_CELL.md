# Stage 5 Generated Curriculum SFT Cell

Paste the contents of `colab/STAGE5_CURRICULUM_SFT_CELL.py` into one Colab
cell after the generated curriculum pipeline has completed and produced a green
`curriculum_sft_gate.json`.

Default behavior:

- pulls `mshapiro123/recurrent-qwen-svgd`;
- verifies GitHub/HF auth from Colab secrets;
- mounts Google Drive;
- runs focused safety tests;
- runs `colab/run_stage5_curriculum_sft.py`;
- refuses GPU training unless the SFT gate is green, enough positive rows
  exist, and Drive backup is visible.

Edit these values at the top of the cell when needed:

```python
WORK_DIR = "data/curriculum/run_001"
MIN_POSITIVE_ROWS = "16"
PHASE1_STEPS = "150"
MAX_LOOPS = "4"
```

For an explicit continuation from an existing deterministic recurrent
checkpoint, add:

```python
env["STAGE5_CURRICULUM_RESUME_FROM"] = "outputs/stage5/<run_id>/phase1/phase1_step_125.pt"
```
