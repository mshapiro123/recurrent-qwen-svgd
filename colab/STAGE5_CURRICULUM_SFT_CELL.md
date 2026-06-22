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
- restores `WORK_DIR` from
  `/content/drive/MyDrive/recurrent-qwen-svgd/curriculum_runs/<run_name>` when
  a Colab runtime reset wiped local `data/curriculum`.

Edit these values at the top of the cell when needed:

```python
WORK_DIR = "data/curriculum/run_001"
MIN_POSITIVE_ROWS = "16"
MIN_MODE_ROWS = "direct=64,deep_narrow=64"  # Current objective; edit to "wide=64" for width-only shards.
PHASE1_STEPS = "150"
MAX_LOOPS = "4"
```

Keep `MIN_MODE_ROWS` aligned with the run objective. The default protects the
current direct/deep calibration phase, so a shard should not train just because
it has many `wide` rows:

```python
MIN_MODE_ROWS = "direct=64,deep_narrow=64"
```

For the later width phase, change it deliberately, for example:

```python
MIN_MODE_ROWS = "wide=64"
```

If your CPU/API curriculum artifacts were backed up somewhere else, set:

```python
env["STAGE5_CURRICULUM_INPUT_BACKUP_DIR"] = "/content/drive/MyDrive/path/to/curriculum_runs"
```

For an explicit continuation from an existing deterministic recurrent
checkpoint, add:

```python
env["STAGE5_CURRICULUM_RESUME_FROM"] = "outputs/stage5/<run_id>/phase1/phase1_step_125.pt"
```
