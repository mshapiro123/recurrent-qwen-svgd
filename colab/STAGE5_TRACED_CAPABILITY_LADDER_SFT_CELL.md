# Stage 5 Traced Capability-Ladder SFT Cell

Paste `colab/STAGE5_TRACED_CAPABILITY_LADDER_SFT_CELL.py` into a GPU Colab
runtime after capability-ladder trace collection has produced a gate-ready
`stage5_capability_ladder_trace_collection` summary.

Default behavior:

- pulls `mshapiro123/recurrent-qwen-svgd`;
- verifies GitHub/HF auth from Colab secrets;
- mounts Google Drive;
- resolves the latest gate-ready traced capability-ladder collection;
- derives `STAGE5_CURRICULUM_WORK_DIR`, `STAGE5_CURRICULUM_SUMMARY_JSON`,
  row gates, mode gates, max loops, and bounded training steps from the summary;
- enables answer-line verification for trace-derived rows while preserving
  `answer_match.matched=true` checks on positive traces;
- runs `colab/run_stage5_curriculum_sft.py`;
- pushes the SFT summary and disconnects by default.

Useful overrides:

```python
os.environ["STAGE5_TRACED_CAPABILITY_SFT_SOURCE_SUMMARY"] = "outputs/stage5/<trace_collection_run>/summary.json"
os.environ["STAGE5_TRACED_CAPABILITY_SFT_PHASE1_STEPS"] = "75"
os.environ["STAGE5_TRACED_CAPABILITY_SFT_MODEL_NAME"] = "Qwen/Qwen2.5-0.5B-Instruct"
os.environ["STAGE5_TRACED_CAPABILITY_SFT_DISCONNECT"] = "0"
```

For a deliberate tiny smoke run, set:

```python
os.environ["STAGE5_TRACED_CAPABILITY_SFT_ALLOW_TINY"] = "1"
```
