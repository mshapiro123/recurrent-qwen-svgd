# Natural-Surface Transfer Prep - stage5_natural_surface_transfer_20260708_230229

- Status: `finished`
- Dataset summary: `outputs/stage5/stage5_natural_surface_transfer_20260708_230229/data/summary.json`
- Symbols: `20` names
- Relay train: `2048` rows
- Rung-zero mix: `4096` rows
- Relay eval: `1536` rows
- Pointer eval: `1536` rows
- Tokenizer verification: `True`

## Queue Role

- Ungated CPU/data-prep artifact for the natural-surface transfer program.
- Does not update `config/stage5_current_source_summary.txt`; the GPU battery remains the active line.
- Rung zero trains relay verbal plus symbolic rehearsal; pointer is a held-out zero-shot template read.
