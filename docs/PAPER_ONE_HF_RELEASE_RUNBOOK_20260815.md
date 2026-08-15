# Paper One Hugging Face Release Runbook — 2026-08-15

State of the release as of this document, and the exact remaining sequence.
arXiv:2608.11233 is live; the three model repositories are not yet created.

## What was already true

- The three cards in `docs/hf_cards/` carry the real arXiv ID, `license: apache-2.0`,
  and `base_model: Qwen/Qwen2.5-0.5B-Instruct`.
- `outputs/hf_release_20260731/repos/` holds three folders assembled on 2026-07-31
  with `--allow-missing-weights`. **They contain no weights and their READMEs
  predate the arXiv ID.** Do not upload them. They are superseded by the packaging
  step below, which writes to a fresh directory.
- No conversion has ever run: there is no `recurrent_delta.safetensors` and no
  `conversion_receipt.json` anywhere on the local machine.
- The round-trip gate has never run. It cannot have: it needs converted weights in
  a private repo.

## Two corrections to the plan this replaces

1. **The weights are not on any local machine.** All three keeper checkpoints are
   absent from `outputs/stage5/...`. Each repo's `drive_checkpoint` in
   `hf_release/release_manifest.json` points into `/content/drive/MyDrive/...`.
   The release must run in Colab with Drive mounted.
2. **The weight file is `recurrent_delta.safetensors`, not `model.safetensors`.**
   `config.json` names it in `delta_filename`, and the in-repo loader reads that
   name. A hand-staged folder using `model.safetensors` produces a repo that will
   not load. Use `hf_release/package_repos.py`; do not assemble folders by hand.

## Card edits applied 2026-08-15

The four edits from `HF_RELEASE_CARD_EDIT_REPORT_20260731.md`, held pending
strategy-lane approval, are now applied:

1. Full-block card: dropped "or through the loader in the companion repository".
2. Natural-keeper card: same in-repo-loader clarification.
3. Adapter frontmatter: `library_name: peft` -> `transformers`. The shipped
   artifact uses the project's custom rank-16 modules inside the recurrent
   wrapper and is not a stock PEFT `PeftModel`; the PEFT value would have made the
   Hub advertise a loader path that cannot load this repo.
4. Adapter card: replaced the companion-repository wrapper instruction with the
   in-repo `trust_remote_code=True` path.

`hf_release/card_hygiene.py` enforces all four, and all three cards pass it.

One fix to the gate itself: it required the placeholder `arXiv:XXXX.XXXXX` to
still be present, a pre-publication guard against inventing an ID. That check now
requires the published `arXiv:2608.11233` and rejects the placeholder. Same
intent, correct side of publication.

## Blocker: the release pipeline is untracked

`hf_release/` and `docs/hf_cards/` have **never been committed to any branch** and
are absent from `origin/main`. They are not gitignored — they exist only as
untracked files in the local `codex/coconut-composite` worktree. The converter,
packager, uploader, round-trip verifier, LICENSE, verification assets, and all
three cards exist in exactly one place on one disk.

This must be committed and pushed to `main` before the release cell will work,
because the cell clones `main` from GitHub. It is also the only backup.

## Remaining sequence

1. **Commit and push** `hf_release/`, `docs/hf_cards/`, this runbook, and
   `colab/STAGE5_PAPER_ONE_HF_RELEASE_CELL.py` to `main`.
2. **Create a write token** at huggingface.co -> Settings -> Access Tokens, and add
   it to Colab Secrets as `HF_TOKEN`. Confirm `GH_TOKEN` is present too. The token
   is entered by Mark only; it is never pasted into a chat or a source file.
3. **Run** `colab/STAGE5_PAPER_ONE_HF_RELEASE_CELL.py` in Colab. A T4 is enough;
   the round-trip step loads a 0.5B model. The cell:
   - mounts Drive and fails closed if any keeper checkpoint is not visible;
   - runs the card hygiene gate;
   - converts each keeper, verifying the source SHA-256 against the manifest and
     the parameter counts against the paper;
   - packages the three repo folders, with no `--allow-missing-weights`;
   - creates the three repos **private** and uploads;
   - downloads each one back and runs its frozen verification subset.
4. **Flip to public** only on `RELEASE_GREEN`, one repo at a time, in each repo's
   Settings -> Change visibility. The cell does not do this: `roundtrip_verify.py`
   refuses to run against a public repo, so public-on-upload would permanently
   skip the only end-to-end check that the published weights load and score
   correctly.
5. **Confirm** on each public model page: the `Qwen/Qwen2.5-0.5B-Instruct`
   base-model link and the Apache 2.0 badge. Within a day, the Hub paper page for
   arXiv:2608.11233 should list all three models; claim authorship there.

## If the round-trip gate is red

Leave every repo private. The receipt in `receipts/roundtrip_<repo>.json` records
observed versus expected counts per depth, the loop-1 identity check, and the
snapshot file hashes. A count mismatch means the uploaded delta is not the
keeper the paper describes; an identity failure means the wrapper is not
reproducing the base computation at T = 1.
