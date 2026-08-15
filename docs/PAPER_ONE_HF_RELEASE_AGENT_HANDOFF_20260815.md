# Coding Agent Handoff — Paper One Hugging Face Release

**Date:** 2026-08-15
**For:** a coding agent with a Colab runtime, Google Drive access, and the
Hugging Face write token.
**Goal:** publish three model repositories for arXiv:2608.11233, verified.

Read this whole document before running anything. The pipeline fails closed at
every stage by design; if you find yourself adding a flag to force past a gate,
stop and report instead.

---

## 1. What is being released

Three Hugging Face model repositories under namespace `mshapiro123`:

| Repo | Kind | Trained parameters | Approx. upload |
|---|---|---|---|
| `recurrent-qwen2.5-0.5b-full-block` | `full_block_delta` | 180,556,929 | ~700 MB |
| `recurrent-qwen2.5-0.5b-natural-keeper` | `full_block_delta` | 180,556,929 | ~700 MB |
| `recurrent-qwen2.5-0.5b-r16-adapter` | `lora_adapter` | 6,007,425 | ~25 MB |

All three are deltas over the frozen base `Qwen/Qwen2.5-0.5B-Instruct`, loaded by
custom in-repo modeling code via `trust_remote_code=True`. None of them is a
standalone model, and none is a stock PEFT adapter.

The authoritative spec is `hf_release/release_manifest.json`. It holds, per repo:
the Drive checkpoint path, the expected source SHA-256, the expected parameter
counts, and the card to install as `README.md`. Do not hardcode any of these
values; read them from the manifest.

## 2. Preconditions

**Colab secrets.** Both must exist before you start:

- `GH_TOKEN` — fine-grained GitHub token for the private repo
  `mshapiro123/recurrent-qwen-svgd`.
- `HF_TOKEN` — Hugging Face token with **write** scope.

**Never** print either token, write it into a file, commit it, or paste it into a
report. The release cell already redacts both from echoed commands; preserve that
behavior in anything you add.

**Runtime.** A T4 is sufficient. CPU-only works but the round-trip step is slow.
You need roughly 4 GB of local scratch under `/content` for the converted deltas
and packaged folders.

**Drive.** The keeper checkpoints are Drive-only. They are not, and never were,
on any local disk. If Drive is not mounted or the paths are not visible, the
release cannot proceed — that is a hard stop, not something to work around.

## 3. Run it

```bash
python colab/STAGE5_PAPER_ONE_HF_RELEASE_CELL.py
```

Or paste the file's contents into a Colab cell. It performs, in order:

1. **Clone/pull** `main` from GitHub, and `pip install -U huggingface_hub safetensors`.
2. **Mount Drive** and check all three `drive_checkpoint` paths are visible.
   Prints `PREFLIGHT_GREEN`, or raises `PREFLIGHT_RED` and stops having uploaded
   nothing.
3. **Card hygiene gate** (`hf_release.card_hygiene`). Validates on every card:
   `license: apache-2.0`, `base_model: Qwen/Qwen2.5-0.5B-Instruct`,
   `library_name: transformers`, the required tags, the published arXiv ID, the
   absence of the reserved placeholder, the absence of ledger-prohibited phrases,
   the absence of stale loader prose, and that no card cites the prohibited
   step-4,000 Arm T peak. Any failure stops the run.
4. **Convert** each keeper (`hf_release.convert_checkpoint`). For each, it
   hashes the Drive checkpoint and refuses on any mismatch against the manifest,
   remaps `base_model.*` keys to `backbone.*`, checks the total parameter count,
   and for the adapter additionally checks the LoRA/bridge split and rejects any
   non-adapter tensor. Writes `recurrent_delta.safetensors` plus a
   `conversion_receipt.json`.
5. **Package** the three folders (`hf_release.package_repos`), installing the
   card as `README.md`, the LICENSE, `config.json`, both modeling modules, the
   verification subset and spec, and an LFS `.gitattributes`. Run without
   `--allow-missing-weights` so a missing delta is an error.
6. **Create private repos and upload** (`hf_release.upload_private`). It asserts
   each repo is private after upload and aborts otherwise.
7. **Round-trip verify** (`hf_release.roundtrip_verify`). For each repo it
   re-downloads the snapshot pinned to the commit it just pushed, re-hashes the
   verification subset against its receipt, loads the model through
   `trust_remote_code=True`, runs the frozen forced-depth evaluation, and
   compares correct-counts per depth against the expected values. For repos whose
   spec sets `identity_check`, it also asserts the T = 1 path is bit-identical to
   the dense backbone.

The cell ends with `RELEASE_GREEN` or `RELEASE_RED` and a per-repo table.
Receipts land in `/content/paper_one_release/receipts/`.

## 4. Stop here and report

**Do not flip any repository to public.** On `RELEASE_GREEN`, report to Mark:

- the three repo IDs and their commit SHAs, from `receipts/upload_private.json`;
- for each repo, `observed_correct_by_depth` vs `verification_spec`'s
  `expected_correct_by_depth`, and the `loop1_identity` result;
- the `safetensors_sha256` from each `conversion_receipt.json`.

Going public is Mark's call, taken in each repo's Settings → Change visibility
after he reviews those receipts. There is a hard technical reason the order
matters: `roundtrip_verify.py` raises if the repo is not private, so once a repo
is public the gate can never run against it again. Public-on-upload permanently
forfeits the only end-to-end proof that the published weights load and score
correctly.

Copy the receipts somewhere durable before the runtime is recycled — commit them
under `docs/` or write them to Drive. A Colab runtime disappearing takes the only
copy with it.

## 5. Failure modes

| Symptom | Meaning | Action |
|---|---|---|
| `PREFLIGHT_RED` | A keeper is not visible in Drive | Reauthorize Drive or fix the backup path. Do not substitute a different checkpoint. |
| `Source checkpoint hash mismatch` | The Drive file is not the receipt-bound keeper | **Stop.** Report the expected and actual SHA. Never edit the manifest to match the file. |
| `Parameter mismatch` / `LoRA parameter mismatch` | The checkpoint is not the arm the paper describes | Stop and report. |
| `Adapter release contains non-adapter/base tensors` | The adapter checkpoint carries base weights | Stop. Shipping it would leak full base weights into a repo documented as adapter-only. |
| `Checkpoint has no nonempty trainable_state_dict` | Wrong checkpoint format | Stop and report. |
| `Refusing release: <id> is not private` | A repo already exists and is public | Stop and report. Do not force. |
| `Release gate requires a private repo` | Round-trip ran against a public repo | The repo was flipped early. Report; the gate cannot be run now. |
| Round-trip status `red` | Counts or identity did not reproduce | Leave everything private. Report observed vs expected per depth. |

## 6. Things that will silently produce a broken release

- **Using `model.safetensors`.** The weight file must be
  `recurrent_delta.safetensors`; `config.json` names it in `delta_filename` and
  the in-repo loader reads that name. A repo with `model.safetensors` will not
  load.
- **Hand-staging folders.** Always use `package_repos.py`. A hand-built folder
  will miss `verification_spec.json`, `verification_subset.jsonl`, the LFS
  `.gitattributes`, or one of the two modeling modules.
- **Uploading `outputs/hf_release_20260731/repos/`.** Those folders were built on
  2026-07-31 with `--allow-missing-weights`. They contain no weights and their
  READMEs predate the arXiv ID. They are superseded; the cell writes to
  `/content/paper_one_release/repos` instead. Ignore them.
- **Setting `library_name: peft` on the adapter card.** It was changed to
  `transformers` deliberately: the artifact uses this project's custom rank-16
  modules inside the recurrent wrapper and is not a stock `PeftModel`. The PEFT
  value makes the Hub advertise a loader that cannot load this repo.
- **Re-adding the `arXiv:XXXX.XXXXX` placeholder.** The paper is published;
  `card_hygiene.py` now requires the real `arXiv:2608.11233`.

## 7. After the repos are public (Mark's checks)

- Each model page shows the `Qwen/Qwen2.5-0.5B-Instruct` base-model link and the
  Apache 2.0 badge, both from the card frontmatter.
- Within about a day, the Hub paper page for arXiv:2608.11233 lists all three
  models. That page is where Mark clicks his name to claim authorship.

## 8. Background

`docs/PAPER_ONE_HF_RELEASE_RUNBOOK_20260815.md` records the state of the release
and the two corrections that produced this handoff: the weights are Drive-only
rather than local, and the four card edits from
`HF_RELEASE_CARD_EDIT_REPORT_20260731.md` were applied on 2026-08-15 along with a
fix to `card_hygiene.py`, whose arXiv check had been written to require the
pre-publication placeholder.
