# Hugging Face Card Edit Report — 2026-07-31

The three lineage SHA placeholders are filled from the durable receipts. The
arXiv placeholders remain unchanged.

The selected release implementation is a self-contained custom Transformers
loader shipped in each model repository and invoked with
`trust_remote_code=True`. The following prose/metadata edits are required before
upload and are intentionally not applied without strategy-lane approval:

1. Full-block card, Architecture: remove “or through the loader in the companion
   repository.” The in-repo loader is the supported release path; the GitHub
   repository remains the evidence and development source.
2. Natural-keeper card, What this checkpoint is: make the same in-repo-loader
   clarification.
3. Adapter card frontmatter: change `library_name: peft` to
   `library_name: transformers`. The shipped artifact uses the project's custom
   rank-16 modules inside the recurrent wrapper and is not a stock PEFT
   `PeftModel` repository.
4. Adapter card, Architecture: replace “Load through the wrapper in the
   companion repository” with the in-repo `trust_remote_code=True` path.

No measured-result, limitation, or bounded-claim wording needs revision.

