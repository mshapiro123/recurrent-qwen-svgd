# Stage 2A Pre-Training Binding Request

Date: 2026-08-17. Status: score-blind implementation query. No optimizer has
been constructed and the Stage 2A executed lock remains disabled.

## Why this is needed

The ratified objective binds `0.5 * L_CE + 0.5 * L_KL`, the cached 14B
top-128 lattice, answer-bearing masks, and leave-one-out retrieval. Two details
remain underspecified for an executable training cache. Neither may be chosen
silently after the score-blind memory pass.

## Q1: MBPP answer target and loss mask

The objective authority explicitly defines the MCQ choice-token mask and the
GSM8K/Tier-1 normalized final-answer-token mask, but it does not define MBPP.
The admitted MBPP rows have a programmatically verified 14B response and a
concurrent, independently verified 32B response.

**Coding recommendation:** use the verified 14B generated program as the
teacher-forced target; apply CE and top-128 forward KL to code tokens only;
exclude the prompt, code-fence delimiters, surrounding prose, and position
zero. Store the exact token-span boundaries and the verifier receipt per row.
Do not substitute the dataset reference implementation, because doing so would
change the registered teacher-forced objective and can introduce a different
surface form than the response that passed V(x).

Binding requested: ratify this target/mask or provide the exact alternative.

## Q2: Training population versus memory-slot ownership

The objective authority says all V(x)-admitted non-DEV rows train. The sizing
ruling can select only the largest power-of-two subset, capped at 4,096, as
memory slots. Therefore some admitted training rows may have no owned slot.

**Coding recommendation:** retain every V(x)-admitted non-DEV row in the
training population. For a row that owns a selected slot, exclude that slot
before top-k retrieval. For an admitted row excluded by deterministic memory
subselection, leave-one-out is vacuous because no self slot exists. Publish the
owner mapping, excluded identities, and per-battery counts. This preserves both
the stated training population and the anti-copy purpose of leave-one-out.

Binding requested: ratify conditional leave-one-out over all admitted rows, or
restrict training to slot owners and explicitly amend the training population.

## Cache consequence

After these bindings, the cache pass will store, for every admitted training
row and answer-bearing position: teacher token ID, top-128 teacher token IDs and
logits at temperature 1.0, answer-region mask, row-to-slot ownership (or none),
and the source/verifier hashes. Cache construction remains score-blind and
optimizer-free. Training remains structurally disabled until the materialized
lock includes these bindings and Mark signs it.
