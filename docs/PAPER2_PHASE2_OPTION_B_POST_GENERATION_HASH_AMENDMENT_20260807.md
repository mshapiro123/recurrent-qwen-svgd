# Phase 2 Option B Post-Generation Hash Amendment

Date: 2026-08-07

## Scope

This is the hash-only amendment authorized after the Option B teacher/cache pass.
It binds the landed training population and opens the already-locked four-arm
training protocol. It changes no model parameter, optimizer setting, schedule,
gate, interpretation, rule, or analysis threshold.

The governing strategy handoff is
`STRATEGY_TO_CODING_AGENT_TEACHER_PASS_BANK_AMENDMENT_20260807.md`, Drive file
`1oN3rWgdoVmdNw3--NxgEZLqq7d1tJ78C`, reported size 5,098 bytes. The landed
teacher/cache receipt is commit `dede3f0f4edb2d5210ba3834a1f662ee81f49d7d`.

## Landed Population

- selected training anchors: 140,000
- horizon samples: 560,000
- teacher-14B state samples: 560,000
- fixed new-train diagnostic subset: 8,031 anchors
- full-logit audit samples: 5,600 (exactly 1 percent)
- evaluation partition touched: false
- zero overlap with excluded documents: true
- optimizer steps during generation: zero

## Bound Hashes

- public teacher-cache summary: `04bf069d50753475c03dd817424eae657d27a43458b61c9178249b6f1b7d45e0`
- new data: `bd0c84984f1dd47d1ee5dc06172afb7ea9728443d7712ca82ffa836d54edff9b`
- new document IDs: `be9c15fa429a3a1e79b679b778188583b289f1bc974f0154ad8b41e2c519208c`
- excluded document IDs: `b79447df432f286d9900ac3aaddb1471c8e798b6aa4411f944a48eb1bea9346b`
- position keys: `a4ba19b3356effa95122e43a80ed6aaf0b009d00c0f5546959f5b954c06403df`
- sample manifest: `8a50b8ac3a9cc361be918b968a2a8ae490d4aff79a83d41fcdc774a951788187`
- lattice summary: `fe7735267d03b13973a5f603aba38b2959101f725bdb1ef95230bee612c88b0e`
- full-logit audit keys: `6e52729d66c28218bf88c86270d326471fdc52a6ba43789925d9abe51e8e099b`
- anchor admission ledger: `a5978c71ad0871f00ca00208b6b6997decac964dc0c85d4f6f0fc44535fe9646`
- fixed new-train subset: `9e7fee29fa11190898ae2053295edef491e52de2a270c839c1db44fb38f1432d`
- exclusion-lineage closure: `287afb659921712689b29f4aa354c11daec02cf94a861814e3ca83fdb3321cae`
- student model ledger: `6f1dc39c739c1a065fbe18a23a63e7ddfe3ea23fb5be2c9b0a58efbaf4abc179`
- 7B model ledger: `9441062e01cca925e4358f0150a4358a214dbdcbba10b440629fe713fc73944e`
- 14B model ledger: `084a5badc453afb2872b439f8942eb196493b0210cacdfc2f41ed5a59ecc0cd6`
- 32B model ledger: `eb381b8dd00c582da425c8fe88fd28735859a2b950305e960116f10b9913fed9`

## Cascade Stability

The original 50,000-anchor Stage 0A collection cascaded 35,200 of 200,000
horizon samples to the 32B teacher, or 17.600 percent. The fresh 512-anchor
resource preflight measured a 16.748 percent 32B cascade rate before the full
collection. The change is -0.852 percentage points, or -4.84 percent relative.
This comparison indicates that the expanded source population did not
materially change teacher demand at preflight scale; it is not presented as a
full-population cascade count because that count is not exposed by the public
receipt.

## Training Authorization

The data splice is fixed at optimizer step 4,000. The pre-splice and post-splice
segments are authorized under the existing lock. The four arms remain seed 0
and seed 1 crossed with full-system and draft-only-control. Each arm starts from
its registered A2 endpoint with a fresh AdamW state and follows the already
locked 20,000-step schedule. The locked rule inventory rides with every runtime
receipt.
