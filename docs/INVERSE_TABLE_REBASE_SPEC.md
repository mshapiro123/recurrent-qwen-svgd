# Inverse-Table Rebase Caps 3-4

This bounded continuation follows the matched-dose staircase stop. It does not
reinterpret the forward-table failure as a measured five-fold cost. It asks a
new, narrower question: whether the successful explicit-inverse representation
provides a deterministic recurrent substrate beyond cap 2.

The run starts from the exact green C cap-2 checkpoint SHA256
`bc1de1cd7d2a7acf30b9217c8d7054d805888c341b942ff0dab7691b4f995b01`.
It uses the same locked rows, AdamW objective, dose accounting, 46/64 stage bar,
and synthetic guardrail as the matched run. Caps 3 and 4 run sequentially. A
final natural-surface canary is mandatory even if fewer than 1,000 new optimizer
steps were spent.

Green means both caps pass, both synthetic guardrails pass, and the natural
canary avoids a red hard stop. Green pauses for review; it does not automatically
launch caps 5-8 or Phase G-alpha. Any failed stage or guardrail is a scientific
stop.
