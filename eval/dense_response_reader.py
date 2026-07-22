"""Readers for dense direct and serialized-scratchpad completions."""

from __future__ import annotations

import re


def extract_first_completed_symbol(text: str, candidates: list[str]) -> str | None:
    """Return the answer from the first completed response in ``text``.

    Direct training completions start with a bare answer symbol. Serialized
    scratchpads end at their first ``answer:`` marker. Models were not trained
    to emit EOS after these short completions, so later generated examples or
    repeated answer markers must not overwrite the first completed response.
    """

    allowed = {str(candidate).strip().upper() for candidate in candidates}
    raw = str(text)

    leading = re.match(r"\s*([A-Z])(?![A-Za-z0-9])", raw)
    if leading and leading.group(1).upper() in allowed:
        return leading.group(1).upper()

    answer_matches = re.findall(
        r"(?i:answer)\s*[:=]?\s*([A-Z])(?![A-Za-z0-9])",
        raw,
    )
    answer_valid = [match.upper() for match in answer_matches if match.upper() in allowed]
    if answer_valid:
        return answer_valid[0]

    matches = re.findall(r"(?<![A-Za-z0-9])([A-Z])(?![A-Za-z0-9])", raw)
    valid = [match for match in matches if match in allowed]
    return valid[0] if valid else None
