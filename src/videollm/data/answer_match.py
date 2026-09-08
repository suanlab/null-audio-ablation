"""Match a generative model's free text against a closed answer vocabulary.

MUSIC-AVQA is a closed-set benchmark: every answer is one of 41 words, and the benchmark's
own baselines classify over that vocabulary. Scoring an LLM against it therefore needs a
deterministic reader, not an LLM judge -- the estimand here is a *paired difference*
between audio conditions on the same item, so judge noise would enter every cell of every
confidence interval.

The obvious readers are both wrong:

* **Longest match** picks ``flute`` out of *"No, the flute is not playing"*, inverting a
  yes/no answer into an instrument.
* **Naive substring search** lets ``one`` fire inside *"someone"*, so matching is
  word-boundary anchored.

What survives is: normalise, try the whole reply as an exact answer, otherwise take the
*earliest* vocabulary term appearing as a whole word, breaking ties toward the longer term
so ``acoustic guitar`` wins over a hypothetical bare ``guitar``. Models put their answer
first far more often than they bury it, and the earliest-match rule is what makes the
common *"No, ..."* continuation read correctly.

Both adapters import this module by path so the two models are read by identical code.
"""

from __future__ import annotations

import re
from functools import lru_cache


def normalise(text: str) -> str:
    """Lowercase, unify underscores to spaces, drop punctuation, collapse whitespace."""
    lowered = text.lower().replace("_", " ")
    stripped = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", stripped).strip()


@lru_cache(maxsize=8)
def _patterns(vocabulary: tuple[str, ...]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Compile a word-boundary pattern per vocabulary entry, longest entry first."""
    ordered = sorted(vocabulary, key=lambda v: (-len(v), v))
    return tuple((v, re.compile(rf"\b{re.escape(normalise(v))}\b")) for v in ordered)


def match_answer(text: str, vocabulary: tuple[str, ...]) -> str:
    """Read one vocabulary answer out of a model's reply.

    Args:
        text: The model's raw generation.
        vocabulary: The closed answer set, in any order.

    Returns:
        The matched vocabulary entry in its original spelling, or ``""`` if the reply
        contains no vocabulary term (which is recorded as an unparseable prediction rather
        than silently scored wrong against a guess).

    Raises:
        ValueError: If ``vocabulary`` is empty, which would make every reply unparseable.
    """
    if not vocabulary:
        raise ValueError("vocabulary must be non-empty")
    cleaned = normalise(text)
    if not cleaned:
        return ""

    for entry, _ in _patterns(vocabulary):
        if cleaned == normalise(entry):
            return entry  # the whole reply is the answer

    best: tuple[int, int, str] | None = None
    for entry, pattern in _patterns(vocabulary):
        found = pattern.search(cleaned)
        if found is None:
            continue
        # earliest wins; on a tie the longer term wins (patterns are longest-first).
        candidate = (found.start(), -len(entry), entry)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return best[2] if best else ""


def is_recitation(text: str, vocabulary: tuple[str, ...], threshold: int = 5) -> bool:
    """True when a reply lists the candidate set instead of choosing from it.

    A model that is not instruction-tuned for closed-vocabulary answering sometimes echoes
    the options back ("...including an accordion, acoustic guitar, bagpipe, ..."). The
    earliest-match reader would score that as the first option, which is an artefact of
    option ordering rather than an answer. Such replies are recorded as unparseable.

    Args:
        text: The model's raw generation.
        vocabulary: The closed answer set.
        threshold: How many distinct vocabulary terms constitute a recitation.

    Returns:
        Whether the reply names at least ``threshold`` distinct vocabulary terms.
    """
    cleaned = normalise(text)
    hits = sum(1 for entry, pattern in _patterns(vocabulary) if pattern.search(cleaned))
    return hits >= threshold
