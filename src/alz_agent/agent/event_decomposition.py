"""Compound-event phrase detection for intent decomposition.

A 'compound event phrase' is a label that embeds an event anchor
(e.g. 'age at first AD diagnosis') and cannot be mapped to a single
catalog variable.  The validator in StructuredIntent and the plan
execution layer both use _is_compound_event_phrase to detect these early.
"""

from __future__ import annotations

_COMPOUND_EVENT_PHRASES: frozenset[str] = frozenset({
    "at first diagnosis",
    "when diagnosed",
    "at conversion",
    "first ad diagnosis",
    "at first ad",
    "when first diagnosed",
    "at diagnosis",
    "upon diagnosis",
    "when they were first",
    "when they first",
})


def _is_compound_event_phrase(label: str) -> bool:
    """Return True when label embeds an event-anchor phrase (not a simple variable)."""
    low = label.lower()
    return any(phrase in low for phrase in _COMPOUND_EVENT_PHRASES)
