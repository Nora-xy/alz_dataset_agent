from typing import Final

SUPPRESSION_THRESHOLD: Final[int] = 10


class _Suppressed:
    """Sentinel returned instead of a count in (0, SUPPRESSION_THRESHOLD)."""

    _instance: "_Suppressed | None" = None

    def __new__(cls) -> "_Suppressed":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "SUPPRESSED"

    def __str__(self) -> str:
        return "SUPPRESSED"


SUPPRESSED: Final[_Suppressed] = _Suppressed()


def suppress(n: int) -> int | _Suppressed:
    """Return SUPPRESSED if 0 < n < SUPPRESSION_THRESHOLD, else return n unchanged.

    0 and counts >= threshold pass through; only small positive counts are masked.
    """
    if 0 < n < SUPPRESSION_THRESHOLD:
        return SUPPRESSED
    return n
