"""P8 — Eval harness: protects the project's core guarantees.

See alz_agent.eval.harness for run_eval() / the CLI entry point, and
alz_agent.eval.golden_questions for the golden question set.
"""

from alz_agent.eval.golden_questions import GOLDEN_QUESTIONS, GoldenQuestion
from alz_agent.eval.harness import (
    CategoryResult,
    CheckResult,
    EvalReport,
    default_backend,
    default_catalog,
    run_eval,
)

__all__ = [
    "GOLDEN_QUESTIONS",
    "GoldenQuestion",
    "CategoryResult",
    "CheckResult",
    "EvalReport",
    "default_backend",
    "default_catalog",
    "run_eval",
]
