"""P8 — Golden question set for the eval harness.

Each GoldenQuestion pairs a natural-language research question with the
expected AgentResponse shape: grounded/abstained status, plus substrings the
narrated answer must (or must not) contain. ``intent`` overrides the heuristic
parser when a question needs an exact StructuredIntent to be deterministic
(e.g. forcing an unknown label to exercise the abstain path) rather than being
at the mercy of keyword-matching quirks unrelated to what the question is
testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from alz_agent.agent.intent import QuestionType, StructuredIntent
from alz_agent.schemas import IdSpace

# ---------------------------------------------------------------------------
# Worked-target question (from AGENTS.md / P5 / P6 / P7)
# ---------------------------------------------------------------------------

WORKED_TARGET = (
    "Among MCI participants at an index visit, can we predict 3-year conversion "
    "to AD using only pre-index information, and how many subjects survive that "
    "definition?"
)


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    nl_question: str
    expected_grounded: bool
    intent: StructuredIntent | None = None
    must_contain: tuple[str, ...] = field(default_factory=tuple)
    must_not_contain: tuple[str, ...] = field(default_factory=tuple)


GOLDEN_QUESTIONS: tuple[GoldenQuestion, ...] = (
    # ---- feasible ----------------------------------------------------------
    GoldenQuestion(
        id="worked_target",
        nl_question=WORKED_TARGET,
        expected_grounded=True,
        must_contain=("3",),
    ),
    GoldenQuestion(
        id="cohort_size_normal",
        nl_question="How many normal participants are in the NACC cohort?",
        expected_grounded=True,
    ),
    GoldenQuestion(
        id="feasible_5yr_mri",
        nl_question="Can we use MRI volumes to predict conversion in 5 years from MCI to dementia?",
        expected_grounded=True,
        must_contain=("5",),
    ),
    GoldenQuestion(
        id="multimodal_cognition_mri",
        nl_question="Can we combine cognition and MRI to predict 3-year conversion in MCI?",
        expected_grounded=True,
        must_contain=("cognition", "mri"),
    ),
    # ---- lookup --------------------------------------------------------------
    GoldenQuestion(
        id="lookup_mmse",
        nl_question="What is the MMSE total score variable in NACC?",
        expected_grounded=True,
        must_contain=("mmse",),
        # Heuristic modality keywords don't include "mmse"; supply the lookup
        # target explicitly via candidate_label (the LOOKUP fallback target)
        # so the question exercises lookup_variable deterministically.
        intent=StructuredIntent(
            target_population=[],
            candidate_label="mmse_total_score",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.LOOKUP,
            pre_index_only=True,
            id_space=IdSpace.A,
        ),
    ),
    # ---- infeasible / abstain -------------------------------------------------
    GoldenQuestion(
        id="infeasible_unknown_population",
        nl_question="Can we predict outcomes for the xyz_invalid_cohort population?",
        expected_grounded=False,
        intent=StructuredIntent(
            target_population=["xyz_invalid_cohort"],
            candidate_label="conversion to dementia",
            horizon_years=3,
            modalities=[],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        ),
    ),
    GoldenQuestion(
        id="partial_abstain_invalid_modality",
        # Design portion is still grounded (cohort size + feasibility don't
        # depend on the bad modality name); only the overlap sub-result
        # abstains. Overall response.grounded stays True — see
        # test_p5_agent.py::TestModalityOverlapInToolLoop.
        nl_question="Can we combine retina-scan imaging with cognition to predict MCI conversion?",
        expected_grounded=True,
        intent=StructuredIntent(
            target_population=["mci"],
            candidate_label="conversion to dementia",
            horizon_years=3,
            modalities=["INVALID_MODALITY_XYZ"],
            question_type=QuestionType.MODELING_DESIGN,
            pre_index_only=True,
            id_space=IdSpace.A,
        ),
    ),
)
