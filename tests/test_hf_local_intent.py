"""Tests for hf-local compact intent parsing (src/alz_agent/agent/hf_local_intent.py).

Supersedes the narrow 3-field (intent="count" / population="baseline_mci"-only)
test coverage that used to live in tests/test_hf_provider.py. That schema was
so narrow that "baseline_mci" was the ONLY population token the model was ever
allowed to emit — so every hf-local batch run silently reinterpreted every
question (etiologic-AD counts, modality-overlap questions, dementia counts,
average-age stats) as a baseline-MCI cohort_size count. See
docs/HF_LOCAL_TESTING_NOTE.md and logs/hf_runs/20260708_143725_hf_fixture_batch_*
for the observed failure.

This file covers:
  - the expanded compact schema (count / modality_overlap / aggregate_stat /
    unsupported, with population/etiology/modalities/dataset fields)
  - the semantic-mismatch guard that cross-checks the model's compact object
    against the natural-language question and rejects schema-valid-but-wrong
    parses (the actual fix for the reported failure)
  - fail-closed behavior: malformed JSON, contamination, unsupported values,
    and semantic mismatches all abstain (HFLocalPlannerAbstention) without
    ever producing a StructuredIntent for tool dispatch
  - explicit intent="unsupported" abstains immediately, without burning a
    retry call
  - debug_sink raw-output capture for --show-raw-intent
  - the six required regression questions from the hf-local planning fix

Does not touch DeepSeek/Anthropic (_parse_with_llm, _parse_with_json_prompt,
providers.py) — those are covered by tests/test_hf_provider.py and are
untouched by this module.
"""

from __future__ import annotations

import json

import pytest

from alz_agent.agent.hf_local_intent import (
    CompactIntent,
    HFLocalCompactValidationError,
    HFLocalPlannerAbstention,
    _parse_hf_local_compact,
    _semantic_mismatch_reason,
    build_hf_local_compact_prompt,
    compact_intent_to_structured_intent,
    parse_hf_local_compact_json,
    validate_hf_local_compact_intent,
)
from alz_agent.agent.hf_local_synthesis import (
    hf_local_extra_evidence_digits,
    resolve_modality_union,
    synthesize_hf_local_answer,
)
from alz_agent.agent.intent import QuestionType, parse_intent
from alz_agent.agent.intent_schema import StructuredIntent
from alz_agent.agent.providers import LocalHFPlannerClient, PlannerClient
from alz_agent.schemas import IdSpace, ToolResult

# ---------------------------------------------------------------------------
# Test doubles — self-contained, no dependency on test_hf_provider.py
# ---------------------------------------------------------------------------


class _StubPlannerClient(PlannerClient):
    """A PlannerClient stub that always returns the same complete_text response."""

    def __init__(self, complete_text_return: str = "") -> None:
        self._complete_text_return = complete_text_return
        self.complete_text_calls: list = []

    def invoke_tool(self, messages, tool_schema, max_tokens=1500):
        return None

    def complete_text(self, messages, max_tokens=1024):
        self.complete_text_calls.append(messages)
        return self._complete_text_return


class _SequencedStub(PlannerClient):
    """A PlannerClient stub that returns a different response on each call.

    Raises IndexError if complete_text is called more times than responses
    were queued — used to prove "no retry" for explicit intent="unsupported".
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.complete_text_calls: list = []

    def invoke_tool(self, messages, tool_schema, max_tokens=1500):
        return None

    def complete_text(self, messages, max_tokens=1024):
        self.complete_text_calls.append(messages)
        return self._responses.pop(0)


def _compact(**overrides) -> str:
    base = {
        "intent": "count",
        "population": "baseline_mci",
        "etiology": "none",
        "modalities": [],
        "stat_variable": "none",
        "stat_type": "mean",
        "dataset": "nacc",
    }
    base.update(overrides)
    return json.dumps(base)


_BASELINE_MCI_QUESTION = "Count participants with baseline MCI. Do not infer AD etiology."


# ---------------------------------------------------------------------------
# Prompt hygiene — never leaks full-schema field names or dataset codes
# ---------------------------------------------------------------------------


def test_hf_local_compact_prompt_never_mentions_full_schema_or_dataset_terms():
    prompt = build_hf_local_compact_prompt(_BASELINE_MCI_QUESTION)
    forbidden_terms = [
        "StructuredIntent", "question_type", "target_population",
        "target_etiology", "cohort_filter", "cohort_column", "group_column",
        "NACCUDSD", "NACCALZD",
    ]
    for term in forbidden_terms:
        assert term not in prompt, f"prompt leaked forbidden term {term!r}"
    assert '"intent":' in prompt
    assert "modality_overlap" in prompt
    assert "aggregate_stat" in prompt
    assert "unsupported" in prompt


# ---------------------------------------------------------------------------
# Malformed JSON / contamination — unchanged fail-closed behavior
# ---------------------------------------------------------------------------


def test_hf_local_compact_malformed_json_repairs_on_retry():
    malformed = '{\n  "intent": "count",\n  "population": "baseline_mci",\n  "'
    stub = _SequencedStub([malformed, _compact()])

    intent = _parse_hf_local_compact(_BASELINE_MCI_QUESTION, stub, catalog=[])

    assert len(stub.complete_text_calls) == 2
    assert intent.question_type == QuestionType.COHORT_SIZE
    assert intent.target_population == ["mci"]


def test_hf_local_compact_malformed_json_twice_abstains_without_tool_execution(caplog):
    malformed = '{\n  "intent": "count",\n  "population": "baseline_mci",\n  "'
    stub = _StubPlannerClient(complete_text_return=malformed)

    with caplog.at_level("DEBUG", logger="alz_agent.agent.hf_local_intent"):
        with pytest.raises(HFLocalPlannerAbstention, match="repair/retry"):
            _parse_hf_local_compact(_BASELINE_MCI_QUESTION, stub, catalog=[])

    assert len(stub.complete_text_calls) == 2
    assert any("abstain" in rec.getMessage().lower() for rec in caplog.records)


def test_hf_local_compact_rejects_full_structured_intent_contamination():
    contaminated = json.dumps({
        "question_type": "cohort_size",
        "target_population": ["mci"],
        "cohort_filter": {"NACCUDSD": 4},
    })
    data = parse_hf_local_compact_json(contaminated)
    with pytest.raises(HFLocalCompactValidationError, match="forbidden"):
        validate_hf_local_compact_intent(data)

    stub = _StubPlannerClient(complete_text_return=contaminated)
    with pytest.raises(HFLocalPlannerAbstention):
        _parse_hf_local_compact(_BASELINE_MCI_QUESTION, stub, catalog=[])
    assert len(stub.complete_text_calls) == 2


def test_hf_local_compact_rejects_unsupported_enum_values():
    with pytest.raises(HFLocalCompactValidationError, match="unsupported population"):
        validate_hf_local_compact_intent(json.loads(_compact(population="baseline_ad_only")))
    with pytest.raises(HFLocalCompactValidationError, match="unsupported intent"):
        validate_hf_local_compact_intent(json.loads(_compact(intent="please_count_them")))
    with pytest.raises(HFLocalCompactValidationError, match="unsupported etiology"):
        validate_hf_local_compact_intent(json.loads(_compact(etiology="vascular")))
    with pytest.raises(HFLocalCompactValidationError, match="unsupported modality"):
        validate_hf_local_compact_intent(json.loads(_compact(modalities=["genomics"])))


def test_hf_local_compact_modality_overlap_requires_modalities():
    with pytest.raises(HFLocalCompactValidationError, match="modality_overlap"):
        validate_hf_local_compact_intent(json.loads(_compact(intent="modality_overlap", modalities=[])))


def test_hf_local_compact_aggregate_stat_requires_stat_variable():
    with pytest.raises(HFLocalCompactValidationError, match="aggregate_stat"):
        validate_hf_local_compact_intent(
            json.loads(_compact(intent="aggregate_stat", stat_variable="none"))
        )


# ---------------------------------------------------------------------------
# Explicit intent="unsupported" abstains immediately — no retry burned
# ---------------------------------------------------------------------------


def test_intent_unsupported_abstains_without_retry():
    """A single queued response — if the code tried to retry, the second
    complete_text call would raise IndexError, failing the test."""
    stub = _SequencedStub([_compact(intent="unsupported")])

    with pytest.raises(HFLocalPlannerAbstention, match="unsupported"):
        _parse_hf_local_compact("Some out-of-scope research question.", stub, catalog=[])

    assert len(stub.complete_text_calls) == 1


# ---------------------------------------------------------------------------
# debug_sink — raw output capture for --show-raw-intent
# ---------------------------------------------------------------------------


def test_debug_sink_captures_raw_attempt_on_success():
    sink: list[str] = []
    stub = _StubPlannerClient(complete_text_return=_compact())
    _parse_hf_local_compact(_BASELINE_MCI_QUESTION, stub, catalog=[], debug_sink=sink)
    assert sink == [_compact()]


def test_debug_sink_captures_both_raw_attempts_on_abstain():
    sink: list[str] = []
    malformed = "not json at all"
    stub = _SequencedStub([malformed, malformed])
    with pytest.raises(HFLocalPlannerAbstention):
        _parse_hf_local_compact(_BASELINE_MCI_QUESTION, stub, catalog=[], debug_sink=sink)
    assert sink == [malformed, malformed]


# ---------------------------------------------------------------------------
# Regression 1: "Count participants with baseline MCI. Do not infer AD etiology."
# ---------------------------------------------------------------------------


def test_regression_1_baseline_mci_no_etiology():
    stub = _StubPlannerClient(complete_text_return=_compact())
    intent = _parse_hf_local_compact(_BASELINE_MCI_QUESTION, stub, catalog=[])
    assert intent.question_type == QuestionType.COHORT_SIZE
    assert intent.target_population == ["mci"]
    assert intent.target_etiology == []


# ---------------------------------------------------------------------------
# Regression 2: "Count participants with baseline dementia, regardless of
# Alzheimer etiology." — must not silently return baseline_mci.
# ---------------------------------------------------------------------------


def test_regression_2_baseline_dementia_no_etiology():
    q = "Count participants with baseline dementia, regardless of Alzheimer etiology."
    stub = _StubPlannerClient(
        complete_text_return=_compact(population="baseline_dementia")
    )
    intent = _parse_hf_local_compact(q, stub, catalog=[])
    assert intent.question_type == QuestionType.COHORT_SIZE
    assert intent.target_population == ["dementia"]
    assert intent.target_population != ["mci"]
    assert intent.target_etiology == []


def test_regression_2_guard_rejects_baseline_mci_default_on_retry_and_abstains():
    """Reproduces the exact reported failure mode: an unreliable model keeps
    defaulting to population='baseline_mci' regardless of the question. Both
    attempts return the wrong population -> must abstain, never silently
    return the baseline_mci cohort_size."""
    q = "Count participants with baseline dementia, regardless of Alzheimer etiology."
    wrong = _compact(population="baseline_mci")
    stub = _SequencedStub([wrong, wrong])
    with pytest.raises(HFLocalPlannerAbstention):
        _parse_hf_local_compact(q, stub, catalog=[])
    assert len(stub.complete_text_calls) == 2


def test_regression_2_guard_recovers_via_repair_retry():
    """Attempt 1 wrongly defaults to baseline_mci; the repair prompt (which
    includes the rejection reason) lets the model self-correct on attempt 2."""
    q = "Count participants with baseline dementia, regardless of Alzheimer etiology."
    wrong = _compact(population="baseline_mci")
    right = _compact(population="baseline_dementia")
    stub = _SequencedStub([wrong, right])
    intent = _parse_hf_local_compact(q, stub, catalog=[])
    assert intent.target_population == ["dementia"]


# ---------------------------------------------------------------------------
# Regression 3: "Count participants diagnosed with etiologic Alzheimer's
# disease, not all-cause dementia." — must not return baseline_mci or execute
# compute_cohort_size for baseline_mci; must not claim etiologic AD with
# target_etiology=[].
# ---------------------------------------------------------------------------

_REGRESSION_3_QUESTION = (
    "Count participants diagnosed with etiologic Alzheimer's disease, "
    "not all-cause dementia."
)


def test_regression_3_well_formed_etiologic_ad_count():
    stub = _StubPlannerClient(
        complete_text_return=_compact(population="unspecified", etiology="ad")
    )
    intent = _parse_hf_local_compact(_REGRESSION_3_QUESTION, stub, catalog=[])
    assert intent.question_type == QuestionType.COHORT_SIZE
    assert intent.target_population != ["mci"]
    assert intent.target_etiology == ["ad"]


def test_regression_3_explicit_unsupported_abstains_cleanly():
    stub = _SequencedStub([_compact(intent="unsupported")])
    with pytest.raises(HFLocalPlannerAbstention):
        _parse_hf_local_compact(_REGRESSION_3_QUESTION, stub, catalog=[])
    assert len(stub.complete_text_calls) == 1


def test_regression_3_guard_rejects_baseline_mci_reproduction_of_original_bug():
    """The exact bug this task fixes: the model emits count/baseline_mci/
    etiology=none for a question that is clearly about etiologic AD, not MCI.
    Both attempts reproduce the bug -> must abstain, never execute
    compute_cohort_size for baseline_mci."""
    buggy = _compact(population="baseline_mci", etiology="none")
    stub = _SequencedStub([buggy, buggy])
    with pytest.raises(HFLocalPlannerAbstention):
        _parse_hf_local_compact(_REGRESSION_3_QUESTION, stub, catalog=[])
    assert len(stub.complete_text_calls) == 2


def test_regression_3_end_to_end_never_executes_baseline_mci_tool():
    """Full pipeline check via run_agent: an abstention means tool_results is
    never even constructed (the exception propagates out of parse_intent)."""
    from unittest.mock import MagicMock

    from alz_agent.agent.orchestrator import run_agent
    from alz_agent.backend.fixture_backend import build_fixture_backend

    buggy = _compact(population="baseline_mci", etiology="none")
    client = LocalHFPlannerClient("Qwen/Qwen2.5-3B-Instruct")
    client.complete_text = MagicMock(side_effect=[buggy, buggy])
    backend = build_fixture_backend(n_normal=50, n_mci=40, n_dementia=30, n_other=20)

    with pytest.raises(HFLocalPlannerAbstention):
        run_agent(_REGRESSION_3_QUESTION, [], backend, client=client)


# ---------------------------------------------------------------------------
# Regression 4: "How many participants have both cognition and any imaging
# data?" — modalities must be populated; must not collapse to a plain
# baseline_mci cohort_size.
# ---------------------------------------------------------------------------

_REGRESSION_4_QUESTION = "How many participants have both cognition and any imaging data?"


def test_regression_4_modality_overlap_no_population_restriction():
    stub = _StubPlannerClient(
        complete_text_return=_compact(
            intent="modality_overlap",
            population="unspecified",
            modalities=["cognition", "any_imaging"],
        )
    )
    intent = _parse_hf_local_compact(_REGRESSION_4_QUESTION, stub, catalog=[])
    assert intent.question_type == QuestionType.COHORT_SIZE
    assert intent.target_population == []
    assert intent.target_population != ["mci"]
    assert "cognition" in intent.modalities
    assert "mri" in intent.modalities and "pet" in intent.modalities  # any_imaging expanded


def test_regression_4_guard_rejects_missing_modalities():
    stub = _SequencedStub([_compact(population="unspecified", modalities=[])] * 2)
    with pytest.raises(HFLocalPlannerAbstention):
        _parse_hf_local_compact(_REGRESSION_4_QUESTION, stub, catalog=[])


def test_regression_4_end_to_end_runs_more_than_plain_cohort_size():
    from unittest.mock import MagicMock

    from alz_agent.agent.orchestrator import run_agent
    from alz_agent.backend.fixture_backend import build_fixture_backend

    client = LocalHFPlannerClient("Qwen/Qwen2.5-3B-Instruct")
    client.complete_text = MagicMock(return_value=_compact(
        intent="modality_overlap",
        population="unspecified",
        modalities=["cognition", "any_imaging"],
    ))
    backend = build_fixture_backend(n_normal=50, n_mci=40, n_dementia=30, n_other=20)
    response = run_agent(_REGRESSION_4_QUESTION, [], backend, client=client)

    assert len(response.tool_results) == 2  # cohort_size + modality_overlap, not just one


# ---------------------------------------------------------------------------
# Regression 5: "Among baseline MCI participants, how many have both
# cognition and MRI data?" — population=baseline_mci AND modalities populated.
# ---------------------------------------------------------------------------

_REGRESSION_5_QUESTION = (
    "Among baseline MCI participants, how many have both cognition and MRI data?"
)


def test_regression_5_population_and_modalities_both_populated():
    stub = _StubPlannerClient(
        complete_text_return=_compact(
            population="baseline_mci", modalities=["cognition", "mri"],
        )
    )
    intent = _parse_hf_local_compact(_REGRESSION_5_QUESTION, stub, catalog=[])
    assert intent.target_population == ["mci"]
    assert "cognition" in intent.modalities
    assert "mri" in intent.modalities


def test_regression_5_end_to_end_does_not_invoke_only_compute_cohort_size():
    from unittest.mock import MagicMock

    from alz_agent.agent.orchestrator import run_agent
    from alz_agent.backend.fixture_backend import build_fixture_backend

    client = LocalHFPlannerClient("Qwen/Qwen2.5-3B-Instruct")
    client.complete_text = MagicMock(return_value=_compact(
        population="baseline_mci", modalities=["cognition", "mri"],
    ))
    backend = build_fixture_backend(n_normal=50, n_mci=40, n_dementia=30, n_other=20)
    response = run_agent(_REGRESSION_5_QUESTION, [], backend, client=client)

    assert len(response.tool_results) == 2


# ---------------------------------------------------------------------------
# Regression 6: "In NACC dataset, among those who were diagnosed as AD, what
# is the average age when they were first diagnosed?" — event-anchored stat;
# the compact schema cannot express it correctly (no event_anchor field), so
# it must always abstain rather than compute a plain (wrong) baseline stat.
# ---------------------------------------------------------------------------

_REGRESSION_6_QUESTION = (
    "In NACC dataset, among those who were diagnosed as AD, what is the "
    "average age when they were first diagnosed?"
)


def test_regression_6_event_anchored_stat_always_abstains():
    plausible_but_wrong = _compact(
        intent="aggregate_stat", population="unspecified", etiology="ad",
        stat_variable="age", stat_type="mean",
    )
    stub = _SequencedStub([plausible_but_wrong, plausible_but_wrong])
    with pytest.raises(HFLocalPlannerAbstention):
        _parse_hf_local_compact(_REGRESSION_6_QUESTION, stub, catalog=[])
    assert len(stub.complete_text_calls) == 2


def test_regression_6_never_parses_as_baseline_mci_cohort_size():
    """Even a maximally-wrong model output for this question must not slip
    through as a baseline_mci cohort_size — the mean/average guard forces
    intent to be aggregate_stat, and aggregate_stat is then blocked by the
    event-anchor guard."""
    wrong = _compact(population="baseline_mci")  # intent="count" default
    stub = _SequencedStub([wrong, wrong])
    with pytest.raises(HFLocalPlannerAbstention):
        _parse_hf_local_compact(_REGRESSION_6_QUESTION, stub, catalog=[])


def test_regression_6_plain_aggregate_stat_without_event_anchor_is_supported():
    """Sanity check: aggregate_stat itself is supported when the question is
    NOT event-anchored (e.g. no 'first diagnosed' phrasing)."""
    q = "What is the average age of baseline MCI participants?"
    stub = _StubPlannerClient(
        complete_text_return=_compact(
            intent="aggregate_stat", population="baseline_mci",
            stat_variable="age", stat_type="mean",
        )
    )
    intent = _parse_hf_local_compact(q, stub, catalog=[])
    assert intent.question_type == QuestionType.AGGREGATE_STAT
    assert intent.stat_variable == "age"
    assert intent.event_anchor is None


# ---------------------------------------------------------------------------
# Additional semantic-guard coverage (P1 rules not covered above)
# ---------------------------------------------------------------------------


def test_semantic_guard_adni_mention_requires_dataset_adni():
    q = "How many participants in the ADNI dataset have baseline MCI?"
    compact = validate_hf_local_compact_intent(
        json.loads(_compact(population="baseline_mci", dataset="nacc"))
    )
    assert _semantic_mismatch_reason(q, compact) is not None

    compact_ok = validate_hf_local_compact_intent(
        json.loads(_compact(population="baseline_mci", dataset="adni"))
    )
    assert _semantic_mismatch_reason(q, compact_ok) is None


def test_regression_adni_dataset_maps_to_id_space_b():
    q = "How many participants in the ADNI dataset have baseline MCI?"
    stub = _StubPlannerClient(
        complete_text_return=_compact(population="baseline_mci", dataset="adni")
    )
    intent = _parse_hf_local_compact(q, stub, catalog=[])
    assert intent.id_space == IdSpace.B


def test_semantic_guard_etiology_negation_phrases_are_recognized():
    compact_no_etiology = validate_hf_local_compact_intent(json.loads(_compact(etiology="none")))
    assert _semantic_mismatch_reason(
        "Count participants with baseline MCI. Do not infer AD etiology.",
        compact_no_etiology,
    ) is None
    assert _semantic_mismatch_reason(
        "Count participants with baseline dementia, regardless of Alzheimer etiology.",
        validate_hf_local_compact_intent(json.loads(_compact(population="baseline_dementia"))),
    ) is None
    # Mention without negation must require etiology="ad"
    assert _semantic_mismatch_reason(
        "How many participants have an Alzheimer's diagnosis?",
        compact_no_etiology,
    ) is not None


# ---------------------------------------------------------------------------
# Dispatch + shared-execution coverage (kept from the prior implementation)
# ---------------------------------------------------------------------------


def test_parse_intent_dispatches_local_hf_planner_client_to_compact_path():
    from unittest.mock import MagicMock

    client = LocalHFPlannerClient("Qwen/Qwen2.5-3B-Instruct")
    client.complete_text = MagicMock(return_value=_compact())

    intent = parse_intent(_BASELINE_MCI_QUESTION, client)

    assert client.complete_text.call_count == 1
    assert intent.question_type == QuestionType.COHORT_SIZE
    assert intent.target_population == ["mci"]
    assert intent.target_etiology == []

    sent_prompt = (
        client.complete_text.call_args.kwargs.get("messages")
        or client.complete_text.call_args.args[0]
    )
    prompt_text = sent_prompt[0]["content"]
    assert "plan_analysis" not in prompt_text
    assert "fill_intent" not in prompt_text
    assert "StructuredIntent" not in prompt_text
    assert "NACCUDSD" not in prompt_text


def test_parse_intent_forwards_debug_sink_only_for_hf_local():
    """debug_sink must have zero effect on non-hf-local clients (Deepseek/
    Anthropic path is untouched by this parameter)."""
    stub = _StubPlannerClient(complete_text_return="not used by this test")
    sink: list[str] = []
    # A plain PlannerClient (not LocalHFPlannerClient) goes through
    # _parse_with_llm, which does not accept/use debug_sink at all.
    with pytest.raises(Exception):
        parse_intent("some question", stub, debug_sink=sink)
    assert sink == []  # never touched


def test_hf_local_compact_backend_execution_matches_deterministic_filter_spec():
    from alz_agent.agent.orchestrator import run_agent
    from alz_agent.backend.fixture_backend import build_fixture_backend

    stub = _StubPlannerClient(complete_text_return=_compact())
    compact_intent = _parse_hf_local_compact(_BASELINE_MCI_QUESTION, stub, catalog=[])

    backend = build_fixture_backend(n_normal=50, n_mci=40, n_dementia=30, n_other=20)
    response = run_agent(_BASELINE_MCI_QUESTION, [], backend, intent=compact_intent)

    assert response.grounded
    assert len(response.tool_results) == 1
    assert response.tool_results[0].grounded


def test_hf_local_compact_baseline_mci_never_maps_to_dementia_code():
    compact = validate_hf_local_compact_intent(json.loads(_compact()))
    intent = compact_intent_to_structured_intent(compact)

    assert intent.target_population == ["mci"]
    assert intent.target_population != ["dementia"]
    assert intent.target_etiology == []
    assert intent.cohort_filter is None

    from alz_agent.agent.tool_loop import _build_filter_spec

    filter_spec = _build_filter_spec(intent)
    assert filter_spec.baseline_diagnosis == ["mci"]


# ---------------------------------------------------------------------------
# Follow-up audit (commit db5c3bb, Qwen2.5-3B / fixture backend): the raw
# parse + semantic guard were correct, but two residual bugs survived into
# the adapted StructuredIntent / final answer:
#   1. stat_type="mean" leaked into COHORT_SIZE intents (meaningless for a
#      count question).
#   2. "any imaging" (an OR of MRI/PET) was flattened into a plain
#      ["cognition","mri","pet"] AND/intersection by the adapter, and
#      synthesis then answered with the *total* cohort size instead of the
#      requested overlap.
# ---------------------------------------------------------------------------


# --- Acceptance test 1: etiologic AD should not become baseline MCI, and
# count intents must not carry aggregate-stat leftovers -------------------


def test_acceptance_1_etiologic_ad_repair_path_clears_stat_leftovers():
    """Reproduces the exact repair-path scenario from commit db5c3bb: attempt 1
    (population='baseline_mci') is rejected by the dementia guard, attempt 2
    (population='baseline_dementia') is accepted. The adapted intent must not
    carry stat_type='mean' even though every raw compact object in this
    exchange set stat_type='mean' (the schema default when not aggregate_stat)."""
    q = _REGRESSION_3_QUESTION
    attempt_1 = _compact(population="baseline_mci", etiology="ad")  # rejected
    attempt_2 = _compact(population="baseline_dementia", etiology="ad")  # accepted
    stub = _SequencedStub([attempt_1, attempt_2])

    intent = _parse_hf_local_compact(q, stub, catalog=[])

    assert intent.target_population == ["dementia"]
    assert intent.target_population != ["mci"]
    assert intent.target_etiology == ["ad"]
    assert intent.question_type == QuestionType.COHORT_SIZE
    assert intent.stat_variable is None
    assert intent.stat_type is None
    assert "ad" in intent.candidate_label.lower() or "etiologic" in intent.candidate_label.lower()


def test_acceptance_5_count_and_modality_overlap_never_carry_stat_type_mean():
    """For any count/modality_overlap compact object, regardless of the raw
    stat_type value (schema default 'mean' when the field wasn't otherwise
    populated), the adapted intent must not carry it forward."""
    for intent_kind, extra in (
        ("count", {}),
        ("modality_overlap", {"modalities": ["cognition", "mri"]}),
    ):
        compact = validate_hf_local_compact_intent(
            json.loads(_compact(intent=intent_kind, stat_type="mean", **extra))
        )
        structured = compact_intent_to_structured_intent(compact)
        assert structured.stat_variable is None
        assert structured.stat_type is None


# --- Acceptance test 2: "any imaging" must preserve OR semantics ----------


def test_acceptance_2_any_imaging_preserves_or_semantics_in_adapter():
    compact = validate_hf_local_compact_intent(
        json.loads(_compact(
            intent="modality_overlap", population="unspecified",
            modalities=["cognition", "any_imaging"],
        ))
    )
    intent = compact_intent_to_structured_intent(compact)

    assert intent.question_type == QuestionType.COHORT_SIZE
    assert intent.any_of_modalities is not None
    assert sorted(intent.any_of_modalities) == ["mri", "pet"]
    # The flat list still carries mri/pet (the deterministic tool needs the
    # concrete Modality names to query), but any_of_modalities marks them as
    # an OR group rather than a plain AND/intersection.
    assert sorted(intent.modalities) == ["cognition", "mri", "pet"]
    assert "any imaging" in intent.candidate_label.lower()
    assert "mri" in intent.candidate_label.lower()
    assert "pet" in intent.candidate_label.lower()
    assert " or " in intent.candidate_label.lower()


def test_acceptance_2_guard_rejects_flattened_intersection_for_any_imaging_phrasing():
    """The exact bug from commit db5c3bb: raw hf-local correctly emitted
    modalities=["cognition","any_imaging"], but a buggy/unreliable model
    output of plain ["cognition","mri","pet"] (no "any_imaging" token) for
    the same 'any imaging' question must be rejected by the semantic guard,
    not silently accepted as an AND/intersection."""
    flattened = _compact(
        intent="modality_overlap", population="unspecified",
        modalities=["cognition", "mri", "pet"],
    )
    stub = _SequencedStub([flattened, flattened])
    with pytest.raises(HFLocalPlannerAbstention):
        _parse_hf_local_compact(_REGRESSION_4_QUESTION, stub, catalog=[])
    assert len(stub.complete_text_calls) == 2


def test_acceptance_2_end_to_end_never_answers_with_unrestricted_total():
    """Full pipeline: given tool facts where the total cohort (195) is much
    larger than the correct cognition-AND-(MRI-OR-PET) union (29), the final
    answer must not be 195."""
    from unittest.mock import MagicMock

    from alz_agent.agent.orchestrator import run_agent
    from alz_agent.backend.fixture_backend import (
        FixtureDataBackend,
        FixtureParticipant,
        FixtureVisit,
    )
    from alz_agent.schemas import CatalogEntry, Modality, SourceDataset

    # Hand-build a fixture backend where the union (29) is small and
    # distinct from the total cohort (195), so a "just report the total"
    # bug is detectable: 29 participants have cognition+MRI; one participant
    # has PET alone (so "pet" is a known fixture modality, avoiding an
    # unrelated "modality not in fixture data" abstain) but does not overlap
    # with cognition, so cognition,pet=0 and cognition,mri,pet=0 — the union
    # collapses to exactly 29. The remaining participants have no modality data.
    def _mods(i: int) -> frozenset:
        if i < 29:
            return frozenset({"cognition", "mri"})
        if i == 194:
            return frozenset({"pet"})
        return frozenset()

    participants = [
        FixtureParticipant(
            pid=f"p{i}",
            visits=[FixtureVisit(vnum=1, year=2015, diagnosis=1)],
            modalities=_mods(i),
        )
        for i in range(195)
    ]
    backend = FixtureDataBackend(participants=participants)
    catalog = [
        CatalogEntry(
            variable_id=f"NACC.{mod.upper()}", concept=mod,
            source_dataset=SourceDataset.NACC, id_space=IdSpace.A,
            raw_code=mod.upper(), release_version="nacc70",
            modality=Modality(mod),
        )
        for mod in ("cognition", "mri", "pet")
    ]

    client = LocalHFPlannerClient("Qwen/Qwen2.5-3B-Instruct")
    client.complete_text = MagicMock(return_value=_compact(
        intent="modality_overlap", population="unspecified",
        modalities=["cognition", "any_imaging"],
    ))

    response = run_agent(_REGRESSION_4_QUESTION, catalog, backend, client=client)

    assert "195" not in response.answer
    assert "N=29" in response.answer


# --- Acceptance test 3: explicit all-modality question stays AND ----------

_ACCEPTANCE_3_QUESTION = "How many participants have cognition, MRI, and PET data?"


def test_acceptance_3_explicit_all_modality_stays_and():
    compact = validate_hf_local_compact_intent(
        json.loads(_compact(
            intent="count", population="unspecified",
            modalities=["cognition", "mri", "pet"],
        ))
    )
    intent = compact_intent_to_structured_intent(compact)

    assert intent.any_of_modalities is None
    assert sorted(intent.modalities) == ["cognition", "mri", "pet"]
    label = intent.candidate_label.lower()
    assert "cognition" in label and "mri" in label and "pet" in label
    assert "any imaging" not in label


def test_acceptance_3_guard_rejects_any_imaging_token_for_explicit_and_phrasing():
    """The reverse mistake: the question explicitly asks for MRI and PET
    together, but the model uses the OR-token 'any_imaging' instead of
    listing them — must be rejected, not silently treated as an OR."""
    q = "How many participants have both MRI and PET in addition to cognition?"
    wrong = _compact(
        intent="modality_overlap", population="unspecified",
        modalities=["cognition", "any_imaging"],
    )
    stub = _SequencedStub([wrong, wrong])
    with pytest.raises(HFLocalPlannerAbstention):
        _parse_hf_local_compact(q, stub, catalog=[])


# --- Acceptance test 4: synthesis must not use total cohort as overlap ----


def _overlap_tool_result(counts: dict) -> ToolResult:
    return ToolResult.ok(counts)


def test_acceptance_4_resolve_modality_union_computes_correct_union():
    intent = StructuredIntent(
        target_population=[], target_etiology=[], candidate_label="x",
        question_type=QuestionType.COHORT_SIZE,
        modalities=["cognition", "mri", "pet"], any_of_modalities=["mri", "pet"],
    )
    counts = {
        "cognition": 150, "mri": 60, "pet": 40,
        "cognition,mri": 29, "cognition,pet": 0, "mri,pet": 10,
        "cognition,mri,pet": 0,
    }
    assert resolve_modality_union(intent, counts) == 29  # 29 + 0 - 0


def test_acceptance_4_resolve_modality_union_abstains_when_cells_missing():
    intent = StructuredIntent(
        target_population=[], target_etiology=[], candidate_label="x",
        question_type=QuestionType.COHORT_SIZE,
        modalities=["cognition", "mri", "pet"], any_of_modalities=["mri", "pet"],
    )
    # cognition,pet cell missing/unavailable -> must abstain, not guess.
    counts = {"cognition,mri": 29}
    assert resolve_modality_union(intent, counts) is None


def test_acceptance_4_synthesis_does_not_answer_with_unrestricted_total():
    intent = StructuredIntent(
        target_population=[], target_etiology=[],
        candidate_label="count of all participants with cognition + any imaging (mri or pet) data",
        question_type=QuestionType.COHORT_SIZE,
        modalities=["cognition", "mri", "pet"], any_of_modalities=["mri", "pet"],
    )
    total_cohort = ToolResult.ok(195)
    overlap = _overlap_tool_result({
        "cognition": 150, "mri": 60, "pet": 40,
        "cognition,mri": 29, "cognition,pet": 0, "mri,pet": 10,
        "cognition,mri,pet": 0,
    })
    answer = synthesize_hf_local_answer(intent, [total_cohort, overlap], backend_name="FixtureDataBackend")

    assert "195" not in answer
    assert "N=29" in answer
    assert "synthetic fixture data" in answer


def test_acceptance_4_synthesis_abstains_when_union_unavailable_instead_of_using_total():
    intent = StructuredIntent(
        target_population=[], target_etiology=[],
        candidate_label="count of all participants with cognition + any imaging (mri or pet) data",
        question_type=QuestionType.COHORT_SIZE,
        modalities=["cognition", "mri", "pet"], any_of_modalities=["mri", "pet"],
    )
    total_cohort = ToolResult.ok(195)
    # cognition,pet cell unavailable -> union cannot be derived.
    overlap = _overlap_tool_result({"cognition,mri": 29})
    answer = synthesize_hf_local_answer(intent, [total_cohort, overlap], backend_name="FixtureDataBackend")

    assert "195" not in answer
    assert "cannot determine" in answer.lower()


def test_acceptance_4_extra_evidence_digits_whitelists_only_the_derived_union():
    intent = StructuredIntent(
        target_population=[], target_etiology=[], candidate_label="x",
        question_type=QuestionType.COHORT_SIZE,
        modalities=["cognition", "mri", "pet"], any_of_modalities=["mri", "pet"],
    )
    overlap = _overlap_tool_result({
        "cognition,mri": 29, "cognition,pet": 0, "cognition,mri,pet": 0,
    })
    digits = hf_local_extra_evidence_digits(intent, [overlap])
    assert digits == {"29"}


def test_acceptance_4_plain_and_count_answer_is_short_and_deterministic():
    """Case 1's other complaint: final answers should be short/deterministic,
    not long LLM prose, and must state the fixture-data caveat."""
    intent = StructuredIntent(
        target_population=["dementia"], target_etiology=["ad"],
        candidate_label="count of baseline all-cause dementia participants with etiologic AD attribution",
        question_type=QuestionType.COHORT_SIZE,
    )
    cohort_result = ToolResult.ok(42)
    answer = synthesize_hf_local_answer(intent, [cohort_result], backend_name="FixtureDataBackend")

    assert answer == (
        "Fixture backend returned N=42 for count of baseline all-cause "
        "dementia participants with etiologic AD attribution. This is "
        "synthetic fixture data, not a real NACC/ADSP-PHC/ADNI count."
    )
    assert len(answer.splitlines()) == 1  # short, not a multi-paragraph explanation


# ---------------------------------------------------------------------------
# Acceptance test 6: explicit pairwise-OR language ("either MRI or PET",
# "MRI or PET", "MRI/PET", "any MRI or PET") must execute the OR/union path
# without requiring the model to emit the "any_imaging" alias token, and
# must not abstain. Reproduces the exact failing question from the hf-local
# batch: modalities=['cognition','mri','pet'] with no 'any_imaging' token was
# previously rejected by the semantic guard (which then abstained after the
# repair retry also failed) even though the question spells out the OR
# unambiguously.
# ---------------------------------------------------------------------------

_ACCEPTANCE_6_QUESTION = (
    "How many participants have cognition data and either MRI or PET imaging data?"
)


@pytest.mark.parametrize(
    "question",
    [
        "How many participants have cognition data and either MRI or PET imaging data?",
        "How many participants have cognition data and MRI or PET data?",
        "How many participants have cognition data and MRI/PET data?",
        "How many participants have cognition data and any MRI or PET data?",
    ],
)
def test_acceptance_6_explicit_or_language_does_not_reject_flattened_parse(question):
    compact = validate_hf_local_compact_intent(
        json.loads(_compact(
            intent="modality_overlap", population="unspecified",
            modalities=["cognition", "mri", "pet"],
        ))
    )
    assert _semantic_mismatch_reason(question, compact) is None


@pytest.mark.parametrize(
    "question",
    [
        "How many participants have cognition data and either MRI or PET imaging data?",
        "How many participants have cognition data and MRI or PET data?",
        "How many participants have cognition data and MRI/PET data?",
        "How many participants have cognition data and any MRI or PET data?",
    ],
)
def test_acceptance_6_explicit_or_language_normalizes_to_any_of_modalities(question):
    compact = validate_hf_local_compact_intent(
        json.loads(_compact(
            intent="modality_overlap", population="unspecified",
            modalities=["cognition", "mri", "pet"],
        ))
    )
    intent = compact_intent_to_structured_intent(compact, nl=question)

    assert intent.question_type == QuestionType.COHORT_SIZE
    assert intent.any_of_modalities is not None
    assert sorted(intent.any_of_modalities) == ["mri", "pet"]
    assert sorted(intent.modalities) == ["cognition", "mri", "pet"]


def test_acceptance_6_end_to_end_executes_or_path_not_abstain():
    """Full raw-parse -> adapt path (no retry needed): the flattened
    ['cognition','mri','pet'] parse must be accepted on the first attempt
    and produce any_of_modalities, not raise HFLocalPlannerAbstention."""
    stub = _StubPlannerClient(
        complete_text_return=_compact(
            intent="modality_overlap", population="unspecified",
            modalities=["cognition", "mri", "pet"],
        )
    )
    intent = _parse_hf_local_compact(_ACCEPTANCE_6_QUESTION, stub, catalog=[])

    assert len(stub.complete_text_calls) == 1  # accepted on the first attempt, no retry
    assert intent.question_type == QuestionType.COHORT_SIZE
    assert intent.any_of_modalities is not None
    assert sorted(intent.any_of_modalities) == ["mri", "pet"]
    assert sorted(intent.modalities) == ["cognition", "mri", "pet"]
    assert "any imaging" in intent.candidate_label.lower()


def test_acceptance_6_compact_intent_to_structured_intent_without_nl_is_unchanged():
    """Callers that don't pass nl (e.g. existing direct unit tests, or the
    'unsupported' early-return path) keep the prior alias-only behavior —
    no silent behavior change for call sites that predate this fix."""
    compact = validate_hf_local_compact_intent(
        json.loads(_compact(
            intent="modality_overlap", population="unspecified",
            modalities=["cognition", "mri", "pet"],
        ))
    )
    intent = compact_intent_to_structured_intent(compact)
    assert intent.any_of_modalities is None


def test_acceptance_6_vague_any_imaging_phrasing_without_explicit_pair_still_rejected():
    """The pre-existing guard for the vaguer 'any imaging'/'imaging data'
    phrasing (no explicit MRI/PET pair language) must still reject a
    flattened parse and abstain — this fix only covers self-describing OR
    language, not the ambiguous case that still needs the alias token."""
    stub = _SequencedStub([
        _compact(
            intent="modality_overlap", population="unspecified",
            modalities=["cognition", "mri", "pet"],
        ),
        _compact(
            intent="modality_overlap", population="unspecified",
            modalities=["cognition", "mri", "pet"],
        ),
    ])
    with pytest.raises(HFLocalPlannerAbstention):
        _parse_hf_local_compact(_REGRESSION_4_QUESTION, stub, catalog=[])


def test_acceptance_6_explicit_and_phrasing_still_rejects_or_token():
    """Reverse-direction guard must still work: explicit pairwise-OR
    detection must not accidentally suppress the AND-phrasing rejection."""
    q = "How many participants have both MRI and PET in addition to cognition?"
    compact = validate_hf_local_compact_intent(
        json.loads(_compact(
            intent="modality_overlap", population="unspecified",
            modalities=["cognition", "any_imaging"],
        ))
    )
    assert _semantic_mismatch_reason(q, compact) is not None
