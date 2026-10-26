"""DeterministicExecutor — maps EnrichedAnalysisPlan operations to ToolResults.

No LLM calls.  Maps each PlanOperation.op_type to an existing deterministic tool
function.  The core innovation is fan-out: a longitudinal_feasibility op with
parameters.horizons=[2,3,5] expands into three backend calls and merges results
into a single ToolResult[dict[str, dict[str, int | None]]].

Fan-out is implemented entirely here; no backend interface changes are required.

Usage:
    executor = DeterministicExecutor(catalog, backend)
    ctx = executor.execute(plan, blocked_operations=validation.blocked_operations)
"""

from __future__ import annotations

from typing import Any

from alz_agent.backend.interface import DataBackend
from alz_agent.backend.local_backend import (
    CENSORING_HORIZON_DEPENDENT,
    FOLLOWUP_STATUS_DEFINITIONS,
)
from alz_agent.schemas import (
    CatalogEntry,
    DataEvidence,
    EnrichedAnalysisPlan,
    ExecutionContext,
    FilterSpec,
    IdSpace,
    ToolResult,
)
from alz_agent.tools.deterministic import (
    assess_longitudinal_feasibility,
    assess_modality_overlap,
    check_leakage,
    compute_cohort_size,
    compute_descriptive_stat,
    compute_variable_stat,
    describe_variable_family,
    explore_catalog,
    lookup_variable,
    suggest_modeling_design,
)

# op_types that delegate to the analysis engine (not available in fixture/CI)
_ENGINE_OPS = frozenset({"group_comparison"})


class DeterministicExecutor:
    """Execute EnrichedAnalysisPlan operations against catalog and backend.

    Each op_type is an adapter over an existing deterministic tool function (v0).
    The executor runs operations in order, stores results in ExecutionContext,
    and continues on per-op errors (partial execution is allowed).
    """

    def __init__(
        self,
        catalog: list[CatalogEntry],
        backend: DataBackend,
        *,
        analysis_engine: Any = None,
    ) -> None:
        self._catalog = catalog
        self._backend = backend
        self._engine = analysis_engine

    def execute(
        self,
        plan: EnrichedAnalysisPlan,
        *,
        blocked_operations: list[str] | None = None,
    ) -> ExecutionContext:
        """Execute all non-blocked operations and return ExecutionContext.

        ExecutionContext.results maps output_name → ToolResult.
        On per-op exception, stores an abstain ToolResult and records the error.

        Post-execution validation: if an explore_catalog operation returns 0 results,
        dependent describe_variable_family/variable_stat ops are blocked (discovery failed).
        """
        blocked: set[str] = set(blocked_operations or [])
        ctx = ExecutionContext()

        for op in plan.operations:
            if op.output_name in blocked:
                continue
            try:
                result = self._dispatch(op, plan)
            except Exception as exc:  # noqa: BLE001
                msg = f"executor error in op '{op.output_name}': {exc}"
                result = ToolResult.abstain(msg)
                ctx.errors.append(msg)

            ctx.results[op.output_name] = result
            ctx.operations_run.append(op.output_name)

            # Hard-abstain on failed discovery: explore_catalog() abstains (grounded=False)
            # when nothing matches the keyword — it never returns grounded-with-empty-list.
            # Block all downstream operations in that case to prevent silent substitution
            # of an unrelated variable (e.g. falling back to a stat op from a prior query).
            if op.op_type == "explore_catalog" and not result.grounded:
                keyword = op.parameters.get("keyword", "")
                msg = (
                    f"No catalog entries matched explore_catalog('{keyword}'). "
                    f"Blocking dependent operations to prevent fallback substitution."
                )
                ctx.errors.append(msg)
                # Block ALL remaining ops (conservative safety — prevents unrelated answers)
                for remaining_op in plan.operations:
                    if remaining_op.output_name not in ctx.results:
                        blocked.add(remaining_op.output_name)

        return ctx

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, op: "PlanOperation", plan: EnrichedAnalysisPlan) -> ToolResult:  # type: ignore[name-defined]
        """Route op_type to the appropriate deterministic tool function."""
        p = op.parameters
        filter_spec = self._build_filter_spec(plan, p)

        if op.op_type == "count_eligible":
            return compute_cohort_size(filter_spec, self._backend)

        if op.op_type == "longitudinal_feasibility":
            return self._fan_out_longitudinal(p, filter_spec)

        if op.op_type == "modality_overlap":
            modalities: list[str] = p.get("modalities", [])
            return assess_modality_overlap(
                modalities, filter_spec, self._backend, self._catalog
            )

        if op.op_type == "variable_stat":
            variable_id: str = p["variable_id"]
            stat_type: str = p.get("stat_type", "mean")
            event_anchor: str | None = p.get("event_anchor")
            return compute_variable_stat(
                variable_id, stat_type, filter_spec, self._backend,
                event_anchor=event_anchor,
            )

        if op.op_type == "descriptive_stat":
            variable_id = p["variable_id"]
            event_anchor = p.get("event_anchor")
            return compute_descriptive_stat(
                variable_id, filter_spec, self._backend,
                event_anchor=event_anchor,
            )

        if op.op_type == "lookup_variable":
            query: str = p.get("query", "")
            return lookup_variable(query, self._catalog)

        if op.op_type == "leakage_check":
            feature_set: list[str] = p.get("feature_set", [])
            label_def: str = p.get("label_def", "")
            return check_leakage(feature_set, label_def, self._catalog)

        if op.op_type == "modeling_design":
            horizon: int = int(p.get("horizon", 3))
            return suggest_modeling_design(
                filter_spec, horizon, self._catalog, self._backend
            )

        if op.op_type == "group_comparison":
            return self._dispatch_group_comparison(p, filter_spec)

        if op.op_type == "explore_catalog":
            keyword: str = p.get("keyword", "")
            explore_id_space: IdSpace | None = None
            id_space_param = p.get("id_space")
            if id_space_param:
                try:
                    explore_id_space = IdSpace(id_space_param)
                except ValueError:
                    explore_id_space = None
            return explore_catalog(keyword, self._catalog, id_space=explore_id_space)

        if op.op_type == "describe_variable_family":
            family: str = p.get("family", "")
            return describe_variable_family(
                family, filter_spec.id_space, self._catalog, self._backend,
                filter_spec=filter_spec,
            )

        return ToolResult.abstain(
            f"Unknown op_type '{op.op_type}' — no executor handler registered."
        )

    def _fan_out_longitudinal(
        self,
        parameters: dict[str, Any],
        filter_spec: FilterSpec,
    ) -> ToolResult:
        """Execute longitudinal feasibility for each horizon and merge results.

        Returns ToolResult[dict[str, dict[str, int | None]]] where:
          outer key  → "{horizon}yr" (e.g. "2yr", "3yr", "5yr")
          inner dict → {"sufficient": N, "insufficient": N, "censored": N}

        Evidence from all sub-calls is merged into a single evidence list.
        Partial success is allowed: failed horizons are omitted from the value dict
        and their abstain reasons are recorded in evidence.

        All-horizon abstain → single abstain ToolResult.
        """
        horizons: list[int] = parameters.get("horizons", [3])
        # Accept singular 'horizon' key as a single-element list for LLM robustness
        if not horizons:
            single = parameters.get("horizon")
            if isinstance(single, int) and single > 0:
                horizons = [single]
        if not horizons:
            return ToolResult.abstain(
                "longitudinal_feasibility fan-out: 'horizons' parameter is empty "
                "(pass horizons=[2,3,5] for multi-horizon or horizon=3 for single)."
            )

        merged: dict[str, Any] = {}
        all_evidence: list[Any] = []
        partial_abstains: list[str] = []

        for h in horizons:
            sub = assess_longitudinal_feasibility(h, filter_spec, self._backend)
            key = f"{h}yr"
            if sub.grounded and sub.value is not None:
                merged[key] = sub.value
                all_evidence.extend(sub.evidence)
            else:
                reason = sub.abstain_reason or f"horizon {h}yr abstained"
                partial_abstains.append(reason)

        if not merged:
            return ToolResult.abstain(
                "longitudinal_feasibility fan-out: all horizons abstained. "
                + "; ".join(partial_abstains)
            )

        if partial_abstains:
            # Partial success — include a note in evidence
            all_evidence.append(
                DataEvidence(
                    query_spec={"horizons": horizons, "partial_abstains": partial_abstains},
                    aggregate_ref=(
                        f"executor.fan_out_longitudinal: "
                        f"{len(merged)}/{len(horizons)} horizons succeeded"
                    ),
                )
            )
        elif not all_evidence:
            all_evidence.append(
                DataEvidence(
                    query_spec={"horizons": horizons},
                    aggregate_ref=(
                        f"executor.fan_out_longitudinal: "
                        f"all {len(horizons)} horizons succeeded"
                    ),
                )
            )

        # Attach backend-sourced status definitions so the synthesizer can
        # emit accurate explanations without LLM inference.
        merged["_status_metadata"] = {
            "status_definitions": FOLLOWUP_STATUS_DEFINITIONS,
            "censoring_horizon_dependent": CENSORING_HORIZON_DEPENDENT,
        }

        return ToolResult.ok(merged, evidence=all_evidence)

    def _dispatch_group_comparison(
        self,
        p: dict[str, Any],
        filter_spec: FilterSpec,
    ) -> ToolResult:
        """Delegate group_comparison to the analysis engine via a minimal AnalysisPlan stub."""
        if self._engine is None:
            return ToolResult.abstain(
                "group_comparison requires a LocalAnalysisEngine (not available — "
                "pass analysis_engine= to DeterministicExecutor)."
            )

        from alz_agent.agent.plan_execution import _plan_group_comparison
        from alz_agent.schemas import AnalysisPlan, AnalysisRole

        def _make_role(role: str, raw_code: str | None, label: str) -> AnalysisRole:
            return AnalysisRole(
                role=role,
                researcher_label=label,
                semantic_definition="",
                selected_raw_code=raw_code,
                resolution_status="resolved" if raw_code else "unresolved",
            )

        stub = AnalysisPlan(
            original_question="",
            analysis_type="group_comparison",
            research_intent_summary="",
            group_variable=_make_role(
                "group_variable",
                p.get("group_variable_id"),
                p.get("group_variable_label", ""),
            ),
            outcome_variable=_make_role(
                "outcome_variable",
                p.get("outcome_variable_id"),
                p.get("outcome_variable_label", ""),
            ),
        )

        results = _plan_group_comparison(stub, self._catalog, filter_spec, self._engine)
        # _plan_group_comparison returns list[ToolResult]; return the first one
        if not results:
            return ToolResult.abstain("group_comparison returned no results")
        return results[0]

    def _build_filter_spec(
        self,
        plan: EnrichedAnalysisPlan,
        op_params: dict[str, Any],
    ) -> FilterSpec:
        """Build FilterSpec from op parameters, falling back to plan-level baseline.

        Priority: op_params["filter"] > plan.baseline_definition.

        Etiology routing (critical — see FilterSpec docstring for semantics):
        - filter {"event_etiology": "ad"}: any-visit NACCALZD==1 → event_etiology (explicit)
        - filter {"etiology": "ad"} + event_anchor set (non-baseline): auto-promoted to
          event_etiology, since event-anchored stats (first_diagnosis, etc.) always imply
          any-visit (a first_diagnosis anchor requires scanning all visits, not just visit 1)
        - filter {"etiology": "ad"} without event_anchor: baseline_etiology (first visit only)
        """
        filter_dict: dict[str, Any] = op_params.get("filter", {})
        if not filter_dict:
            filter_dict = plan.baseline_definition

        baseline_diagnosis: list[str] | None = None
        baseline_etiology: list[str] | None = None
        event_etiology: list[str] | None = None

        diag = filter_dict.get("diagnosis") or filter_dict.get("baseline_diagnosis")
        if diag:
            baseline_diagnosis = [diag] if isinstance(diag, str) else list(diag)

        # Explicit event_etiology key: any-visit etiology filter
        event_etio_raw = filter_dict.get("event_etiology")
        if event_etio_raw:
            event_etiology = (
                [event_etio_raw] if isinstance(event_etio_raw, str) else list(event_etio_raw)
            )
        else:
            etio = filter_dict.get("etiology") or filter_dict.get("baseline_etiology")
            if etio:
                # Auto-promote to event_etiology when a non-baseline event_anchor is set.
                # An event_anchor of "first_diagnosis" requires scanning all visits to find
                # the first qualifying row, so baseline_etiology would silently drop
                # participants whose NACCALZD==1 first appeared after their first visit.
                event_anchor_in_params = op_params.get("event_anchor")
                if event_anchor_in_params and event_anchor_in_params != "baseline":
                    event_etiology = [etio] if isinstance(etio, str) else list(etio)
                else:
                    baseline_etiology = [etio] if isinstance(etio, str) else list(etio)

        try:
            id_space_enum = IdSpace(plan.id_space)
        except ValueError:
            id_space_enum = IdSpace.A

        return FilterSpec(
            baseline_diagnosis=baseline_diagnosis,
            baseline_etiology=baseline_etiology,
            event_etiology=event_etiology,
            id_space=id_space_enum,
        )
