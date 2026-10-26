"""Abstract DataBackend interface (P0).

All implementations must return AGGREGATES ONLY. No method may return subject IDs
or row-level data. suppression must be applied before returning any count.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from alz_agent.schemas import CoverageStats, FilterSpec, ToolResult


class DataBackend(ABC):
    """Abstract base for all data backends.

    Invariant: every returned count has had suppress() applied; no raw subject
    data crosses the backend boundary. Implementations for real data live in
    local_backend.py; synthetic test data lives in fixture_backend.py.
    """

    @abstractmethod
    def count_eligible(self, filter_spec: FilterSpec) -> ToolResult[int]:
        """Return the number of participants matching filter_spec.

        Evidence must include a DataEvidence referencing the filter_spec and
        the source table(s) used. Counts in (0, SUPPRESSION_THRESHOLD) must
        be suppressed; return abstain with reason 'count suppressed — below
        reporting threshold' in that case.
        """

    @abstractmethod
    def modality_intersection(
        self,
        modalities: list[str],
        filter_spec: FilterSpec,
    ) -> ToolResult[dict[str, int]]:
        """Return n participants with ALL of the listed modalities present.

        Result dict maps each modality combination (as sorted comma-joined key)
        to its intersection count. Every count is independently suppressed.
        Abstain if a requested modality is not present in the catalog for the
        given id_space / release.
        """

    @abstractmethod
    def followup_distribution(
        self,
        horizon: int,
        filter_spec: FilterSpec,
    ) -> ToolResult[dict[str, int]]:
        """Return longitudinal follow-up counts for the given horizon (years).

        Result dict must contain at minimum:
          'sufficient': n with enough follow-up to determine conversion
          'censored':   n whose last visit precedes horizon end
          'insufficient': n with repeat visits but too short a span
        All counts are independently suppressed.
        """

    @abstractmethod
    def variable_coverage(self, variable_id: str) -> ToolResult[CoverageStats]:
        """Compute CoverageStats for a single catalog variable.

        Abstain if the variable_id is not found in the local tables for the
        relevant id_space. The returned CoverageStats.last_computed should be
        set to the time the computation ran.
        """

    @abstractmethod
    def aggregate_stat(
        self,
        variable_id: str,
        stat_type: str,
        filter_spec: FilterSpec,
        event_anchor: str | None = None,
    ) -> ToolResult[dict]:
        """Compute an aggregate statistic on a continuous variable over a filtered cohort.

        Args:
            variable_id:  Catalog variable_id (e.g. 'NACC.NACCAGE').
            stat_type:    One of 'mean', 'median', 'std', 'min', 'max'.
            filter_spec:  Cohort filter (diagnosis, etiology, id_space).
            event_anchor: Which visit per participant to use:
                          'first_diagnosis'       — first visit where diagnosis condition is met.
                          'baseline' / None       — first visit (min NACCVNUM).
                          'last_visit'            — last visit (max NACCVNUM).
                          'first_conversion_visit'— first visit where NACCUDSD changed to ≥ 4.

        Returns a ToolResult whose value dict has keys:
            stat_type (str), value (float), n (int).

        Abstains when:
        - n eligible < SUPPRESSION_THRESHOLD
        - variable_id not found in the local tables
        - stat_type is invalid
        - event_anchor / id_space combo is unsupported
        """

    @abstractmethod
    def descriptive_stats(
        self,
        variable_id: str,
        filter_spec: FilterSpec,
        event_anchor: str | None = None,
    ) -> ToolResult[dict]:
        """Compute full descriptive statistics for a variable over a filtered cohort.

        Returns a ToolResult whose value dict has keys:
            n_total, n_missing, count, mean, median, sd, q1, q3, iqr, min_val, max_val, proportion.

        proportion is the fraction of non-missing rows where the value is > 0 (for binary variables).
        Abstains under same conditions as aggregate_stat.
        """

    @abstractmethod
    def variant_subject_counts(
        self,
        variable_ids: list[str],
        filter_spec: FilterSpec,
    ) -> ToolResult[dict[str, int | None]]:
        """Return distinct-subject counts for each of the given variable_ids.

        Counts the number of unique participants (NACCID for Space A, RID for
        Space B) with a non-null value for each variable_id, within the
        filter_spec-eligible cohort. Unlike variable_coverage (row-level), this
        counts subjects — the correct denominator when comparing biomarker/
        imaging variants that may have multiple longitudinal draws per subject.

        Result dict maps variable_id -> suppressed count (None if suppressed).
        A variable_id absent from local data is simply omitted from the result
        (not fabricated as 0). Abstains if no eligible participants or if none
        of the requested variable_ids were found in local data.
        """
