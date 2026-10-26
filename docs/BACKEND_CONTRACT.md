# Backend Contract — alz_dataset_agent

> **Source files remain the source of truth.**
> Read `src/alz_agent/backend/interface.py` and `local_backend.py` for the authoritative contract.

---

## Overview

The backend is accessed only through the `DataBackend` ABC (`backend/interface.py`).
All implementations must return aggregates only — no subject IDs, no row-level data.
`suppress()` must be applied before any count is returned.

Two implementations:
- **`LocalDataBackend`** — reads real local CSV/xlsx files; used in production and `--backend local` CLI mode.
- **`FixtureDataBackend`** — in-memory synthetic data; used in all tests and `--backend fixture` CLI mode.

---

## Interface Methods

### `count_eligible(filter_spec: FilterSpec) → ToolResult[int]`

**Purpose:** Count unique participants matching a filter.

**Input:** `FilterSpec` (see below).

**Output:** `ToolResult[int]` — grounded with participant count, or abstain.

**Suppression:** If `0 < n < 10`, returns `ToolResult.abstain("count suppressed — N is below reporting threshold (10)")`.

**Semantics:** Participant-level (deduplicates to one row per participant for diagnosis/etiology filters). The count is of distinct IDs, not visits.

**Dataset/id_space:** Dispatches to `_eligible_nacc_ids` (Space A) or `_eligible_rid_set` (Space B) based on `filter_spec.id_space`.

---

### `modality_intersection(modalities: list[str], filter_spec: FilterSpec) → ToolResult[dict[str, int | None]]`

**Purpose:** Return intersection counts for all non-empty subsets of the requested modalities.

**Input:** `modalities` — list of modality name strings (must be valid `Modality` enum values). `filter_spec` — cohort filter.

**Output:** `ToolResult[dict[str, int | None]]` where keys are sorted comma-joined modality names (e.g. `"cognition,mri"`) and values are counts or `None` (suppressed).

**Suppression:** Each combination count is independently suppressed; suppressed cells become `None`.

**Abstain conditions:**
- Any requested modality is not present in the local data files for the given id_space.
- Filter spec is invalid (unknown dx label, etc.).

**Missing data:** If a modality key is absent from the data files entirely (not just zero intersection), the whole call abstains.

**Dataset/id_space:**
- Space A: checks NACCID presence in ADSP-PHC modality files (configured via `backend.adsp_phc_data_files` in `config/paths.yaml`).
- Space B: checks RID presence in ADNI modality tables (CDR, MMSE, MoCA, ADAS → cognition; ADSP_PHC_CSF → fluid_biomarker; ADSP_PHC_T1_FS → mri; FAMHXPAR/SIB → demographic).

---

### `followup_distribution(horizon: int, filter_spec: FilterSpec) → ToolResult[dict[str, int | None]]`

**Purpose:** Classify eligible participants by longitudinal follow-up relative to a horizon (years).

**Input:** `horizon` (positive integer, years). `filter_spec`.

**Output:** `ToolResult[dict]` with keys:
- `"sufficient"` — n with last visit ≥ horizon years after first visit
- `"insufficient"` — n with ≥2 visits but span < horizon years
- `"censored"` — n with only 1 visit

Each bucket value is `int | None` (None = suppressed).

**Suppression:** Each bucket is independently suppressed.

**Dataset/id_space:**
- Space A (NACC): uses `VISITYR` column; computes `max(VISITYR) - min(VISITYR)` per participant.
- Space B (ADNI): uses CDR `VISDATE` column; parsed to year via `pd.to_datetime`.

---

### `variable_coverage(variable_id: str) → ToolResult[CoverageStats]`

**Purpose:** Compute coverage statistics for a single catalog variable.

**Input:** `variable_id` — fully qualified ID in the form `SOURCE.RAWCODE` (e.g. `NACC.NACCAGE`) or `SOURCE.SUBFOLDER.RAWCODE` (e.g. `ADSP_PHC.Cognition.PHC_MEM`).

**Output:** `ToolResult[CoverageStats]` where `CoverageStats` has:
- `n_available: int` — non-null value count
- `longitudinal_availability: int` — participants with ≥2 non-null observations
- `missingness_rate: float` — fraction missing (0.0–1.0)
- `last_computed: datetime | None`

**Abstain conditions:** Variable ID not found in the local data files for the matching source.

**Semantics:** Computed over all rows (all visits), not per-participant baseline. `longitudinal_availability` counts unique IDs with ≥2 non-null observations.

---

### `aggregate_stat(variable_id, stat_type, filter_spec, event_anchor) → ToolResult[dict]`

**Purpose:** Compute a summary statistic on a numeric column over a filtered cohort.

**Input:**
- `variable_id` — `"NACC.RAWCODE"` for Space A, or `"ADNI.<TABLE>.<FIELD>"` for Space B
  (e.g. `"ADNI.ADSP_PHC_PLASMA.PHC_pTau217"`); dispatch is by `filter_spec.id_space`.
  Space B also accepts the derived id `"ADNI.DERIVED.age_at_event"` (see below).
- `stat_type` — one of `"mean"`, `"median"`, `"std"`, `"min"`, `"max"`.
- `filter_spec` — cohort filter.
- `event_anchor` — Space A: `"first_diagnosis"`, `"first_conversion_visit"`, `"last_visit"`, or
  `None`/`"baseline"` (first visit by NACCVNUM). **Space B now supports the diagnosis anchors
  too**, resolved cross-table: `_adni_event_anchor_visits` locates each RID's first qualifying
  DXSUM visit (etiologic AD via `DXAD`/`DXMDUE`/`DXDDUE==1` when `event_etiology`/
  `baseline_etiology` is set; all-cause dementia via `DIAGNOSIS==3` for `first_conversion_visit`
  or otherwise), and `_select_adni_measure_rows` joins the measure table on `(RID, VISCODE)` so
  **any** `ADNI.<TABLE>.<FIELD>` can be measured at that visit (e.g. MMSE at first AD diagnosis).
  `"last_visit"`/`None`/`"baseline"` still resolve via `_select_adni_event_rows` (date-ordered).

**Derived `ADNI.DERIVED.age_at_event`:** participant age at the anchored diagnosis visit.
Primary value = raw age from `PTDEMOG.PTDOB` (month/year precision) and the anchored
`DXSUM.EXAMDATE`; a harmonized cross-check from `ADSP_PHC_DEMODX.PHC_Age_Diagnosis` (joined on
`(RID, VISCODE)`) rides in the evidence `aggregate_ref`/`query_spec`. Etiologic AD and all-cause
dementia are selected via `filter_spec` — callers emit one op per basis to report them separately.

**Output:** `ToolResult[dict]` with keys `stat_type` (str), `value` (float, rounded to 2dp), `n` (int).

**Provenance (WU 2026-07):** each `DataEvidence` carries `source_table`, `source_field`,
`source_level` (`raw`|`harmonized`), and `diagnosis_basis` (e.g. `DXAD/DXMDUE/DXDDUE=1 etiologic
AD (raw DXSUM)`). The derived-age evidence additionally carries `measurement_label`
(`"first observed AD diagnosis in ADNI (study record)"`) and `dob_precision` (`month/year`).

**Suppression:** If `n < 10`, abstains with suppression message.

**Abstain conditions:**
- `variable_id` does not start with `"NACC."` (Space A) or `"ADNI.<TABLE>.<FIELD>"` (Space B)
- `stat_type` invalid
- ADNI `<TABLE>` not registered in `backend.adni_data_files` (config/paths.yaml)
- Column not found in the resolved NACC/ADNI file
- No non-null values for the eligible cohort
- Diagnosis anchor not computable (missing DXSUM / etiology columns / no matching visit) —
  abstains with an explicit **capability/coverage-gap** message, never "data absent".

`descriptive_stats(variable_id, filter_spec, event_anchor)` has the identical Space A/B
dispatch and abstain conditions, returning the full descriptive-stats dict (see schemas.py).

---

### `variant_subject_counts(variable_ids, filter_spec) → ToolResult[dict[str, int | None]]`

**Purpose:** Distinct-**subject** (not row) counts of non-null values for a list of
variable_ids, within one id_space. Backs `describe_variable_family` — the correct
denominator when comparing biomarker/imaging variants that may have multiple
longitudinal draws per subject (row-level counting, as in `variable_coverage`, would
overcount).

**Output:** dict mapping each resolvable `variable_id` → suppressed count (`None` if
suppressed, `0` passes through unsuppressed per `suppress()` semantics). A `variable_id`
that can't be resolved (unregistered table, missing column) is **omitted**, never
fabricated as 0.

**Abstain conditions:**
- No eligible participants for `filter_spec`
- None of the requested `variable_ids` resolved to a column

---

## `FilterSpec` — Input Schema

Defined in `schemas.py`. All fields optional except `id_space`.

| Field | Type | Meaning |
|-------|------|---------|
| `baseline_diagnosis` | `list[str] \| None` | Cognitive-status labels: `"normal"`, `"mci"`, `"dementia"`, `"impaired-not-mci"` (NACC) or `"normal"`, `"mci"`, `"dementia"` (ADNI). `"ad"` is **not** valid here. |
| `baseline_etiology` | `list[str] \| None` | Etiologic attribution: `"ad"` / `"alzheimers"` / `"alzheimer"` / `"alzheimer's disease"`. NACC: `NACCALZD==1`. ADNI: `DXMDUE==1 OR DXDDUE==1 OR DXAD==1`. |
| `visit_window` | `tuple[int,int] \| None` | Filter to participants with ≥1 visit where `NACCVNUM` is in `[lo, hi]`. |
| `required_modalities` | `list[str] \| None` | Require participants to have at least one record in each of these modality data files. |
| `label_certainty` | `str \| None` | Reserved; non-None → abstain on all backends. |
| `id_space` | `IdSpace` | `IdSpace.A` (NACCID, default) or `IdSpace.B` (RID/PTID). |

**NACCUDSD codes** (Space A):
- `"normal"` / `"nc"` / `"1"` → 1
- `"impaired-not-mci"` / `"impaired not mci"` / `"2"` → 2
- `"mci"` / `"3"` → 3
- `"dementia"` / `"4"` → 4

**ADNI DIAGNOSIS codes** (Space B):
- `"normal"` / `"cn"` / `"nc"` / `"nl"` / `"1"` → 1
- `"mci"` / `"2"` → 2
- `"dementia"` / `"3"` → 3

---

## What LocalDataBackend Actually Does

### Data loading
- Lazy-loads files on first access; caches by `(rel_path, frozenset_of_usecols)`.
- Reads CSV with `low_memory=False`; reads xlsx with `pd.read_excel`.
- NACC core loads 5 columns: `NACCID`, `NACCVNUM`, `VISITYR`, `NACCUDSD`, `NACCALZD`.

### Space A eligibility (`_eligible_nacc_ids`)
1. Start with all unique NACCIDs.
2. Build baseline rows (first `NACCVNUM` per participant) when dx or etiology filter is active.
3. Apply `baseline_diagnosis` filter via `NACCUDSD` integer match.
4. Apply `baseline_etiology` filter via `NACCALZD == 1`.
5. Apply `visit_window` filter (any visit in range).
6. Apply `required_modalities` filter (NACCID intersection with ADSP-PHC modality files).

### Space B eligibility (`_eligible_rid_set`)
1. Loads DXSUM with columns: `RID`, `VISCODE`, `DIAGNOSIS`/`DXCURREN`, and optionally `EXAMDATE`, `DXMDUE`, `DXDDUE`, `DXAD`.
2. Selects baseline row per RID via `_select_adni_baseline()`:
   - Prefers EXAMDATE chronological order; falls back to VISCODE lexicographic sort for NaT rows.
3. Apply `baseline_diagnosis` filter via DIAGNOSIS integer match.
4. Apply `baseline_etiology` filter via `DXMDUE==1 OR DXDDUE==1 OR DXAD==1` at the **baseline** row; abstains if no etiology columns present.
5. Apply `event_etiology` filter via `DXAD/DXMDUE/DXDDUE==1` at **any** visit (incident + prevalent AD), mirroring the NACC `event_etiology` vs `baseline_etiology` distinction; abstains if no etiology columns present.
6. Apply `required_modalities` filter (RID intersection with ADNI table files).

### Modality coverage files
- **Space A:** `adsp_phc_data_files` in `config/paths.yaml` maps modality name → list of relative file paths; loads only the `NACCID` column.
- **Space B:** hard-coded table-to-modality mapping in `LocalDataBackend._modality_ids_b()`
  (kept in sync with `catalog.adni_parser._TABLE_MODALITY` — two independent hand-maintained
  maps, must not drift): `CDR/MMSE/MoCA/ADAS/ADSP_PHC_COGN/FAQ/NEUROBAT → cognition`,
  `ADSP_PHC_CSF/ADSP_PHC_PLASMA/UPENN_PLASMA_FUJIREBIO_QUANTERIX/UPENNBIOMK_MASTER/
  UPENNBIOMK_ROCHE_ELECSYS → fluid_biomarker`, `ADSP_PHC_T1_FS/UCSFFSX7/MRIMETA/MRI3META →
  mri`, `AMYMETA/AV45META/TAUMETA/UCBERKELEY_AMY_6MM/UCBERKELEY_TAU_6MM → pet` (PET is now
  mapped for Space B — was previously unsupported), `FAMHXPAR/SIB/PTDEMOG → demographic`.
  `ROSTER`/`REGISTRY`/`ARM`/`VISITS`/`APOERES`/`ALIQUOT_COUNT`/`BIOMARK`/`LABDATA`/`LABTESTS`
  are catalog-searchable (`lookup_variable`/`explore_catalog`) but intentionally excluded
  from modality-overlap computation — they're registry/lab-logistics/genotype tables, not
  a measured clinical modality. Neuropath is still not mapped for Space B.

### Date-agnostic ADNI path resolution
`backend.adni_data_files` values (and `adni.datadic_glob`) are glob patterns relative to
`adni.base_dir`, resolved via `catalog.adni_parser.resolve_latest_snapshot()` — picks the
newest snapshot by parsed `_DDMonYYYY` date token (falling back to file mtime), not
lexicographic filename sort. `LocalDataBackend._resolve_adni_table(tbl)` centralizes this
resolution (cached per table name) for every `adni_data_files` lookup in the class; a fresh
LONI re-download with a different date suffix requires no config changes. See
`docs/CURRENT_STATUS.md` → "ADNI Data Layout" for the researcher-onboarding version of this.

---

## What LocalDataBackend Does Not Do

- **No `label_certainty` filtering** — abstains if non-None.
- **No cross-space queries** — each method dispatches to Space A or Space B; never both.
- **No per-variable coverage cache** — coverage is recomputed on each `variable_coverage` call (no persistent `outputs/catalog.json` yet).

---

## How Missing Data / Unavailable Modalities Are Represented

| Situation | Representation |
|-----------|---------------|
| Suppressed count (0 < n < 10) | `None` in dict value; `ToolResult.abstain` for top-level int |
| Modality absent from data files | `ToolResult.abstain(reason)` — the whole call abstains |
| Column not in CSV | `ToolResult.abstain(reason)` |
| No eligible participants after filter | `ToolResult.abstain("No eligible participants")` or zero-bucket dict |
| ADNI table not in `backend.adni_data_files` | `ToolResult.abstain("... not a configured data file")` |
| Space B diagnosis anchor not computable (missing DXSUM / etiology cols / no matching visit) | `ToolResult.abstain(...)` worded as a **pipeline capability/coverage gap**, explicitly distinguished from missing/low-quality data |

---

## Fixture Backend vs Local Backend

| Property | FixtureDataBackend | LocalDataBackend |
|----------|-------------------|-----------------|
| Data source | In-memory `FixtureParticipant` list (seed-42) | Real CSV/xlsx files on disk |
| `aggregate_stat` | Age variables only (`"age"` / `"naccage"`) | Any NACC column |
| `variable_coverage` | Pre-set `coverage_map` dict only | Full column scan |
| Modality fractions | Configurable per-participant probabilities | ADSP-PHC / ADNI file presence |
| id_space | Configurable (default Space A) | Both; dispatched per FilterSpec |
| Suppression | Same threshold and logic as LocalDataBackend | Same |
| Etiology filter | `FixtureParticipant.baseline_etiology == "ad"` | `NACCALZD==1` (NACC) / `DXMDUE/DXDDUE/DXAD==1` (ADNI) |

The fixture backend does **not** require data files. It is used for all unit and integration tests.

---

## Rules for Adding a New Backend Operation

1. Add an `@abstractmethod` to `DataBackend` in `interface.py` with the same suppression and abstain contract.
2. Implement in both `LocalDataBackend` and `FixtureDataBackend` before considering the feature complete.
3. Every returned count must pass through `suppress()` before return.
4. The method must never return a row-level identifier or raw subject-level data.
5. Add a `DataEvidence` entry with `query_spec` and `aggregate_ref` to every grounded `ToolResult`.
6. Write tests covering: normal case, suppressed case, invalid filter label (abstain), and empty eligible set.
7. If the operation is specific to one id_space, abstain clearly for the other space.
