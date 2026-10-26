# Current status

Updated 2026-10-26 for the public code snapshot. This page is the single place
for current capabilities, limits, and verification status. Start with the
[README](../README.md) for installation and examples. The exact data paths and
download requirements are in [DATA_DOWNLOADS.md](DATA_DOWNLOADS.md) and
[config/paths.yaml](../config/paths.yaml). Code and configuration determine
behavior if documentation becomes outdated.

## Data boundaries

| ID space | Sources | Participant key | Use |
| --- | --- | --- | --- |
| A | NACC UDS and the separate ADSP-PHC R3 release | NACCID | NACC cohorts and NACC-linked harmonized features |
| B | ADNI and the ADSP_PHC tables inside its download | RID or PTID | ADNI cohorts and ADNI-linked features |

The project has no verified crosswalk between A and B. It does not join their
participant records or add their cohort counts. Potential participant overlap
between the two sources cannot be checked here, which limits claims about
independent external validation. Research data is supplied locally and is not
part of the public repository.

## Implemented paths

| Area | Current behavior |
| --- | --- |
| Cohort selection | Counts participants using supported diagnosis, etiology, modality, and visit filters within one ID space. |
| Modality overlap | Computes available intersections from configured local tables. Missing or suppressed cells may prevent an exact combined answer. |
| Follow-up feasibility | Groups eligible participants by observed follow-up span for a requested year horizon. |
| Variable discovery | Builds a catalog from configured dictionaries and data files; looks up names, descriptions, coding, and leakage tags where available. |
| Aggregate statistics | Computes supported statistics through deterministic backend operations, subject to data availability and suppression. |
| Modeling support | Provides cohort and feature evidence, leakage checks, cohort-map output, and selected analysis paths. Availability depends on the question and planner path. |
| Evidence | Tool results carry query and source details; unsupported or ungrounded requests should abstain. |

For method signatures and data-operation details, see
[BACKEND_CONTRACT.md](BACKEND_CONTRACT.md). For source navigation, see
[TASK_ROUTING.md](TASK_ROUTING.md).

## Question modes and backends

| CLI mode | Status | Boundary |
| --- | --- | --- |
| heuristic | Built-in natural-language parser; no API key | Handles common patterns and may abstain on complex requests. |
| anthropic | Claude API planner | Requires an API key and sends question and catalog context to the provider. |
| deepseek | DeepSeek API planner | Requires an API key and sends question and catalog context to the provider. |
| hf-local | Optional, experimental local inference | Small-model tests found slow responses and intent errors; inspect the parsed intent before relying on a result. |

The CLI defaults to the synthetic fixture backend. Select the local backend
only after placing authorized data at the configured paths. The enriched
multi-horizon planning example in the README requires an API planner; the
compact hf-local parser does not support every API-planner question. See
[HF_LOCAL_TESTING_NOTE.md](HF_LOCAL_TESTING_NOTE.md) for its tested limits.

## Interpretation rules

- NACC UDS cognitive status 4 and ADNI DXSUM diagnosis 3 represent all-cause
  dementia, not an Alzheimer-specific etiology. The NACC NACCALZD field and
  ADNI DXAD-family fields represent the separate etiologic attribution used
  by supported filters.
- A participant has sufficient observed follow-up for a horizon when at
  least two visits span that horizon; insufficient follow-up means at least
  two visits span less time. The backend labels single-visit participants
  censored. These are project feasibility categories, not a survival-analysis
  estimate or proof that conversion did or did not occur.
- The NACC follow-up calculation uses visit years; the ADNI calculation uses
  years parsed from CDR visit dates. Neither performs Kaplan-Meier or Cox
  analysis. Available counts vary with the authorized data release and filters.
- Small positive cells below 10 are suppressed in returned values. Review
  the evidence limitation below before exposing direct backend results.

## Known limitations and next work

1. The backend follow-up operation still puts unsuppressed bucket counts in
   its evidence text before the transport boundary redacts them. Returned
   values suppress small cells, but direct backend callers could see those
   counts in evidence. Remove the counts at the source before treating direct
   backend output as safe for external sharing.
2. A full catalog coverage scan has not been persisted in the public copy;
   coverage fields may be empty. Some supplementary NACC dictionaries are
   outside the configured core RDD parser.
3. The label-certainty filter is reserved and abstains when requested.
   ADSP-PHC R4 is not configured; the separate source targets R3.
4. The ADNI harmonized diagnosis table ADSP_PHC_DEMODX is configured for
   backend reads but is absent from the catalog table allowlist. Its
   dictionary entries may therefore be missing until the configuration is
   aligned.
5. Local Hugging Face mode remains experimental. The tested small models had
   latency and question-interpretation failures; see its separate testing
   note. More extensive modeling and bounded conversion analyses need their
   own validation before research use.

## Verification record

The clean public copy's fixture suite last ran on 2026-10-25: 1,217 passed
and 49 skipped. The skipped checks included cases needing licensed local
data. These results do not validate a new researcher's download, API key,
model choice, or hardware. Use the README commands to test a local setup;
check the parsed plan and evidence for each real-data answer.
