# Task Routing — alz_dataset_agent

> **Source files remain the source of truth.** Use this file to locate the right
> symbols quickly. Run `rg` commands to confirm names before reading full files.

---

## If the task changes intent parsing

Inspect:
- [src/alz_agent/agent/intent.py](../src/alz_agent/agent/intent.py) — `StructuredIntent`, `QuestionType`, `parse_intent()`, `_parse_heuristic()`, `_parse_with_llm()`, `_normalize_llm_output()`
- [src/alz_agent/agent/hf_local_intent.py](../src/alz_agent/agent/hf_local_intent.py) — hf-local compact-JSON parser front end: `_parse_hf_local_compact()`, `_adapt_compact_to_intent_data()`, `HFLocalPlannerAbstention`. `parse_intent()` dispatches here when `client` is a `LocalHFPlannerClient` — different raw parsing strategy, same `StructuredIntent` output as the DeepSeek/Anthropic and heuristic paths.
- [src/alz_agent/agent/orchestrator.py](../src/alz_agent/agent/orchestrator.py) — `run_agent()`, `_run_tool_loop()`, `_build_filter_spec()`

```powershell
rg "class StructuredIntent|QuestionType|parse_intent|_parse_heuristic|_parse_with_llm|_normalize_llm_output" src/alz_agent/agent/
rg "AGGREGATE_STAT|STATISTICAL_ANALYSIS|FEASIBILITY|MODELING_DESIGN|COHORT_SIZE|LOOKUP" src/alz_agent/agent/intent.py
rg "_parse_hf_local_compact|_adapt_compact_to_intent_data|HFLocalPlannerAbstention|_COMPACT_" src/alz_agent/agent/hf_local_intent.py
```

---

## If the task changes data operations

Inspect:
- [src/alz_agent/backend/interface.py](../src/alz_agent/backend/interface.py) — `DataBackend` ABC, method signatures
- Relevant functions in [src/alz_agent/backend/local_backend.py](../src/alz_agent/backend/local_backend.py)
- [src/alz_agent/tools/deterministic.py](../src/alz_agent/tools/deterministic.py) — the 11 deterministic tools
- [src/alz_agent/schemas.py](../src/alz_agent/schemas.py) — `FilterSpec`, `ToolResult`, `CoverageStats`

```powershell
rg "class LocalDataBackend|def .*cohort|def .*modality|def .*longitudinal|def .*coverage|NACCUDSD|NACCALZD|NACCAGE|suppression|id_space" src/alz_agent/backend/
rg "def count_eligible|def modality_intersection|def followup_distribution|def variable_coverage|def aggregate_stat" src/alz_agent/backend/
```

---

## If the task changes catalog or variable lookup

Inspect:
- [src/alz_agent/catalog/](../src/alz_agent/catalog/) — all parsers and builder
- [src/alz_agent/tools/deterministic.py](../src/alz_agent/tools/deterministic.py) — `lookup_variable()`
- [tests/test_p1_catalog.py](../tests/test_p1_catalog.py) — catalog builder tests
- [tests/test_nacc_rdd_parser.py](../tests/test_nacc_rdd_parser.py) — RDD parser tests

```powershell
rg "CatalogEntry|description|value_coding|concept|alias|RDD|NACCMMSE|NACCAGE" src/alz_agent/catalog/
rg "lookup_variable|Pass 1|Pass 2|Pass 3|Pass 4|Pass 5|_STOPWORDS" src/alz_agent/tools/deterministic.py
rg "NaccRddEntry|parse_nacc_rdd|_parse_rdd_text|short_descriptor" src/alz_agent/catalog/nacc_rdd_parser.py
rg "build_catalog|populate_catalog_coverage|build_retrieval_corpus" src/alz_agent/catalog/builder.py
```

---

## If the task changes CLI behavior

Inspect:
- [scripts/ask_agent.py](../scripts/ask_agent.py) — all CLI modes and backend loading

```powershell
rg "_backend_local|_backend_fixture|_run_direct_tool|_ask_once_nl|_build_arg_parser" scripts/ask_agent.py
rg "direct_tool|backend|show_intent|show_meta|mode|interactive" scripts/ask_agent.py
```

---

## If the task changes retrieval or doc evidence

Inspect:
- [src/alz_agent/retrieval/doc_retriever.py](../src/alz_agent/retrieval/doc_retriever.py) — `retrieve_doc_evidence()`, `_load_corpus()`, BM25 threshold
- [src/alz_agent/catalog/builder.py](../src/alz_agent/catalog/builder.py) — `build_retrieval_corpus()`

```powershell
rg "retrieve_doc_evidence|BM25_RELEVANCE_THRESHOLD|_load_corpus|_tokenize" src/alz_agent/retrieval/
rg "build_retrieval_corpus|retrieval_notes" src/alz_agent/catalog/builder.py
```

---

## If the task changes the boundary / suppression layer

Inspect:
- [src/alz_agent/boundary/transport.py](../src/alz_agent/boundary/transport.py) — `execute_tool_call`, `_enforce_boundary`, `_check_and_redact_value`, `BoundaryViolation`

```powershell
rg "execute_tool_call|_enforce_boundary|BoundaryViolation|_check_and_redact_value|_redact_evidence|SUPPRESSION_THRESHOLD" src/alz_agent/boundary/
```

---

## If the task changes the cohort map

Inspect:
- [src/alz_agent/cohort_map/generator.py](../src/alz_agent/cohort_map/generator.py) — `generate_cohort_map()`, `serialize_cohort_map()`, `deserialize_cohort_map()`
- [src/alz_agent/schemas.py](../src/alz_agent/schemas.py) — `CohortMap`, `Tier`, `CohortRole`

```powershell
rg "generate_cohort_map|serialize_cohort_map|deserialize_cohort_map|CohortMapGenerationReport|FieldProvenance" src/alz_agent/cohort_map/
rg "class CohortMap|class Tier|class CohortRole" src/alz_agent/schemas.py
```

---

## If the task changes the analysis engine

Inspect:
- [src/alz_agent/analysis/engine.py](../src/alz_agent/analysis/engine.py) — `LocalAnalysisEngine.group_comparison()`, `run_generated_code()`, `generate_analysis_code()`

```powershell
rg "class LocalAnalysisEngine|def group_comparison|def run_generated_code|generate_analysis_code|_SAFE_BUILTINS" src/alz_agent/analysis/engine.py
rg "_run_statistical_analysis|_run_group_comparison|_run_generated_analysis" src/alz_agent/agent/orchestrator.py
```

---

## If the task changes the eval harness

Inspect:
- [src/alz_agent/eval/harness.py](../src/alz_agent/eval/harness.py) — `run_eval()`, four category functions
- [src/alz_agent/eval/golden_questions.py](../src/alz_agent/eval/golden_questions.py) — `GOLDEN_QUESTIONS`

```powershell
rg "run_eval|GoldenQuestion|GOLDEN_QUESTIONS|EvalReport|default_catalog|default_backend" src/alz_agent/eval/
```

---

## General symbol search commands

```powershell
# Find any backend method or cohort/modality/longitudinal/coverage function
rg "class LocalDataBackend|def .*cohort|def .*modality|def .*longitudinal|def .*coverage|NACCUDSD|NACCALZD|NACCAGE|suppression|id_space" src/alz_agent

# Find intent and tool dispatch
rg "StructuredIntent|question_type|lookup|modality_overlap|longitudinal|descriptive_stat|candidate_label" src/alz_agent

# Find catalog enrichment symbols
rg "CatalogEntry|description|value_coding|concept|alias|RDD|NACCMMSE|NACCAGE" src/alz_agent

# Find all public tool entry points
rg "^def " src/alz_agent/tools/deterministic.py

# Find suppression usage
rg "suppress|SUPPRESSION_THRESHOLD|SUPPRESSED" src/alz_agent --include="*.py"

# Find all abstain calls
rg "ToolResult.abstain|\.abstain(" src/alz_agent --include="*.py"

# Find id_space checks
rg "id_space.*IdSpace|IdSpace\.A|IdSpace\.B" src/alz_agent --include="*.py"
```

---

## Schema / type quick reference

```powershell
# Key enums and types
rg "class LeakageTag|class Modality|class IdSpace|class Tier|class CohortRole|class SourceDataset|class QuestionType" src/alz_agent

# FilterSpec fields
rg "baseline_diagnosis|baseline_etiology|visit_window|required_modalities|label_certainty" src/alz_agent/schemas.py
```

---

## Where to add things

| What | Where |
|------|-------|
| New Pydantic schema | `src/alz_agent/schemas.py` |
| New backend method | `backend/interface.py` (ABC) + `local_backend.py` + `fixture_backend.py` |
| New deterministic tool | `tools/deterministic.py` |
| New intent question type | `agent/intent.py` (QuestionType enum + heuristic pattern + LLM tool desc) |
| New orchestrator tool dispatch | `agent/orchestrator.py` (`_run_tool_loop`) |
| New leakage rule | `catalog/rules/leakage_rules.yaml` |
| New catalog parser | `catalog/` (new file) + `builder.py` (add to `build_catalog`) |
| New eval golden question | `eval/golden_questions.py` (GOLDEN_QUESTIONS list) |
