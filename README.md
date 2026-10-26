# Alzheimer's Dataset Expert Agent

A local-first research assistant for NACC and ADNI cohort feasibility, follow-up, modality overlap, variable lookup, and modeling questions. The default backend uses synthetic data; real-data questions need researcher-supplied local files.

## Start here

1. Get authorized data using the [dataset download and file guide](docs/DATA_DOWNLOADS.md).
2. Keep the datasets in their configured local folders, or edit [config/paths.yaml](config/paths.yaml).
3. Install Python 3.10 or newer and the project dependencies:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[agent,catalog,rdd,retrieval,test]"
python -m pytest tests -q
~~~

The source has two backends: fixture (synthetic; default) and local (your authorized files). Do not interpret fixture counts as research findings.

## Ask in natural language

### Built-in heuristic parser, no API key

~~~powershell
python scripts/ask_agent.py --mode heuristic --backend fixture -q "For NACC, how many participants have baseline MCI?" --show-intent
python scripts/ask_agent.py --mode heuristic --backend local -q "For ADNI, how many participants have baseline MCI?" --show-intent --show-meta
~~~

Name one dataset in the question and check the printed ID space: NACC is A, ADNI is B. The heuristic parser handles common patterns; it may abstain on complex questions. Use --interactive instead of -q for a question loop.

### Claude API

Set ANTHROPIC_API_KEY in your shell, then run:

~~~powershell
$env:ANTHROPIC_API_KEY = "your-key"
python scripts/ask_agent.py --mode anthropic --backend local -q "For NACC, how many participants have baseline MCI?" --show-intent --show-meta
~~~

### DeepSeek API

The current adapter uses the Anthropic SDK with a DeepSeek endpoint, so the agent extra is required for both API modes:

~~~powershell
$env:DEEPSEEK_API_KEY = "your-key"
python scripts/ask_agent.py --mode deepseek --backend local -q "For ADNI, how many participants have baseline MCI?" --show-intent --show-meta
~~~

### Worked API example: follow-up at 2, 3, and 5 years

Research question: “Among participants with baseline MCI, how many have sufficient follow-up to determine conversion status within 2, 3, or 5 years?”

With authorized NACC data, prefix the question with “In NACC” to fix the intended ID space. Run the command for the API key you configured. The enriched planner preserves all three horizons:

~~~powershell
$question = "In NACC, among participants with baseline MCI, how many have sufficient follow-up to determine conversion status within 2, 3, or 5 years?"
python scripts/ask_agent.py --mode anthropic --backend local --use-enriched-planner -q $question --show-plan --show-meta
python scripts/ask_agent.py --mode deepseek --backend local --use-enriched-planner -q $question --show-plan --show-meta
~~~

The planner represents baseline MCI as NACCUDSD=3, extracts horizons [2, 3, 5], and requests a cohort count plus longitudinal feasibility. Conversion here means all-cause dementia (NACCUDSD=4), not Alzheimer-specific diagnosis. The backend calls a participant **sufficient** when at least two visits span the horizon, **insufficient** when at least two visits span less than the horizon, and **censored** when only one visit is recorded.

The supplied historical NACC example reported 12,260 baseline-MCI participants:

| Horizon | Sufficient | Insufficient | Single-visit “censored” |
| --- | ---: | ---: | ---: |
| 2 years | 6,632 (54.1%) | 1,686 (13.8%) | 3,942 (32.2%) |
| 3 years | 5,162 (42.1%) | 3,156 (25.7%) | 3,942 (32.2%) |
| 5 years | 3,057 (24.9%) | 5,261 (42.9%) | 3,942 (32.2%) |

The 2-year horizon has the largest pool satisfying the follow-up rule. A 5-year horizon allows more time to observe slower conversions while reducing that pool. The large single-visit group raises possible selection bias if follow-up availability is related to cognitive outcomes. Kaplan–Meier or Cox regression could be considered in a future time-to-event analysis; those methods are not run by this command. Counts vary with the local data release and filters. The old screenshot showed a DeepSeek model label that differs from the current adapter's default.

### Optional local Hugging Face mode

This path uses a model through transformers on your computer. It is experimental: documented small-model runs were slow and sometimes misread intent.

~~~powershell
python -m pip install -e ".[hf-local,catalog,rdd,retrieval]"
python scripts/ask_agent.py --mode hf-local --model Qwen/Qwen2.5-3B-Instruct --backend fixture -q "For NACC, count participants with baseline MCI." --show-intent --show-meta
~~~

See [HF local testing notes](docs/HF_LOCAL_TESTING_NOTE.md) before using it with local data.

## Explicit dataset selection without intent parsing

The direct tools use --dataset and do not call a model:

~~~powershell
python scripts/ask_agent.py --direct-tool cohort-size --dataset nacc --baseline-diagnosis mci --backend local --show-meta
python scripts/ask_agent.py --direct-tool cohort-size --dataset adni --baseline-diagnosis mci --backend local --show-meta
~~~

For other direct tools, run python scripts/ask_agent.py --help. The --dataset flag is for direct tools; natural-language questions should name the dataset.

API modes may send question and catalog context to the provider. Counts come from deterministic local tools and small cells are suppressed. Keep participant data and keys out of Git.

## Project documentation

- [Data downloads and configured files](docs/DATA_DOWNLOADS.md)
- [Current status and limitations](docs/CURRENT_STATUS.md)
- [Architecture overview](OVERVIEW.md)
- [Backend contract](docs/BACKEND_CONTRACT.md)
- [Contributor rules](AGENTS.md)
- [Documentation upkeep](docs/MAINTENANCE_GUIDE.md)

