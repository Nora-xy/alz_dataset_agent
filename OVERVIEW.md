# Architecture overview

Start with [README.md](README.md) for setup and usage. Current capabilities and
limitations live in [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md); project
rules live in [AGENTS.md](AGENTS.md).

The question path is: intent parsing, plan validation, deterministic tools,
local backend, and evidence-backed answer generation. The heuristic parser
needs no API client; Claude and DeepSeek use the same planner interface; local
Hugging Face inference is optional and experimental.

NACC is ID space A (NACCID). ADNI is ID space B (RID/PTID). The separate
ADSP-PHC R3 source can add harmonized features to A. ADSP_PHC_* tables inside
ADNI remain in B. No verified crosswalk joins A and B.

For source navigation see [docs/TASK_ROUTING.md](docs/TASK_ROUTING.md); for
backend methods see [docs/BACKEND_CONTRACT.md](docs/BACKEND_CONTRACT.md).

