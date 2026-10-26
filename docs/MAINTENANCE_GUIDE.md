# Documentation upkeep

Use this guide when setup, data paths, supported questions, provider modes,
privacy rules, or known limitations change. It is a review aid for
contributors. Source code and config/paths.yaml establish current behavior;
AGENTS.md contains the project rules.

| Document | Main job |
| --- | --- |
| README.md | Entry point, installation, and runnable examples. |
| AGENTS.md | Durable rules for contributors and coding agents. |
| docs/DATA_DOWNLOADS.md | Authorized dataset access, configured files, and reasons for using them. |
| docs/CURRENT_STATUS.md | One current account of capabilities, limits, and verification. |
| docs/BACKEND_CONTRACT.md | Data-operation contracts. |
| docs/TASK_ROUTING.md | Source navigation. |
| docs/HF_LOCAL_TESTING_NOTE.md | Evidence and cautions for the optional local model mode. |

When something changes:

1. Check the code, configuration, and relevant fixture tests before changing
   a documented claim or example.
2. Update the document that owns the fact. Link to it from other documents
   instead of copying the same status or file list into several places.
3. Search for outdated names, paths, defaults, counts, and broken links.
   Keep README examples aligned with the CLI and config/paths.yaml.
4. Inspect the files selected for Git. Keep licensed datasets, participant
   identifiers, API keys, local logs, generated outputs, and private progress
   reports out of the public repository.
5. Run the relevant fixture checks and try any changed CLI example. Record
   what was verified and whether live API or real-data checks were run.
6. Update docs/CURRENT_STATUS.md for current capabilities or limits. Keep
   dated development notes in the private project rather than presenting
   historical claims as current guidance.

This checklist applies to any contributor or coding tool.
