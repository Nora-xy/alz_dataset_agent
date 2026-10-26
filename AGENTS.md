# Agent instructions

These are the project rules for contributors and coding agents. Start with
README.md for setup and use. Source code and config/paths.yaml define current
behavior and paths. docs/CURRENT_STATUS.md tracks detailed capabilities and
limitations; docs/BACKEND_CONTRACT.md documents data operations; docs/TASK_ROUTING.md
helps locate relevant code; docs/MAINTENANCE_GUIDE.md covers documentation upkeep.
docs/DATA_DOWNLOADS.md lists the authorized local files used by this project.

The project answers cohort, follow-up, modality, and modeling questions for NACC
and ADNI, with ADSP-PHC as a harmonized feature source. Check relevant code and
fixture tests before editing. The rules below protect research data and computed
results.

---

## ID-SPACE & LINKAGE MODEL (critical — drives cohort computation)

Two disjoint ID spaces with NO crosswalk between them:
- **Space A** (key: NACCID): `NACC/investigator_nacc70.csv` + `ADSP-PHC-122024-investigator/*`
  ADSP-PHC assigns a NACCID to every participant (including ADNI-origin ones), so NACC raw
  and the ADSP-PHC main release join cleanly on NACCID. Confirm the NACCID column exists in
  the ADSP-PHC release tables before relying on it (they may also carry SUBJID).
- **Space B** (key: RID/PTID): the configured `ADNI/` download, including its `ADSP_PHC_*` tables
  files. These are keyed by ADNI IDs (RID/PTID/SUBJID), carry NO NACCID, and cannot be
  joined to Space A without a crosswalk that does NOT exist locally.

**RULES:**
- Cohort-size / modality-overlap / feasibility are computed WITHIN one id_space only.
- Any query that joins or sums across Space A and Space B must ABSTAIN with reason
  `"cross-cohort linkage unverified — crosswalk not built"`. Never add A and B counts.
- Do not treat ADSP_PHC_* files under ADNI/ as Space A; they are Space B's harmonized
  feature set (the ADNI external-validation bundle).

---

## DATA TIERS

- **Tier 1** (primary training): Space A, restricted to participants present in NACC raw
  (join FROM NACC raw, left/inner — keeps cohort NACC-origin, excludes ADNI-origin ADSP-PHC
  participants). Carries conversion + cognitive-decline labels.
- **Tier 2** (harmonized feature source): ADSP-PHC harmonized variables within Space A,
  attached to Tier 1 via NACCID.
- **Tier 3** (external validation candidate): Space B (ADNI). Tagged
  `external_validation_candidate`; never merged into Tier 1.

**LEAKAGE CAVEAT:** A small number of individuals may be enrolled in both NACC and ADNI.
Without an RID<->NACCID crosswalk this overlap cannot be detected, so Tier 3 external
validation may share subjects with Tier 1. Record this as a known limitation on any
external-validation cohort map entry.

---

## RELEASE NOTES

- ADSP-PHC is R3 only; R4 (plasma biomarkers + harmonized diagnosis) is NOT present.
  Any query requiring R4 variables must abstain with reason `"R4 not available locally"`.
- NACC `investigator_nacc70.csv` is wide-format UDS; participant ID is NACCID, visit
  identifier is NACCVNUM. Print df.columns and confirm exact names before hardcoding.
- ADNI tables use RID or PTID depending on table; ADSP-PHC release uses SUBJID and/or
  NACCID. Confirm per-file before use; do not assume.

---

## PROJECT INVARIANTS (apply to every task; do not violate without explicit instruction)

**1. COMPUTATION LIVES IN DETERMINISTIC TOOLS, NEVER IN THE LLM.**
The LLM only (a) parses intent, (b) selects/sequences tool calls, (c) narrates the
values a tool returned. No tool function may itself call an LLM. No number in any
user-facing answer may originate from model generation — it must trace to a tool result.

**2. ABSTAIN WHEN UNGROUNDED.**
If the catalog/aggregate store has no grounding for a request (modality combo absent,
release does not cover the variable, cells suppressed, mapping unverified), return an
explicit abstain result (`"not available in current data"`) — never a guess.

**3. EVERY ANSWER CARRIES EVIDENCE.**
Two evidence kinds: data evidence (aggregate table + query spec that produced a number)
and documentation evidence (source doc / paper-note citation). Tool results carry
structured evidence objects; the narration layer surfaces them.

**4. SUBJECT-LEVEL DATA NEVER LEAVES THE LOCAL BACKEND.**
Tools operate through a DataBackend interface. Only aggregate results cross any boundary.
Small-cell suppression (SUPPRESSION_THRESHOLD) is enforced at the aggregate layer AND
re-checked at the cloud/local transfer boundary.

**5. REAL DATA IS LOCAL; FIXTURES ARE FOR TESTS ONLY.**
All datasets are downloaded locally and may be read directly. The primary DataBackend
reads this local data. Keep a small synthetic fixture backend for tests/CI where the
real data is not mounted. Invariant 4 still holds: only aggregates ever leave the local
backend toward the agent, regardless of where the data lives.

**6. FILE PLACEMENT.**
Keep source under src/alz_agent/, tests under tests/, and configuration under config/.
Use config/paths.yaml as the current data-path map. Keep research data and generated
outputs out of Git.

**7. TESTS ARE PART OF DONE.**
No task is complete without tests that run on fixtures. Prefer pure functions.

**8. ID-SPACE PARTITION.**
Every catalog entry and every backend query carries an id_space (A=NACCID, B=RID/PTID).
Counts and joins happen within one id_space only. Any cross-space operation abstains
(`"cross-cohort linkage unverified"`). Counts from different id_spaces are never summed.
See the ID-SPACE & LINKAGE MODEL above.
