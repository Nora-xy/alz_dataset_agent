"""CLI for manually testing the Alzheimer's dataset agent.

Two operating modes:

  NL mode   — passes a natural-language question through intent parsing and
               run_agent(); supports heuristic (no key), anthropic, deepseek,
               hf-local.

  Direct mode — bypasses intent parsing entirely; calls one deterministic P3
                tool with structured arguments you supply on the command line.
                Always use this when the heuristic parser would misclassify
                the question (e.g. dataset-specific modality-overlap queries).

Two backend choices:

  fixture   — FixtureDataBackend (synthetic, seed-deterministic; what the test
               suite uses). Default. Eligible N ≈ 68 for MCI — never real data.

  local     — LocalDataBackend (reads actual CSV/xlsx files from config/paths.yaml).
               Required for real ADSP-PHC / NACC / ADNI counts.

Valid --modalities values:  cognition  mri  pet  fluid_biomarker  vascular  neuropath  demographic
Alias:  any_imaging  →  mri + pet

Valid --dataset values:
  nacc       id_space = A  (NACCID)
  adsp_phc   id_space = A  (NACCID, same Space A as NACC)
  adni       id_space = B  (RID/PTID)

Valid --baseline-diagnosis values:  normal  mci  dementia  impaired-not-mci

Example commands:
  # NL heuristic, fixture backend, with intent debug + parser warnings
  python scripts/ask_agent.py \\
    -q "Among MCI participants, can we predict 3-year conversion to AD?" \\
    --show-intent --show-meta

  # NL heuristic showing WHY the ADSP-PHC question is mis-parsed
  python scripts/ask_agent.py \\
    -q "How many participants in ADSP-PHC dataset have both cognition and any kind of imaging data?" \\
    --show-intent

  # Direct modality-overlap on REAL ADSP-PHC data (the right command for that question)
  python scripts/ask_agent.py \\
    --direct-tool modality-overlap \\
    --dataset adsp_phc \\
    --modalities cognition any_imaging \\
    --backend local --show-meta

  # Direct longitudinal-feasibility on real NACC data
  python scripts/ask_agent.py \\
    --direct-tool longitudinal-feasibility \\
    --dataset nacc --baseline-diagnosis mci --horizon-years 3 \\
    --backend local --show-meta

  # Anthropic LLM mode
  python scripts/ask_agent.py --mode anthropic \\
    -q "How many participants in ADSP-PHC dataset have both cognition and imaging?" \\
    --show-intent --show-meta

  # Semantic-gate planner — recommended for multi-group / multi-variant questions
  # (e.g. comparing an ADNI biomarker across CN/MCI/AD, or across several assay
  # variants). Implies --use-enriched-planner automatically; --show-meta prints
  # planner_path=semantic_enriched to confirm the semantic gate was reached.
  python scripts/ask_agent.py --mode anthropic --backend local \\
    -q "In ADNI, what are the mean CSF p-tau181 levels for CN, MCI, and AD at baseline?" \\
    --use-semantic-planner --show-meta

  # Local Hugging Face mode (hf-local) — runs model weights on local machine.
  # First run downloads weights from HF Hub; subsequent runs are fully offline.
  # Set HF_HOME (or --hf-cache-dir) to redirect the weight cache to a larger drive.
  #
  #   $env:HF_HOME = "D:\\hf_cache"          # PowerShell: redirect cache
  #
  python scripts/ask_agent.py \\
    --mode hf-local \\
    --model Qwen/Qwen2.5-7B-Instruct \\
    -q "Among MCI participants, can we predict 3-year conversion to AD?" \\
    --backend local \\
    --use-enriched-planner
  #
  # Switch models by changing --model only:
  #   --model meta-llama/Llama-3.2-3B-Instruct
  #   --model microsoft/Phi-3.5-mini-instruct
  #
  # On low-VRAM hardware, add --hf-local-load-in-4bit (requires bitsandbytes):
  #   pip install bitsandbytes
  #   python scripts/ask_agent.py --mode hf-local --model Qwen/Qwen2.5-7B-Instruct \\
  #     --hf-local-load-in-4bit -q "..." --backend local

  # Interactive NL loop with real backend
  python scripts/ask_agent.py --interactive --backend local --show-intent
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Make src/ importable when running the script directly.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

_SEP = "-" * 66
_DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
_DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic"

# Modality alias expansion table
_MODALITY_ALIASES: dict[str, list[str]] = {
    "any_imaging": ["mri", "pet"],
    "imaging":     ["mri", "pet"],
}
_VALID_MODALITIES = frozenset(
    {"cognition", "mri", "pet", "fluid_biomarker", "vascular", "neuropath", "demographic"}
)


# ---------------------------------------------------------------------------
# Backend construction
# ---------------------------------------------------------------------------


def _backend_fixture():
    """Return (catalog, backend, label, analysis_engine=None) using synthetic backend."""
    from alz_agent.eval.harness import default_backend, default_catalog

    catalog = default_catalog()
    backend = default_backend()
    label = (
        "FixtureDataBackend  [SYNTHETIC — no real data]\n"
        "  Built from seed-42 random participants; "
        "eligible N values are not real counts.\n"
        "  Use --backend local for actual NACC / ADSP-PHC / ADNI figures."
    )
    return catalog, backend, label, None  # no analysis engine for fixture backend


def _backend_local():
    """Return (catalog, backend, label, analysis_engine) using the real LocalDataBackend."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        print(
            "ERROR: PyYAML is required for --backend local.\n"
            "Install: pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = _ROOT / "config" / "paths.yaml"
    if not config_path.exists():
        print(
            f"ERROR: config/paths.yaml not found at {config_path}.\n"
            "Expected project layout: alz_dataset_agent/config/paths.yaml",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    from alz_agent.analysis.engine import LocalAnalysisEngine
    from alz_agent.backend.local_backend import LocalDataBackend
    from alz_agent.catalog.nacc_parser import parse_nacc
    from alz_agent.catalog.adni_parser import parse_adni, resolve_latest_snapshot
    from alz_agent.catalog.adsp_parser import parse_adsp_phc

    backend = LocalDataBackend(config, _ROOT)
    analysis_engine = LocalAnalysisEngine(backend)

    # Build the full catalog (NACC + ADNI + ADSP-PHC) via direct parser calls.
    # This ensures the enriched planner has access to all available catalog entries.
    # (We skip the slow leakage-rule tagging from build_catalog; that's optional enrichment.)
    try:
        # NACC
        nacc_cfg = config["nacc"]
        nacc_entries = parse_nacc(
            data_csv=_ROOT / nacc_cfg["data_csv"],
            modality_map_path=_ROOT / nacc_cfg["modality_map"],
            rdd_pdf=None,  # skip 72s PDF parse
        )

        # ADNI
        adni_cfg = config["adni"]
        adni_base_dir = _ROOT / adni_cfg["base_dir"]
        datadic_csv = resolve_latest_snapshot(adni_base_dir, adni_cfg["datadic_glob"])
        if datadic_csv is None:
            adni_entries = []
        else:
            adni_data_files: dict = {}
            for table, pattern in config.get("backend", {}).get("adni_data_files", {}).items():
                resolved = resolve_latest_snapshot(adni_base_dir, pattern)
                if resolved is not None:
                    adni_data_files[table] = resolved
            adni_entries = parse_adni(
                datadic_csv=datadic_csv,
                local_tables=adni_cfg.get("local_tables", {}),
                data_files=adni_data_files,
            )

        # ADSP-PHC
        adsp_cfg = config["adsp_phc"]
        adsp_entries = parse_adsp_phc(
            base_dir=_ROOT / adsp_cfg["base_dir"],
            modality_dirs={k: v for k, v in adsp_cfg["modality_dirs"].items()},
            release_version=adsp_cfg.get("release_version", "R3"),
        )

        catalog = nacc_entries + adni_entries + adsp_entries
        catalog_note = (
            f"Full catalog: {len(catalog)} entries "
            f"(NACC={len(nacc_entries)}, "
            f"ADNI={len(adni_entries)}, "
            f"ADSP_PHC={len(adsp_entries)})"
        )
    except Exception as exc:
        # Fallback: load NACC only if full load fails
        from alz_agent.eval.harness import default_catalog

        nacc_cfg = config["nacc"]
        try:
            nacc_entries = parse_nacc(
                data_csv=_ROOT / nacc_cfg["data_csv"],
                modality_map_path=_ROOT / nacc_cfg["modality_map"],
                rdd_pdf=None,
            )
            # Merge with fixture for leakage tags
            fixture_ids = {e.variable_id for e in default_catalog()}
            extra = [e for e in nacc_entries if e.variable_id not in fixture_ids]
            catalog = default_catalog() + extra
            catalog_note = (
                f"NACC fallback ({len(catalog)} entries; "
                f"full catalog load failed)"
            )
        except Exception as exc2:
            catalog = default_catalog()
            catalog_note = f"fixture 6-var catalog (load failed: {exc2})"

    label = (
        f"LocalDataBackend  [REAL DATA from config/paths.yaml]\n"
        f"  project_root = {_ROOT}\n"
        f"  Catalog: {catalog_note}\n"
        f"  For RDD-enriched NACC concepts: python -m alz_agent.catalog.builder"
    )
    return catalog, backend, label, analysis_engine


def _load_backend(name: str):
    """Return (catalog, backend, label, analysis_engine)."""
    if name == "fixture":
        return _backend_fixture()
    if name == "local":
        return _backend_local()
    raise ValueError(f"Unknown backend: {name!r}")


# ---------------------------------------------------------------------------
# LLM client construction
# ---------------------------------------------------------------------------


def _build_client(
    mode: str,
    model: str | None,
    base_url: str | None,
    *,
    hf_cache_dir: str | None = None,
    hf_device_map: str = "auto",
    hf_dtype: str = "auto",
    hf_load_in_4bit: bool = False,
    hf_max_new_tokens: int = 512,
):
    """Return a provider client, or None for heuristic mode.

    For anthropic and deepseek, returns an AnthropicPlannerClient.
    For hf-local, returns a LocalHFPlannerClient (local transformers execution —
    no data leaves the machine during inference).
    """
    if mode == "heuristic":
        return None

    from alz_agent.agent.providers import build_planner_client  # noqa: PLC0415

    if mode == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            print(
                "ERROR: ANTHROPIC_API_KEY is not set.\n"
                "Set it in PowerShell:\n"
                '    $env:ANTHROPIC_API_KEY = "sk-ant-..."',
                file=sys.stderr,
            )
            sys.exit(1)
        return build_planner_client("anthropic", model=model, api_key=key, base_url=base_url)

    if mode == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            print(
                "ERROR: DEEPSEEK_API_KEY is not set.\n"
                "Set it in PowerShell:\n"
                '    $env:DEEPSEEK_API_KEY = "sk-..."',
                file=sys.stderr,
            )
            sys.exit(1)
        resolved_url = base_url or _DEEPSEEK_DEFAULT_BASE_URL
        return build_planner_client("deepseek", model=model, api_key=key, base_url=resolved_url)

    if mode == "hf-local":
        if not model:
            print(
                "ERROR: --model MODEL_NAME is required for --mode hf-local.\n"
                "Example: --model Qwen/Qwen2.5-7B-Instruct\n"
                "         --model meta-llama/Llama-3.2-3B-Instruct",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"hf-local: loading {model!r} via transformers (local execution).\n"
            "  First run downloads weights from HF Hub; subsequent runs use local cache.\n"
            "  Set --hf-cache-dir or $env:HF_HOME to redirect the weight cache.\n"
            "  No participant data is sent to any remote service during inference.\n",
            file=sys.stderr,
        )
        return build_planner_client(
            "hf-local",
            model=model,
            hf_cache_dir=hf_cache_dir,
            hf_device_map=hf_device_map,
            hf_dtype=hf_dtype,
            hf_load_in_4bit=hf_load_in_4bit,
            hf_max_new_tokens=hf_max_new_tokens,
        )

    raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Modality expansion
# ---------------------------------------------------------------------------


def _expand_modalities(raw: list[str]) -> tuple[list[str], list[str]]:
    """Expand aliases; deduplicate; validate.  Returns (expanded, note_lines)."""
    expanded: list[str] = []
    notes: list[str] = []

    for token in raw:
        if token in _MODALITY_ALIASES:
            targets = _MODALITY_ALIASES[token]
            notes.append(f"alias '{token}' → {', '.join(targets)}")
            for t in targets:
                if t not in expanded:
                    expanded.append(t)
        elif token not in expanded:
            expanded.append(token)

    invalid = [m for m in expanded if m not in _VALID_MODALITIES]
    if invalid:
        print(
            f"ERROR: Unknown modality value(s): {invalid}\n"
            f"Valid: {sorted(_VALID_MODALITIES)}\n"
            f"Alias: any_imaging → mri + pet",
            file=sys.stderr,
        )
        sys.exit(1)

    return expanded, notes


# ---------------------------------------------------------------------------
# Dataset → IdSpace
# ---------------------------------------------------------------------------


def _dataset_to_id_space(dataset: str):
    from alz_agent.schemas import IdSpace

    mapping = {"nacc": IdSpace.A, "adsp_phc": IdSpace.A, "adni": IdSpace.B}
    if dataset not in mapping:
        print(
            f"ERROR: Unknown --dataset '{dataset}'.\n"
            "Valid: nacc (Space A), adsp_phc (Space A), adni (Space B)",
            file=sys.stderr,
        )
        sys.exit(1)
    return mapping[dataset]


# ---------------------------------------------------------------------------
# Direct tool mode
# ---------------------------------------------------------------------------


def _print_direct_header(tool: str, dataset: str, id_space, extra_lines: list[str]) -> None:
    from alz_agent.schemas import IdSpace

    print(_SEP)
    print("DIRECT TOOL:")
    print(f"  tool      : {tool}")
    print(f"  dataset   : {dataset}")
    space_desc = "NACCID / Space A" if id_space == IdSpace.A else "RID/PTID / Space B"
    print(f"  id_space  : {id_space.value}  ({space_desc})")
    for line in extra_lines:
        print(f"  {line}")


def _print_backend_block(label: str) -> None:
    print()
    print("BACKEND:")
    for line in label.splitlines():
        print(f"  {line}")


def _print_metadata(grounded: bool, abstain_reason: str | None, show_meta: bool,
                    evidence_refs: list[str]) -> None:
    print()
    print("METADATA:")
    print(f"  grounded                  : {grounded}")
    print(f"  hallucination_check_passed: N/A  (direct mode — no LLM narration)")
    if not grounded and abstain_reason:
        print(f"  abstain_reason            : {abstain_reason}")
    if show_meta and grounded:
        for ref in evidence_refs:
            print(f"  evidence                  : {ref}")
    print(_SEP)


def _run_modality_overlap(args, catalog, backend, backend_label: str) -> None:
    if not args.modalities:
        print("ERROR: --modalities is required for --direct-tool modality-overlap", file=sys.stderr)
        sys.exit(1)

    from alz_agent.schemas import FilterSpec

    dataset = args.dataset or "nacc"
    id_space = _dataset_to_id_space(dataset)
    expanded, alias_notes = _expand_modalities(args.modalities)

    extra = []
    for note in alias_notes:
        extra.append(f"modality alias : {note}")
    extra.append(f"modalities     : {', '.join(expanded)}")
    if args.baseline_diagnosis:
        extra.append(f"dx filter      : {', '.join(args.baseline_diagnosis)}")

    _print_direct_header("modality-overlap", dataset, id_space, extra)

    # Dataset-specific fixture note
    if args.backend == "fixture":
        print()
        print(
            "  NOTE: --backend fixture returns SYNTHETIC counts (seed-42 random data).\n"
            "        Use --backend local for real ADSP-PHC / NACC / ADNI counts."
        )

    _print_backend_block(backend_label)

    filter_spec = FilterSpec(
        baseline_diagnosis=args.baseline_diagnosis or None,
        id_space=id_space,
    )

    # Call backend.modality_intersection directly — the catalog pre-check in
    # assess_modality_overlap is for the NL pipeline; in direct mode the user
    # explicitly specifies modalities and the backend handles unknowns itself.
    result = backend.modality_intersection(expanded, filter_spec)

    print()
    if result.grounded:
        val = result.value or {}
        print("ANSWER (modality intersection counts):")
        # Print single-modality counts first, then combinations
        singles = {k: v for k, v in val.items() if "," not in k}
        combos  = {k: v for k, v in val.items() if "," in k}
        for k in sorted(singles):
            _print_count_row(k, singles[k])
        if combos:
            print()
        for k in sorted(combos):
            _print_count_row(k, combos[k])
    else:
        print("ANSWER:")
        print(f"  NOT ANSWERABLE: {result.abstain_reason}")

    ev_refs = [ev.aggregate_ref for ev in result.evidence]
    _print_metadata(result.grounded, result.abstain_reason, args.show_meta, ev_refs)


def _print_count_row(key: str, val: int | None) -> None:
    display = "suppressed (below reporting threshold)" if val is None else str(val)
    print(f"  {key:<40}: {display}")


def _run_longitudinal_feasibility(args, catalog, backend, backend_label: str) -> None:
    horizon = args.horizon_years
    if horizon is None:
        print(
            "ERROR: --horizon-years is required for --direct-tool longitudinal-feasibility",
            file=sys.stderr,
        )
        sys.exit(1)

    from alz_agent.schemas import FilterSpec
    from alz_agent.tools.deterministic import assess_longitudinal_feasibility

    dataset = args.dataset or "nacc"
    id_space = _dataset_to_id_space(dataset)
    extra = [f"horizon       : {horizon} year(s)"]
    if args.baseline_diagnosis:
        extra.append(f"dx filter      : {', '.join(args.baseline_diagnosis)}")

    _print_direct_header("longitudinal-feasibility", dataset, id_space, extra)
    _print_backend_block(backend_label)

    filter_spec = FilterSpec(
        baseline_diagnosis=args.baseline_diagnosis or None,
        id_space=id_space,
    )
    result = assess_longitudinal_feasibility(horizon, filter_spec, backend)

    print()
    if result.grounded:
        val = result.value or {}
        print("ANSWER (follow-up distribution):")
        for k in ("sufficient", "insufficient", "censored"):
            if k in val:
                _print_count_row(k, val[k])
    else:
        print("ANSWER:")
        print(f"  NOT ANSWERABLE: {result.abstain_reason}")

    ev_refs = [ev.aggregate_ref for ev in result.evidence]
    _print_metadata(result.grounded, result.abstain_reason, args.show_meta, ev_refs)


def _run_cohort_size(args, catalog, backend, backend_label: str) -> None:
    from alz_agent.schemas import FilterSpec
    from alz_agent.tools.deterministic import compute_cohort_size

    dataset = args.dataset or "nacc"
    id_space = _dataset_to_id_space(dataset)
    extra = []
    if args.baseline_diagnosis:
        extra.append(f"dx filter  : {', '.join(args.baseline_diagnosis)}")

    _print_direct_header("cohort-size", dataset, id_space, extra)
    _print_backend_block(backend_label)

    filter_spec = FilterSpec(
        baseline_diagnosis=args.baseline_diagnosis or None,
        id_space=id_space,
    )
    result = compute_cohort_size(filter_spec, backend)

    print()
    if result.grounded:
        print("ANSWER:")
        print(f"  Eligible participants : {result.value}")
    else:
        print("ANSWER:")
        print(f"  NOT ANSWERABLE: {result.abstain_reason}")

    ev_refs = [ev.aggregate_ref for ev in result.evidence]
    _print_metadata(result.grounded, result.abstain_reason, args.show_meta, ev_refs)


def _run_modeling_design(args, catalog, backend, backend_label: str) -> None:
    horizon = args.horizon_years
    if horizon is None:
        print(
            "ERROR: --horizon-years is required for --direct-tool modeling-design",
            file=sys.stderr,
        )
        sys.exit(1)

    from alz_agent.schemas import FilterSpec
    from alz_agent.tools.deterministic import suggest_modeling_design

    dataset = args.dataset or "nacc"
    id_space = _dataset_to_id_space(dataset)
    extra = [f"horizon       : {horizon} year(s)"]
    if args.baseline_diagnosis:
        extra.append(f"dx filter      : {', '.join(args.baseline_diagnosis)}")
    if args.label:
        extra.append(f"label override : {args.label}")

    _print_direct_header("modeling-design", dataset, id_space, extra)
    _print_backend_block(backend_label)

    filter_spec = FilterSpec(
        baseline_diagnosis=args.baseline_diagnosis or None,
        id_space=id_space,
    )
    result = suggest_modeling_design(filter_spec, horizon, catalog, backend)

    print()
    if result.grounded:
        val = result.value
        print("ANSWER:")
        n_display = str(val.eligible_n) if val.eligible_n is not None else "suppressed"
        print(f"  eligible_n              : {n_display}")
        print(f"  horizon_years           : {val.horizon_years}")
        print(f"  outcome_label_def       : {val.outcome_label_def}")
        print(f"  index_visit_def         : {val.index_visit_def}")
        if val.feasibility:
            print("  follow-up distribution  :")
            for k in ("sufficient", "insufficient", "censored"):
                if k in val.feasibility:
                    v = val.feasibility[k]
                    display = str(v) if v is not None else "suppressed"
                    print(f"    {k:<18}: {display}")
        print(f"  feature_whitelist       : {len(val.feature_whitelist)} variable(s)")
        print(f"  leakage_blacklist       : {len(val.leakage_blacklist)} variable(s)")
    else:
        print("ANSWER:")
        print(f"  NOT ANSWERABLE: {result.abstain_reason}")

    ev_refs = [ev.aggregate_ref for ev in result.evidence]
    _print_metadata(result.grounded, result.abstain_reason, args.show_meta, ev_refs)


def _run_direct_tool(args, catalog, backend, backend_label: str) -> None:
    tool = args.direct_tool
    if tool == "modality-overlap":
        _run_modality_overlap(args, catalog, backend, backend_label)
    elif tool == "longitudinal-feasibility":
        _run_longitudinal_feasibility(args, catalog, backend, backend_label)
    elif tool == "cohort-size":
        _run_cohort_size(args, catalog, backend, backend_label)
    elif tool == "modeling-design":
        _run_modeling_design(args, catalog, backend, backend_label)
    else:
        print(f"ERROR: Unknown --direct-tool '{tool}'", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# NL mode — heuristic parser confidence check
# ---------------------------------------------------------------------------


def _heuristic_warnings(question: str, parsed_intent) -> list[str]:
    """Return list of warning strings when the parsed intent looks unreliable."""
    warnings: list[str] = []
    q = question.lower()

    # Population defaulted to mci but the word "mci" is absent
    if parsed_intent.target_population == ["mci"] and "mci" not in q:
        warnings.append(
            "target_population defaulted to ['mci'] but 'mci' does not appear in the "
            "question.  The heuristic parser may have misclassified the population.  "
            "Consider --mode anthropic or --direct-tool."
        )

    if "adsp" in q:
        warnings.append(
            "Question mentions ADSP-PHC, but heuristic mode selects only its ID "
            "space, not that release alone. Use a direct tool or API planner for "
            "a release-specific question."
        )

    return warnings


# ---------------------------------------------------------------------------
# Trace display helpers
# ---------------------------------------------------------------------------


def _print_trace(trace) -> None:
    """Print the full AnalysisTrace in a human-readable format."""
    print()
    print("ANALYSIS TRACE:")
    print(f"  original_question   : {trace.original_question}")
    print(f"  parsing_mode        : {trace.parsing_mode}")
    print(f"  question_type       : {trace.question_type}")
    print(f"  analysis_type       : {trace.analysis_type}")
    if trace.research_intent_summary:
        print(f"  research_intent     : {trace.research_intent_summary}")
    if trace.statistic:
        print(f"  statistic           : {trace.statistic}")
    if trace.visit_anchor:
        print(f"  visit_anchor        : {trace.visit_anchor}")
    if trace.unit_of_analysis:
        print(f"  unit_of_analysis    : {trace.unit_of_analysis}")

    if trace.plan_steps:
        print()
        print("  ANALYSIS PLAN STEPS:")
        for i, step in enumerate(trace.plan_steps, 1):
            print(f"    {i}. {step}")

    if trace.variable_resolutions:
        print()
        print("  VARIABLE RESOLUTIONS:")
        for vr in trace.variable_resolutions:
            code_str = vr.selected_raw_code or "[unresolved]"
            filt_str = f"  filter={vr.filter_expression}" if vr.filter_expression else ""
            status_str = f"  [{vr.resolution_status}"
            if vr.validation_status:
                status_str += f"/{vr.validation_status}"
            status_str += "]"
            print(f"    {vr.role:<18}: {vr.researcher_label!r} → {code_str}{filt_str}{status_str}")

    print()
    print("  FILTER APPLIED:")
    for k, v in trace.filter_trace.items():
        if v is not None:
            print(f"    {k:<22}: {v}")

    if trace.leakage_findings:
        print()
        print("  LEAKAGE FINDINGS:")
        for f in trace.leakage_findings:
            print(f"    {f.raw_code:<16} [{f.role}] — {f.leakage_tag} — {f.severity}")

    if trace.tool_calls:
        print()
        print("  TOOL CALLS:")
        for i, tc in enumerate(trace.tool_calls, 1):
            print(f"    [{i}] {tc.tool_name}  grounded={tc.grounded}")
            if tc.input_summary:
                print(f"        input : {tc.input_summary}")
            if tc.output_summary:
                print(f"        output: {tc.output_summary}")
            if tc.abstain_reason:
                print(f"        abstain: {tc.abstain_reason}")

    if trace.caveats:
        print()
        print("  CAVEATS:")
        for c in trace.caveats:
            print(f"    - {c}")

    if trace.failure_mode or trace.validation_errors:
        print()
        print("  PARSE DEBUG:")
        print(f"    failure_mode      : {trace.failure_mode}")
        for err in trace.validation_errors:
            print(f"    validation_error  : {err}")


def _print_plan_summary(trace) -> None:
    """Print a compact analysis-plan summary."""
    print()
    print("ANALYSIS PLAN (compact):")
    print(f"  analysis_type  : {trace.analysis_type}")
    print(f"  statistic      : {trace.statistic}")
    print(f"  visit_anchor   : {trace.visit_anchor}")
    if trace.variable_resolutions:
        print("  variables      :")
        for vr in trace.variable_resolutions:
            filt = f"  filter={vr.filter_expression}" if vr.filter_expression else ""
            print(f"    {vr.role:<18}: {vr.selected_raw_code or '[unresolved]'}{filt}")
    print(f"  filter         : {trace.filter_trace}")


def _print_debug_info(trace, parsed_intent=None) -> None:
    """Print parsing debug info including failure_mode and validation_errors."""
    print()
    print("PARSE DEBUG:")
    print(f"  parsing_mode     : {trace.parsing_mode}")
    print(f"  failure_mode     : {trace.failure_mode}")
    if trace.validation_errors:
        print("  validation_errors:")
        for err in trace.validation_errors:
            print(f"    {err}")
    else:
        print("  validation_errors: []")
    if parsed_intent is not None and parsed_intent.parse_debug:
        dbg = parsed_intent.parse_debug
        if dbg.raw_llm_plan_json:
            import json as _json
            print("  raw_llm_plan_json :")
            raw_str = _json.dumps(dbg.raw_llm_plan_json, indent=4)
            for line in raw_str.splitlines()[:40]:  # cap at 40 lines
                print(f"    {line}")


# ---------------------------------------------------------------------------
# NL mode — single question
# ---------------------------------------------------------------------------


def _print_hf_local_raw(
    raw_attempts: list[str],
    parsed: Any | None,
    abstain_exc: Exception | None,
) -> None:
    """Print raw hf-local compact JSON text and the adapted StructuredIntent
    (or the abstain reason if none was produced). Used by --show-raw-intent /
    --show-intent to distinguish a Qwen generation failure from an adapter
    normalization failure — see P0 requirement in the hf-local planning fix."""
    print()
    print("HF-LOCAL RAW OUTPUT:")
    if not raw_attempts:
        print("  (no model call recorded)")
    for i, raw in enumerate(raw_attempts, start=1):
        label = "attempt 1" if i == 1 else "attempt 2 / repair"
        print(f"  {label}:")
        for line in (raw.splitlines() or [""]):
            print(f"    {line}")
    print()
    print("ADAPTED STRUCTURED INTENT:")
    if abstain_exc is not None:
        print(f"  ABSTAINED — no StructuredIntent produced. Reason: {abstain_exc}")
    elif parsed is not None:
        for k, v in parsed.model_dump().items():
            print(f"  {k}: {v}")
    else:
        print("  (not parsed)")


def _ask_once_nl(
    question: str,
    catalog,
    backend,
    backend_label: str,
    client,
    mode: str,
    model: str | None,
    show_intent: bool,
    show_meta: bool,
    debug: bool,
    *,
    analysis_engine=None,
    show_trace: bool = False,
    show_plan: bool = False,
    show_debug: bool = False,
    use_enriched_planner: bool = False,
    use_semantic_planner: bool = False,
    show_raw_intent: bool = False,
) -> None:
    from alz_agent.agent.intent import parse_intent
    from alz_agent.agent.orchestrator import run_agent
    from alz_agent.agent.hf_local_intent import HFLocalPlannerAbstention

    try:
        print(_SEP)
        print("QUESTION:")
        print(f"  {question}")
        print()
        print("MODE:")
        print(f"  {mode}" + (f"  [{model}]" if model else ""))
        print()
        print("BACKEND:")
        for line in backend_label.splitlines():
            print(f"  {line}")

        parsed = None
        raw_intent_attempts: list[str] = []
        want_raw_intent = show_raw_intent and mode == "hf-local"
        if show_intent or show_debug or want_raw_intent:
            # Pre-parse so we can display intent and warnings
            debug_sink = raw_intent_attempts if mode == "hf-local" else None
            try:
                parsed = parse_intent(question, client, model=model, debug_sink=debug_sink)
            except HFLocalPlannerAbstention as exc:
                if want_raw_intent or show_intent:
                    _print_hf_local_raw(raw_intent_attempts, None, exc)
                raise

        if want_raw_intent:
            _print_hf_local_raw(raw_intent_attempts, parsed, None)

        if show_intent and parsed is not None:
            print()
            print("PARSED INTENT:")
            print(f"  question_type     : {parsed.question_type.value}")
            print(f"  target_population : {parsed.target_population}")
            print(f"  target_etiology   : {parsed.target_etiology}")
            print(f"  candidate_label   : {parsed.candidate_label}")
            print(f"  horizon_years     : {parsed.horizon_years}")
            print(f"  modalities        : {parsed.modalities}")
            print(f"  id_space          : {parsed.id_space.value}")
            print(f"  pre_index_only    : {parsed.pre_index_only}")
            print(f"  stat_variable     : {parsed.stat_variable}")
            print(f"  stat_type         : {parsed.stat_type}")
            print(f"  event_anchor      : {parsed.event_anchor}")
            print(f"  analysis_type     : {parsed.analysis_type}")
            print(f"  group_variable    : {parsed.group_variable}")
            print(f"  outcome_variable  : {parsed.outcome_variable}")

            if mode == "heuristic":
                for w in _heuristic_warnings(question, parsed):
                    print()
                    print(f"  [PARSER WARNING]")
                    for wline in w.splitlines():
                        print(f"    {wline}")

        resp = run_agent(
            question, catalog, backend,
            # When use_enriched_planner=True, always pass intent=None so that
            # _run_enriched_path() calls plan_enriched_analysis() from the raw
            # question. The pre-parsed StructuredIntent (if any) is displayed
            # above for reference only and must not short-circuit the enriched path.
            # use_semantic_planner implies use_enriched_planner (enforced in run_agent()
            # itself), so intent=None must trigger on either flag being set here too.
            intent=None if (use_enriched_planner or use_semantic_planner) else parsed,
            client=client,
            model=model,
            analysis_engine=analysis_engine,
            use_enriched_planner=use_enriched_planner,
            use_semantic_planner=use_semantic_planner,
        )

        print()
        print("ANSWER:")
        for line in resp.answer.splitlines():
            print(f"  {line}")

        print()
        print("METADATA:")
        print(f"  grounded                  : {resp.grounded}")
        print(f"  hallucination_check_passed: {resp.hallucination_check_passed}")
        print(f"  use_enriched_planner      : {use_enriched_planner}")
        print(f"  use_semantic_planner      : {use_semantic_planner}")
        print(f"  planner_path              : {resp.planner_path}")
        print(f"  enriched_plan_used        : {resp.enriched_plan_used}")
        if resp.enriched_plan_error:
            print(f"  enriched_plan_error       : {resp.enriched_plan_error}")
        if resp.enriched_plan_operations:
            print(f"  enriched_plan_operations  : {resp.enriched_plan_operations}")
        if resp.requested_horizons:
            print(f"  requested_horizons        : {resp.requested_horizons}")
        if resp.validation_hard_errors:
            print(f"  validation_hard_errors    : {resp.validation_hard_errors}")
        if resp.validation_soft_warnings:
            print(f"  validation_soft_warnings  : {resp.validation_soft_warnings}")
        if resp.execution_operations_run:
            print(f"  execution_operations_run  : {resp.execution_operations_run}")
        print(f"  synthesis_report_present  : {resp.synthesis_report_present}")
        print(f"  question_answered         : {resp.question_answered}")
        print(f"  requested_measure_resolved: {resp.requested_measure_resolved}")
        print(f"  semantic_grounding_passed : {resp.semantic_grounding_passed}")
        if show_meta:
            if resp.abstain_reason:
                print(f"  abstain_reason            : {resp.abstain_reason}")
            print(f"  tool_results_count        : {len(resp.tool_results)}")
            print(f"  doc_evidence_count        : {len(resp.doc_evidence)}")

        if show_trace and resp.trace is not None:
            _print_trace(resp.trace)
        elif show_plan and resp.trace is not None:
            _print_plan_summary(resp.trace)

        if show_debug and resp.trace is not None:
            _print_debug_info(resp.trace, parsed)

        print(_SEP)

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        if debug:
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ask_agent.py",
        description="Alzheimer's dataset agent — natural-language and direct-tool CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "See module docstring at the top of this file for full usage examples,\n"
            "backend descriptions, and valid enum values."
        ),
    )

    # ── NL query ────────────────────────────────────────────────────────
    nl = parser.add_argument_group("Natural-language (NL) mode")
    nl.add_argument("--question", "-q", metavar="QUESTION",
                    help="Natural-language research question.")
    nl.add_argument("--interactive", action="store_true",
                    help="Open an interactive REPL loop (NL mode).")
    nl.add_argument(
        "--use-enriched-planner", action="store_true", dest="use_enriched_planner",
        help=(
            "(NL mode) Use the 4-layer enriched planner instead of the classic StructuredIntent "
            "path. Requires --mode anthropic or deepseek (needs an LLM client). "
            "Calls plan_enriched_analysis() from the raw question; ignores pre-parsed intent. "
            "No silent fallback — if enriched planning fails, returns an explicit error "
            "with enriched_plan_error set."
        ),
    )
    nl.add_argument(
        "--use-semantic-planner", action="store_true", dest="use_semantic_planner",
        help=(
            "(NL mode) Use the semantic gate ahead of the enriched planner: LLM produces a "
            "SemanticAnalysisPlan (explicit population_unit/cohort_scope/etiology_definition/"
            "groups/variable_selections/specimen/analyte slots), PlanValidator.validate_semantic() "
            "checks it (catalog existence, specimen/analyte consistency, diagnosis-domain "
            "rules, etc.), one repair pass on failure, then compiles deterministically to an "
            "EnrichedAnalysisPlan. Requires --mode anthropic or deepseek (needs an LLM client). "
            "ALWAYS implies --use-enriched-planner (enforced in run_agent(), not just here) — "
            "you never need to pass both, and there is no silent fallback to the classic "
            "planner when this is set. Recommended for multi-group/multi-variant biomarker "
            "questions (e.g. comparing an ADNI biomarker across CN/MCI/AD)."
        ),
    )

    # ── Direct tool ──────────────────────────────────────────────────────
    dt = parser.add_argument_group(
        "Direct tool mode  (bypasses intent parsing; calls a single P3 tool)"
    )
    dt.add_argument(
        "--direct-tool",
        dest="direct_tool",
        choices=["modality-overlap", "longitudinal-feasibility", "cohort-size", "modeling-design"],
        metavar="TOOL",
        help=(
            "Run a deterministic P3 tool directly.  "
            "Choices: modality-overlap | longitudinal-feasibility | cohort-size | modeling-design"
        ),
    )
    dt.add_argument(
        "--dataset",
        choices=["nacc", "adsp_phc", "adni"],
        default=None,
        help="Dataset / id_space: nacc=Space A, adsp_phc=Space A, adni=Space B.  Default: nacc.",
    )
    dt.add_argument(
        "--modalities", nargs="+", metavar="MOD",
        help=(
            "Modalities for modality-overlap.  "
            "Valid: cognition mri pet fluid_biomarker vascular neuropath demographic.  "
            "Alias: any_imaging → mri + pet."
        ),
    )
    dt.add_argument(
        "--baseline-diagnosis", nargs="+", metavar="DX", dest="baseline_diagnosis",
        help="Baseline cognitive-status filter: normal | mci | dementia | impaired-not-mci.",
    )
    dt.add_argument(
        "--horizon-years", type=int, default=None, metavar="N", dest="horizon_years",
        help="Prediction horizon in integer years (required for longitudinal-feasibility and modeling-design).",
    )
    dt.add_argument(
        "--label", default=None, metavar="TEXT",
        help="Optional outcome-label text for modeling-design.",
    )

    # ── LLM mode ─────────────────────────────────────────────────────────
    lm = parser.add_argument_group("LLM mode  (NL mode only)")
    lm.add_argument(
        "--mode", "--llm-provider", dest="mode",
        choices=["heuristic", "anthropic", "deepseek", "hf-local"],
        default="heuristic",
        help=(
            "heuristic — no API key needed, deterministic (default).  "
            "anthropic — uses ANTHROPIC_API_KEY; model default: claude-haiku-4-5-20251001.  "
            f"deepseek  — uses DEEPSEEK_API_KEY; model default: {_DEEPSEEK_DEFAULT_MODEL}.  "
            "hf-local  — runs open-source Hugging Face models locally via transformers.  "
            "  No API key needed. First run downloads weights from HF Hub; subsequent "
            "  runs load from local cache and can run fully offline. Requires --model.  "
            "  Install deps: pip install transformers torch accelerate"
        ),
    )
    lm.add_argument(
        "--model", default=None, metavar="MODEL_NAME",
        help=(
            "LLM model name.  Passed to both the intent parser and the narration step.  "
            "Required for --mode hf-local "
            "(e.g. Qwen/Qwen2.5-7B-Instruct or meta-llama/Llama-3.2-3B-Instruct)."
        ),
    )
    lm.add_argument(
        "--base-url", dest="base_url", default=None, metavar="URL",
        help=(
            f"Override LLM API base URL (anthropic / deepseek only).  "
            f"DeepSeek default: {_DEEPSEEK_DEFAULT_BASE_URL}"
        ),
    )

    # ── Local HF options ─────────────────────────────────────────────────
    hf = parser.add_argument_group(
        "Local HF options  (--mode hf-local only)",
        description=(
            "hf-local runs open-source Hugging Face models locally using transformers.  "
            "The first run may download model weights from the HF Hub, but later inference "
            "loads from the local cache and can run offline.  All computation and data "
            "remain on local infrastructure."
        ),
    )
    hf.add_argument(
        "--hf-cache-dir", dest="hf_cache_dir", default=None, metavar="PATH",
        help=(
            "Directory for caching model weights.  Equivalent to setting $env:HF_HOME.  "
            "Default: C:\\Users\\<user>\\.cache\\huggingface\\hub  "
            "Example: --hf-cache-dir D:\\hf_cache"
        ),
    )
    hf.add_argument(
        "--hf-local-device-map", dest="hf_local_device_map", default="auto", metavar="MAP",
        help=(
            "Device map passed to from_pretrained.  "
            "'auto' distributes layers across GPU VRAM then CPU RAM automatically.  "
            "Default: auto"
        ),
    )
    hf.add_argument(
        "--hf-local-dtype", dest="hf_local_dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto", metavar="DTYPE",
        help="Torch dtype for model weights.  Default: auto (model's native dtype).",
    )
    hf.add_argument(
        "--hf-local-load-in-4bit", dest="hf_local_load_in_4bit",
        action="store_true",
        help=(
            "Enable 4-bit quantization via bitsandbytes.  "
            "Reduces VRAM ~4× (7B model fits in ~4 GB instead of ~14 GB).  "
            "Requires: pip install bitsandbytes"
        ),
    )
    hf.add_argument(
        "--hf-local-max-new-tokens", dest="hf_local_max_new_tokens",
        type=int, default=512, metavar="N",
        help="Maximum new tokens generated per LLM call in hf-local mode.  Default: 512.",
    )

    # ── Backend ──────────────────────────────────────────────────────────
    be = parser.add_argument_group("Backend")
    be.add_argument(
        "--backend",
        choices=["fixture", "local"],
        default="fixture",
        help=(
            "fixture — synthetic FixtureDataBackend (default; what tests use; no real data).  "
            "local   — real LocalDataBackend reading actual files from config/paths.yaml."
        ),
    )

    # ── Display ──────────────────────────────────────────────────────────
    dv = parser.add_argument_group("Display")
    dv.add_argument(
        "--show-intent", action="store_true",
        help="(NL mode) Print parsed intent before executing.  Enables heuristic parser warnings.",
    )
    dv.add_argument(
        "--show-meta", action="store_true",
        help="Print additional metadata: abstain_reason, tool/evidence counts, evidence refs.",
    )
    dv.add_argument(
        "--show-trace", action="store_true",
        help="Print the full structured audit trace after the answer.",
    )
    dv.add_argument(
        "--show-plan", action="store_true",
        help="Print a compact analysis-plan summary (subset of --show-trace).",
    )
    dv.add_argument(
        "--show-debug", action="store_true",
        help="Print parsing debug info: failure_mode, validation_errors, parse_debug JSON.",
    )
    dv.add_argument(
        "--show-raw-intent", action="store_true", dest="show_raw_intent",
        help=(
            "(hf-local mode only) Print the raw hf-local compact JSON text from "
            "every model call (attempt + repair) and the adapted StructuredIntent "
            "(or the abstain reason), so a Qwen generation failure can be told "
            "apart from an adapter/semantic-guard normalization failure. "
            "No effect in heuristic/anthropic/deepseek mode."
        ),
    )
    dv.add_argument(
        "--debug", action="store_true",
        help="Print full traceback on errors.",
    )

    return parser


# ---------------------------------------------------------------------------
# Planner-flag resolution (pure — no I/O, directly unit-testable)
# ---------------------------------------------------------------------------


def _resolve_planner_flags(
    use_enriched_planner: bool, use_semantic_planner: bool,
) -> tuple[bool, bool, str | None]:
    """Resolve the CLI's two planner flags into (enriched, semantic, note).

    --use-semantic-planner ALWAYS implies --use-enriched-planner — a researcher
    passing only --use-semantic-planner must see use_enriched_planner=True
    reflected everywhere, never a silent fallback to the classic planner. This
    is the CLI-layer mirror of the same rule enforced inside run_agent() itself
    (agent/orchestrator.py); duplicated intentionally as defense in depth, not
    relied upon as the only enforcement point.

    Returns (resolved_use_enriched_planner, resolved_use_semantic_planner, note)
    where note is a user-facing string to print to stderr when the implication
    fired (None otherwise).
    """
    if use_semantic_planner and not use_enriched_planner:
        return True, True, (
            "NOTE: --use-semantic-planner implies --use-enriched-planner; "
            "enabling it automatically."
        )
    return use_enriched_planner, use_semantic_planner, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    is_direct = args.direct_tool is not None
    is_nl = bool(args.question) or args.interactive

    if is_direct and is_nl:
        parser.error("Cannot combine --direct-tool with --question / --interactive.")
    if not is_direct and not is_nl:
        parser.error("Provide --question/-q, --interactive, or --direct-tool TOOL.")
    if is_nl and args.dataset is not None:
        parser.error("--dataset applies only to direct tools; name NACC or ADNI in the question.")
    if is_direct and args.mode != "heuristic":
        parser.error(
            "--mode only applies to NL mode.  "
            "Direct tool mode is always deterministic (no LLM)."
        )

    # DeepSeek: apply default model; print compatibility warning
    if args.mode == "deepseek":
        if args.model is None:
            args.model = _DEEPSEEK_DEFAULT_MODEL
        print(
            "WARNING: DeepSeek mode uses the Anthropic SDK pointed at DeepSeek's endpoint.\n"
            "         Intent parsing sends Anthropic-format structured tool_choice=fill_intent.\n"
            "         DeepSeek's tool-use surface is compatible but may behave differently.\n"
            "         This endpoint (api.deepseek.com/anthropic) only accepts the model\n"
            "         names 'deepseek-v4-pro' or 'deepseek-v4-flash' — 'deepseek-chat' /\n"
            "         'deepseek-v4-chat' will 400 here (those route to a different,\n"
            "         OpenAI-protocol DeepSeek endpoint this codebase does not call).\n"
            "         If you hit a 400/422 error, try --mode heuristic or --mode anthropic.\n"
            "\n"
            "         Model selection IS propagated end-to-end (both intent parse and narration).\n"
            "         The orchestrator no longer hard-codes claude-haiku-4-5-20251001 when\n"
            "         --model is supplied.\n",
            file=sys.stderr,
        )

    # hf-local: model is required (validated again in _build_client, but fail fast here)
    if args.mode == "hf-local" and not args.model:
        parser.error(
            "--mode hf-local requires --model MODEL_NAME.\n"
            "Example: --model Qwen/Qwen2.5-7B-Instruct"
        )

    # --use-semantic-planner ALWAYS implies --use-enriched-planner (see
    # _resolve_planner_flags docstring — also enforced inside run_agent() itself).
    args.use_enriched_planner, args.use_semantic_planner, _planner_note = (
        _resolve_planner_flags(args.use_enriched_planner, args.use_semantic_planner)
    )
    if _planner_note:
        print(_planner_note + "\n", file=sys.stderr)

    # Enriched planner (classic or semantic) requires an LLM client
    if getattr(args, "use_enriched_planner", False) and args.mode == "heuristic":
        print(
            "WARNING: --use-enriched-planner/--use-semantic-planner requires an LLM client.\n"
            "         --mode heuristic has no client; the enriched planner guard will\n"
            "         short-circuit and the classic parse_intent path will run instead.\n"
            "         To use the enriched planner, add --mode anthropic or --mode deepseek.\n",
            file=sys.stderr,
        )

    # Load backend (this may do heavy I/O for local mode)
    print("Loading catalog and backend...", end=" ", flush=True)
    try:
        catalog, backend, backend_label, analysis_engine = _load_backend(args.backend)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"\nERROR loading backend: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        sys.exit(1)
    print("done.")

    # ── Direct tool mode ─────────────────────────────────────────────────
    if is_direct:
        try:
            _run_direct_tool(args, catalog, backend, backend_label)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"\nERROR in direct tool: {exc}", file=sys.stderr)
            if args.debug:
                traceback.print_exc()
        return

    # ── NL mode ──────────────────────────────────────────────────────────
    client = _build_client(
        args.mode, args.model, args.base_url,
        hf_cache_dir=getattr(args, "hf_cache_dir", None),
        hf_device_map=getattr(args, "hf_local_device_map", "auto"),
        hf_dtype=getattr(args, "hf_local_dtype", "auto"),
        hf_load_in_4bit=getattr(args, "hf_local_load_in_4bit", False),
        hf_max_new_tokens=getattr(args, "hf_local_max_new_tokens", 512),
    )

    if args.interactive:
        mode_label = args.mode + (f"/{args.model}" if args.model else "")
        print(
            f"\nalz-agent  ({mode_label}, {args.backend} backend)\n"
            f"Type 'exit' or 'quit' to stop.  Ctrl-C also works.\n"
        )
        try:
            while True:
                try:
                    question = input("alz-agent> ").strip()
                except EOFError:
                    break
                if not question:
                    continue
                if question.lower() in ("exit", "quit"):
                    break
                _ask_once_nl(
                    question, catalog, backend, backend_label,
                    client, args.mode, args.model,
                    args.show_intent, args.show_meta, args.debug,
                    analysis_engine=analysis_engine,
                    show_trace=args.show_trace,
                    show_plan=args.show_plan,
                    show_debug=args.show_debug,
                    use_enriched_planner=args.use_enriched_planner,
                    use_semantic_planner=args.use_semantic_planner,
                    show_raw_intent=args.show_raw_intent,
                )
        except KeyboardInterrupt:
            print("\nExiting.")
    else:
        _ask_once_nl(
            args.question, catalog, backend, backend_label,
            client, args.mode, args.model,
            args.show_intent, args.show_meta, args.debug,
            analysis_engine=analysis_engine,
            show_trace=args.show_trace,
            show_plan=args.show_plan,
            show_debug=args.show_debug,
            use_enriched_planner=args.use_enriched_planner,
            use_semantic_planner=args.use_semantic_planner,
            show_raw_intent=args.show_raw_intent,
        )


if __name__ == "__main__":
    main()
