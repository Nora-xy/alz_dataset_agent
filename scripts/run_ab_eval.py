"""A/B evaluation: compare LLM providers on plan decomposition quality.

Runs run_plan_ab() from eval/harness.py across two or more providers,
reporting per-question results and an overall summary.

Usage (two providers):
    # DeepSeek vs a HuggingFace-hosted model
    $env:DEEPSEEK_API_KEY = "sk-..."
    $env:HF_TOKEN         = "hf_..."
    python scripts/run_ab_eval.py `
        --provider deepseek `
        --provider huggingface `
        --model-huggingface meta-llama/Llama-3.3-70B-Instruct `
        --backend fixture

    # Anthropic vs DeepSeek
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    $env:DEEPSEEK_API_KEY  = "sk-..."
    python scripts/run_ab_eval.py `
        --provider anthropic --provider deepseek --backend fixture

Output: a table printed to stdout; exits 0 when all providers pass all checks.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# ---------------------------------------------------------------------------
# A/B question suite  (5 question types from the spec)
# ---------------------------------------------------------------------------

_AB_QUESTIONS: list[dict] = [
    {
        "id": "event_anchored_ad_age",
        "question": "Among all participants, what is the average age when first diagnosed with AD?",
        "expected_analysis_type": "event_anchored_stat",
        "expected_value_column": "NACCAGE",
        "expected_event_filter": {"NACCALZD": 1},
        "expected_visit_anchor": "first_matching_event_visit",
        "reject_mci_default": True,
    },
    {
        "id": "modeling_design",
        "question": (
            "Among MCI participants, can we build a 3-year prediction model for AD conversion "
            "using cognitive and MRI features?"
        ),
        "expected_analysis_type": "modeling_design",
        "expected_value_column": None,
        "expected_event_filter": None,
        "expected_visit_anchor": None,
        "reject_mci_default": False,
    },
    {
        "id": "group_comparison",
        "question": "Do participants with a history of heart attack have lower MMSE scores?",
        "expected_analysis_type": "group_comparison",
        "expected_value_column": "NACCMMSE",
        "expected_event_filter": None,
        "expected_visit_anchor": None,
        "reject_mci_default": True,
    },
    {
        "id": "modality_ambiguity",
        "question": (
            "How many participants have both cognition assessments and any kind of imaging data?"
        ),
        "expected_analysis_type": "cohort_size",
        "expected_value_column": None,
        "expected_event_filter": None,
        "expected_visit_anchor": None,
        "reject_mci_default": True,
    },
    {
        "id": "ad_vs_dementia",
        "question": (
            "What is the distribution of follow-up years for MCI participants who will convert "
            "to all-cause dementia (not AD-specific)?"
        ),
        "expected_analysis_type": None,  # any valid type acceptable
        "expected_value_column": None,
        "expected_event_filter": None,
        "expected_visit_anchor": None,
        "reject_mci_default": False,
    },
]


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------

_DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-pro"
_ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_HF_DEFAULT_BASE_URL = "https://router.huggingface.co/v1"


def _build_provider(
    provider: str,
    model: str | None,
    base_url: str | None,
    hf_token_env: str = "HF_TOKEN",
) -> tuple[str, object]:
    """Return (display_name, client) for a provider string."""
    from alz_agent.agent.providers import build_planner_client  # noqa: PLC0415

    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            print(f"ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
            sys.exit(1)
        resolved_model = model or _ANTHROPIC_DEFAULT_MODEL
        client = build_planner_client("anthropic", model=resolved_model, api_key=key)
        return f"anthropic/{resolved_model}", client

    if provider == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            print("ERROR: DEEPSEEK_API_KEY not set.", file=sys.stderr)
            sys.exit(1)
        resolved_model = model or _DEEPSEEK_DEFAULT_MODEL
        client = build_planner_client(
            "deepseek", model=resolved_model, api_key=key, base_url=base_url
        )
        return f"deepseek/{resolved_model}", client

    if provider == "huggingface":
        key = os.environ.get(hf_token_env, "")
        if not key:
            print(f"ERROR: {hf_token_env} not set.", file=sys.stderr)
            sys.exit(1)
        if not model:
            print(
                "ERROR: --model-huggingface MODEL is required for the huggingface provider.",
                file=sys.stderr,
            )
            sys.exit(1)
        client = build_planner_client(
            "huggingface",
            model=model,
            api_key=key,
            base_url=base_url or _HF_DEFAULT_BASE_URL,
        )
        return f"huggingface/{model}", client

    print(f"ERROR: Unknown provider {provider!r}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def _load_catalog(backend: str) -> list:
    if backend == "fixture":
        from alz_agent.eval.harness import default_catalog  # noqa: PLC0415
        return default_catalog()

    # local: fast NACC catalog (no RDD PDF)
    import yaml  # noqa: PLC0415
    config_path = _ROOT / "config" / "paths.yaml"
    if not config_path.exists():
        print("ERROR: config/paths.yaml not found; use --backend fixture", file=sys.stderr)
        sys.exit(1)
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    from alz_agent.catalog.nacc_parser import parse_nacc  # noqa: PLC0415
    from alz_agent.eval.harness import default_catalog  # noqa: PLC0415
    nacc_cfg = config["nacc"]
    try:
        nacc_cat = parse_nacc(
            data_csv=_ROOT / nacc_cfg["data_csv"],
            modality_map_path=_ROOT / nacc_cfg["modality_map"],
            rdd_pdf=None,
        )
        base = default_catalog()
        base_ids = {e.variable_id for e in base}
        return base + [e for e in nacc_cat if e.variable_id not in base_ids]
    except Exception as exc:
        print(f"WARNING: NACC catalog parse failed ({exc}); using fixture catalog.", file=sys.stderr)
        return default_catalog()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_ab_eval.py",
        description="A/B evaluation comparing LLM providers on plan decomposition.",
    )
    parser.add_argument(
        "--provider", dest="providers", action="append", default=[],
        choices=["anthropic", "deepseek", "huggingface"],
        metavar="PROVIDER",
        help=(
            "Provider to include in the A/B comparison.  "
            "Repeat to add multiple.  Choices: anthropic | deepseek | huggingface."
        ),
    )
    parser.add_argument(
        "--model-anthropic", default=None, metavar="MODEL",
        help=f"Anthropic model name (default: {_ANTHROPIC_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--model-deepseek", default=None, metavar="MODEL",
        help=f"DeepSeek model name (default: {_DEEPSEEK_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--model-huggingface", default=None, metavar="MODEL",
        help="HuggingFace model name (required if --provider huggingface).",
    )
    parser.add_argument(
        "--base-url", default=None, metavar="URL",
        help="Override API base URL for all providers.",
    )
    parser.add_argument(
        "--hf-token-env", default="HF_TOKEN", metavar="ENV_VAR",
        help="Env var name for HF API token (default: HF_TOKEN).",
    )
    parser.add_argument(
        "--backend", choices=["fixture", "local"], default="fixture",
        help="Catalog backend: fixture (default; synthetic) or local (real NACC).",
    )
    parser.add_argument(
        "--question-id", default=None, metavar="ID",
        help="Run only a specific question by its id (e.g. event_anchored_ad_age).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.providers:
        print("ERROR: specify at least one --provider.", file=sys.stderr)
        sys.exit(1)

    # Build provider list
    providers: list[tuple[str, object]] = []
    for p in args.providers:
        model = {
            "anthropic": args.model_anthropic,
            "deepseek": args.model_deepseek,
            "huggingface": args.model_huggingface,
        }[p]
        name, client = _build_provider(p, model, args.base_url, args.hf_token_env)
        providers.append((name, client))

    catalog = _load_catalog(args.backend)
    print(f"Catalog: {len(catalog)} entries  ({args.backend} backend)")
    print(f"Providers: {[n for n, _ in providers]}")
    print()

    from alz_agent.eval.harness import run_plan_ab  # noqa: PLC0415

    questions = _AB_QUESTIONS
    if args.question_id:
        questions = [q for q in questions if q["id"] == args.question_id]
        if not questions:
            print(f"ERROR: no question with id {args.question_id!r}", file=sys.stderr)
            sys.exit(1)

    all_passed = True
    for q in questions:
        print(f"{'='*66}")
        print(f"Q [{q['id']}]: {q['question']}")
        result = run_plan_ab(
            q["question"],
            catalog=catalog,
            providers=providers,
            expected_analysis_type=q["expected_analysis_type"],
            expected_value_column=q["expected_value_column"],
            expected_event_filter=q["expected_event_filter"],
            expected_visit_anchor=q["expected_visit_anchor"],
            reject_mci_default=q["reject_mci_default"],
        )
        print(result.summary())
        if not all(r.decomposed_correctly for r in result.rows):
            all_passed = False
        print()

    print(f"{'='*66}")
    print(f"OVERALL A/B RESULT: {'PASS' if all_passed else 'FAIL'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
