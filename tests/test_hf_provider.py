"""Tests for the HF / provider-neutral planner client layer.

Covers:
  - AnthropicPlannerClient.invoke_tool + complete_text via mock Anthropic client
  - LocalHFPlannerClient: complete_text (mocked transformers), invoke_tool always None,
    missing-dependency errors, 4-bit import error
  - HFOpenAIPlannerClient (deprecated): invoke_tool success/fallbacks, complete_text
  - build_planner_client factory: anthropic, deepseek, hf-local, huggingface (deprecated)
  - hf-local does NOT construct HFOpenAIPlannerClient
  - hf-local does NOT use https://router.huggingface.co/v1
  - _plan_analysis_internal uses invoke_tool when client is a PlannerClient
  - _parse_with_json_prompt uses complete_text when client is a PlannerClient
  - Provider selection from CLI (ask_agent._build_client)
  - CLI --mode choices do not expose huggingface as a normal user-facing mode
  - CLI help does not advertise router.huggingface.co

Privacy invariant: no row-level data is sent — we verify that content passed
to mocks only contains the NL question, catalog context, and schema.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alz_agent.agent.providers import (
    AnthropicPlannerClient,
    HFOpenAIPlannerClient,
    LocalHFPlannerClient,
    PlannerClient,
    _anthropic_tool_to_openai,
    _split_system,
    build_planner_client,
)


# ---------------------------------------------------------------------------
# Test double: a real PlannerClient subclass so isinstance() dispatch works
# ---------------------------------------------------------------------------


class _StubPlannerClient(PlannerClient):
    """Minimal PlannerClient for integration tests — no real SDK calls."""

    def __init__(
        self,
        invoke_tool_return: dict | None = None,
        complete_text_return: str = "",
    ) -> None:
        self._invoke_tool_return = invoke_tool_return
        self._complete_text_return = complete_text_return
        self.invoke_tool_calls: list = []
        self.complete_text_calls: list = []

    def invoke_tool(self, messages, tool_schema, max_tokens=1500):
        self.invoke_tool_calls.append((messages, tool_schema))
        return self._invoke_tool_return

    def complete_text(self, messages, max_tokens=1024):
        self.complete_text_calls.append(messages)
        return self._complete_text_return


# ---------------------------------------------------------------------------
# Helpers — build fake OpenAI-style responses
# ---------------------------------------------------------------------------


def _make_oai_tool_response(tool_name: str, args: dict) -> Any:
    """Build a minimal mock that looks like an openai ChatCompletion with a tool call."""
    fn = SimpleNamespace(name=tool_name, arguments=json.dumps(args))
    tc = SimpleNamespace(function=fn)
    msg = SimpleNamespace(tool_calls=[tc], content=None)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def _make_oai_text_response(text: str) -> Any:
    """Build a minimal mock that looks like an openai ChatCompletion text response."""
    msg = SimpleNamespace(tool_calls=None, content=text)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def _make_oai_empty_response() -> Any:
    """Simulate a response with no tool calls and no content."""
    msg = SimpleNamespace(tool_calls=None, content=None)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def _make_anthropic_tool_response(tool_name: str, args: dict) -> Any:
    """Minimal Anthropic messages.create response with a tool_use block."""
    block = SimpleNamespace(type="tool_use", name=tool_name, input=args)
    return SimpleNamespace(content=[block])


def _make_anthropic_text_response(text: str) -> Any:
    """Minimal Anthropic messages.create response with a text block."""
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block])


# ---------------------------------------------------------------------------
# _split_system
# ---------------------------------------------------------------------------


def test_split_system_extracts_leading_system():
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]
    system, rest = _split_system(messages)
    assert system == "Be concise."
    assert rest == [{"role": "user", "content": "Hello"}]


def test_split_system_no_system():
    messages = [{"role": "user", "content": "Hello"}]
    system, rest = _split_system(messages)
    assert system is None
    assert rest == messages


# ---------------------------------------------------------------------------
# _anthropic_tool_to_openai
# ---------------------------------------------------------------------------


def test_anthropic_tool_to_openai():
    tool = {
        "name": "plan_analysis",
        "description": "Decompose question.",
        "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
    }
    result = _anthropic_tool_to_openai(tool)
    assert result["type"] == "function"
    assert result["function"]["name"] == "plan_analysis"
    assert result["function"]["description"] == "Decompose question."
    assert result["function"]["parameters"] == tool["input_schema"]


# ---------------------------------------------------------------------------
# AnthropicPlannerClient
# ---------------------------------------------------------------------------


def test_anthropic_client_invoke_tool_success():
    args = {"analysis_type": "cohort_size", "research_intent_summary": "test"}
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_anthropic_tool_response("plan_analysis", args)

    aclient = AnthropicPlannerClient(mock_client, model="claude-haiku-4-5-20251001")
    tool_schema = {"name": "plan_analysis", "input_schema": {}}
    result = aclient.invoke_tool(
        messages=[{"role": "user", "content": "How many MCI?"}],
        tool_schema=tool_schema,
    )
    assert result == args
    # Verify system not passed when no system message
    kwargs = mock_client.messages.create.call_args.kwargs
    assert "system" not in kwargs


def test_anthropic_client_invoke_tool_with_system():
    args = {"analysis_type": "cohort_size", "research_intent_summary": "test"}
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_anthropic_tool_response("plan_analysis", args)

    aclient = AnthropicPlannerClient(mock_client, model="claude-haiku-4-5-20251001")
    tool_schema = {"name": "plan_analysis", "input_schema": {}}
    result = aclient.invoke_tool(
        messages=[
            {"role": "system", "content": "Be an expert."},
            {"role": "user", "content": "How many MCI?"},
        ],
        tool_schema=tool_schema,
    )
    assert result == args
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["system"] == "Be an expert."


def test_anthropic_client_invoke_tool_no_tool_call():
    mock_client = MagicMock()
    block = SimpleNamespace(type="text", text="I don't know.")
    mock_client.messages.create.return_value = SimpleNamespace(content=[block])

    aclient = AnthropicPlannerClient(mock_client)
    tool_schema = {"name": "plan_analysis", "input_schema": {}}
    result = aclient.invoke_tool(
        messages=[{"role": "user", "content": "Q"}],
        tool_schema=tool_schema,
    )
    assert result is None


def test_anthropic_client_complete_text():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_anthropic_text_response("The answer is 42.")

    aclient = AnthropicPlannerClient(mock_client, model="claude-haiku-4-5-20251001")
    text = aclient.complete_text(messages=[{"role": "user", "content": "What is 6x7?"}])
    assert text == "The answer is 42."


def test_anthropic_client_complete_text_no_text_block():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = SimpleNamespace(content=[])

    aclient = AnthropicPlannerClient(mock_client)
    with pytest.raises(RuntimeError, match="no text content"):
        aclient.complete_text(messages=[{"role": "user", "content": "Q"}])


# ---------------------------------------------------------------------------
# LocalHFPlannerClient — all tests use injected mocks; no real torch/transformers
# ---------------------------------------------------------------------------
#
# Design: LocalHFPlannerClient stores self._torch after _ensure_loaded() runs.
# Tests bypass _ensure_loaded entirely by setting _tokenizer, _model, _torch
# directly on the client instance. This lets every test run without installing
# or importing torch, avoiding the numpy/BLAS abort on this machine.


def _make_mock_torch() -> MagicMock:
    """Build a mock that covers the torch API surface used by complete_text."""
    mock_torch = MagicMock()
    # torch.no_grad() is used as a context manager
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=None)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    mock_torch.no_grad.return_value = mock_ctx
    return mock_torch


def _make_mock_tokenizer(decoded_text: str = "42 participants.") -> MagicMock:
    """Build a mock tokenizer — no torch tensors; everything is MagicMock."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<prompt>"

    # tokenizer("<prompt>", return_tensors="pt") → inputs dict
    input_ids_mock = MagicMock()
    input_ids_mock.shape = [1, 5]  # plain list — supports shape[-1]
    inputs_mock = MagicMock()
    inputs_mock.__getitem__ = MagicMock(side_effect=lambda k: input_ids_mock if k == "input_ids" else MagicMock())
    inputs_mock.to.return_value = inputs_mock  # .to(device) returns self
    tokenizer.return_value = inputs_mock

    tokenizer.eos_token_id = 2
    tokenizer.decode.return_value = decoded_text
    return tokenizer


def _make_mock_model() -> MagicMock:
    """Build a mock model — no torch tensors; generate() returns a subscriptable mock."""
    model = MagicMock()
    # output_ids[0][5:] → MagicMock (passed to tokenizer.decode)
    row_mock = MagicMock()
    output_mock = MagicMock()
    output_mock.__getitem__ = MagicMock(return_value=row_mock)
    model.generate.return_value = output_mock

    mock_param = MagicMock()
    mock_param.device = "cpu"
    model.parameters.return_value = iter([mock_param])
    return model


def _inject_client(model_id: str = "Qwen/Qwen2.5-7B-Instruct", **kwargs) -> LocalHFPlannerClient:
    """Return a LocalHFPlannerClient with mocked internals already injected."""
    client = LocalHFPlannerClient(model_id, **kwargs)
    client._tokenizer = _make_mock_tokenizer()
    client._model = _make_mock_model()
    client._torch = _make_mock_torch()
    return client


def test_local_hf_invoke_tool_always_returns_none():
    """LocalHFPlannerClient.invoke_tool always returns None (JSON-prompt fallback path)."""
    client = LocalHFPlannerClient("Qwen/Qwen2.5-7B-Instruct")
    result = client.invoke_tool(
        messages=[{"role": "user", "content": "How many MCI?"}],
        tool_schema={"name": "plan_analysis", "input_schema": {}},
    )
    assert result is None


def test_local_hf_complete_text_calls_chat_template_and_generate():
    """complete_text applies chat template, calls generate, and decodes output."""
    client = _inject_client()
    client._tokenizer.decode.return_value = "42 participants."

    result = client.complete_text([{"role": "user", "content": "How many?"}])

    assert client._tokenizer.apply_chat_template.called
    assert client._model.generate.called
    assert client._tokenizer.decode.called
    assert result == "42 participants."


def test_local_hf_complete_text_no_network():
    """complete_text never contacts any remote URL during inference."""
    client = _inject_client()

    with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network call")):
        client.complete_text([{"role": "user", "content": "Q"}])  # should not raise


def test_local_hf_complete_text_chat_template_fallback():
    """When apply_chat_template raises, complete_text falls back to plain concatenation."""
    client = _inject_client()
    client._tokenizer.apply_chat_template.side_effect = AttributeError("no chat template")

    client.complete_text([
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "How many?"},
    ])

    # tokenizer should have been called with a plain-text prompt (not the chat template output)
    call_args = client._tokenizer.call_args
    prompt_arg = call_args[0][0] if call_args[0] else call_args.kwargs.get("text", "")
    assert "ASSISTANT:" in prompt_arg


def test_local_hf_missing_transformers_raises():
    """Missing transformers raises ImportError with install instructions."""
    client = LocalHFPlannerClient("Qwen/Qwen2.5-7B-Instruct")
    # Simulate ImportError from inside _ensure_loaded by patching builtins.__import__
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def mock_import(name, *args, **kwargs):
        if name in ("torch", "transformers"):
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="transformers"):
            client._ensure_loaded()


def _make_mock_transformers(tok_cls=None, mdl_cls=None) -> MagicMock:
    """Build a mock transformers module with AutoTokenizer and AutoModelForCausalLM."""
    m = MagicMock()
    m.AutoTokenizer = tok_cls or MagicMock()
    m.AutoModelForCausalLM = mdl_cls or MagicMock()
    m.BitsAndBytesConfig = MagicMock()
    return m


def _make_mock_torch_module() -> MagicMock:
    """Build a mock torch module with dtype sentinels."""
    t = MagicMock()
    t.float16 = "float16_sentinel"
    t.bfloat16 = "bfloat16_sentinel"
    t.float32 = "float32_sentinel"
    return t


def test_local_hf_4bit_missing_bitsandbytes_raises():
    """load_in_4bit=True without bitsandbytes raises ImportError with hint."""
    mock_torch = _make_mock_torch_module()
    mock_transformers = _make_mock_transformers()

    # sys.modules["bitsandbytes"] = None causes "import bitsandbytes" to raise ImportError
    with patch.dict("sys.modules", {
        "torch": mock_torch,
        "transformers": mock_transformers,
        "bitsandbytes": None,
    }):
        client = LocalHFPlannerClient("Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True)
        with pytest.raises(ImportError, match="bitsandbytes"):
            client._ensure_loaded()


def test_local_hf_cache_dir_forwarded():
    """cache_dir is forwarded to AutoTokenizer and AutoModelForCausalLM.from_pretrained."""
    mock_torch = _make_mock_torch_module()

    mock_tok_instance = MagicMock()
    mock_mdl_instance = MagicMock()
    mock_tok_cls = MagicMock()
    mock_tok_cls.from_pretrained.return_value = mock_tok_instance
    mock_mdl_cls = MagicMock()
    mock_mdl_cls.from_pretrained.return_value = mock_mdl_instance

    mock_transformers = _make_mock_transformers(tok_cls=mock_tok_cls, mdl_cls=mock_mdl_cls)

    with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
        client = LocalHFPlannerClient("some/model", cache_dir="D:/hf_cache")
        client._ensure_loaded()

    assert mock_tok_cls.from_pretrained.call_args.kwargs.get("cache_dir") == "D:/hf_cache"
    assert mock_mdl_cls.from_pretrained.call_args.kwargs.get("cache_dir") == "D:/hf_cache"


def test_local_hf_is_planner_client():
    """LocalHFPlannerClient must be a PlannerClient subclass for isinstance dispatch."""
    client = LocalHFPlannerClient("some/model")
    assert isinstance(client, PlannerClient)


def test_local_hf_does_not_use_router_url():
    """LocalHFPlannerClient never references router.huggingface.co."""
    import inspect

    source = inspect.getsource(LocalHFPlannerClient)
    assert "router.huggingface.co" not in source


# ---------------------------------------------------------------------------
# HFOpenAIPlannerClient (deprecated — kept for backward compat)
# ---------------------------------------------------------------------------


def test_hf_client_invoke_tool_success():
    args = {"analysis_type": "group_comparison", "research_intent_summary": "test"}
    mock_oai = MagicMock()
    mock_oai.chat.completions.create.return_value = _make_oai_tool_response("plan_analysis", args)

    hf = HFOpenAIPlannerClient(mock_oai, model="meta-llama/Llama-3.3-70B-Instruct")
    tool_schema = {"name": "plan_analysis", "description": "d", "input_schema": {}}
    result = hf.invoke_tool(
        messages=[{"role": "user", "content": "Compare groups."}],
        tool_schema=tool_schema,
    )
    assert result == args


def test_hf_client_invoke_tool_no_tool_call_returns_none():
    """When HF doesn't return a tool call, invoke_tool returns None (not raise)."""
    mock_oai = MagicMock()
    mock_oai.chat.completions.create.return_value = _make_oai_text_response("No tool for you.")

    hf = HFOpenAIPlannerClient(mock_oai, model="some-model")
    tool_schema = {"name": "plan_analysis", "description": "d", "input_schema": {}}
    result = hf.invoke_tool(
        messages=[{"role": "user", "content": "Q"}],
        tool_schema=tool_schema,
    )
    assert result is None


def test_hf_client_invoke_tool_bad_json_returns_none():
    """Tool call with un-parseable arguments → None, not a crash."""
    fn = SimpleNamespace(name="plan_analysis", arguments="NOT JSON {{{{")
    tc = SimpleNamespace(function=fn)
    msg = SimpleNamespace(tool_calls=[tc], content=None)
    choice = SimpleNamespace(message=msg)
    mock_oai = MagicMock()
    mock_oai.chat.completions.create.return_value = SimpleNamespace(choices=[choice])

    hf = HFOpenAIPlannerClient(mock_oai, model="some-model")
    tool_schema = {"name": "plan_analysis", "description": "d", "input_schema": {}}
    result = hf.invoke_tool([{"role": "user", "content": "Q"}], tool_schema)
    assert result is None


def test_hf_client_invoke_tool_unsupported_exception_returns_none():
    """API errors mentioning 'tool' or 'unsupported' → None, not re-raise."""
    mock_oai = MagicMock()
    mock_oai.chat.completions.create.side_effect = ValueError("tool use is not supported")

    hf = HFOpenAIPlannerClient(mock_oai, model="some-model")
    tool_schema = {"name": "plan_analysis", "description": "d", "input_schema": {}}
    result = hf.invoke_tool([{"role": "user", "content": "Q"}], tool_schema)
    assert result is None


def test_hf_client_invoke_tool_other_exception_reraises():
    """Non-tool API errors are re-raised so they surface to the caller."""
    mock_oai = MagicMock()
    mock_oai.chat.completions.create.side_effect = ConnectionError("network down")

    hf = HFOpenAIPlannerClient(mock_oai, model="some-model")
    tool_schema = {"name": "plan_analysis", "description": "d", "input_schema": {}}
    with pytest.raises(ConnectionError):
        hf.invoke_tool([{"role": "user", "content": "Q"}], tool_schema)


def test_hf_client_complete_text():
    mock_oai = MagicMock()
    mock_oai.chat.completions.create.return_value = _make_oai_text_response("42 participants.")

    hf = HFOpenAIPlannerClient(mock_oai, model="some-model")
    text = hf.complete_text(messages=[{"role": "user", "content": "How many?"}])
    assert text == "42 participants."


def test_hf_client_complete_text_empty_raises():
    mock_oai = MagicMock()
    mock_oai.chat.completions.create.return_value = _make_oai_empty_response()

    hf = HFOpenAIPlannerClient(mock_oai, model="some-model")
    with pytest.raises(RuntimeError, match="no text content"):
        hf.complete_text(messages=[{"role": "user", "content": "Q"}])


# ---------------------------------------------------------------------------
# build_planner_client factory
# ---------------------------------------------------------------------------


def test_build_planner_client_anthropic():
    with patch("anthropic.Anthropic") as MockAnth:
        MockAnth.return_value = MagicMock()
        client = build_planner_client("anthropic", api_key="sk-test")
    assert isinstance(client, AnthropicPlannerClient)


def test_build_planner_client_deepseek():
    with patch("anthropic.Anthropic") as MockAnth:
        MockAnth.return_value = MagicMock()
        client = build_planner_client("deepseek", api_key="sk-ds", model="deepseek-v4-pro")
    assert isinstance(client, AnthropicPlannerClient)
    # Verify DeepSeek base URL was used
    call_kwargs = MockAnth.call_args.kwargs
    assert "deepseek" in call_kwargs.get("base_url", "")


def test_build_planner_client_hf_local():
    """build_planner_client('hf-local', ...) returns LocalHFPlannerClient."""
    client = build_planner_client("hf-local", model="Qwen/Qwen2.5-7B-Instruct")
    assert isinstance(client, LocalHFPlannerClient)
    assert client._model_id == "Qwen/Qwen2.5-7B-Instruct"


def test_build_planner_client_hf_local_not_hf_cloud():
    """hf-local must NOT construct HFOpenAIPlannerClient."""
    client = build_planner_client("hf-local", model="meta-llama/Llama-3.2-3B-Instruct")
    assert not isinstance(client, HFOpenAIPlannerClient)


def test_build_planner_client_hf_local_does_not_use_router_url():
    """hf-local never wires up router.huggingface.co as an endpoint."""
    client = build_planner_client(
        "hf-local",
        model="Qwen/Qwen2.5-7B-Instruct",
        hf_cache_dir=None,
    )
    # LocalHFPlannerClient has no _base_url or similar attribute
    assert not hasattr(client, "_base_url")
    # Confirm the class source doesn't embed the cloud URL
    import inspect
    source = inspect.getsource(type(client))
    assert "router.huggingface.co" not in source


def test_build_planner_client_hf_local_requires_model():
    """hf-local raises ValueError when model is omitted."""
    with pytest.raises(ValueError, match="model is required"):
        build_planner_client("hf-local", model=None)


def test_build_planner_client_hf_local_forwards_cache_dir():
    """cache_dir kwarg is threaded through to LocalHFPlannerClient."""
    client = build_planner_client(
        "hf-local", model="some/model", hf_cache_dir="D:/hf_cache"
    )
    assert isinstance(client, LocalHFPlannerClient)
    assert client._cache_dir == "D:/hf_cache"


def test_build_planner_client_hf_local_forwards_4bit():
    """load_in_4bit kwarg is threaded through to LocalHFPlannerClient."""
    client = build_planner_client(
        "hf-local", model="some/model", hf_load_in_4bit=True
    )
    assert client._load_in_4bit is True


def test_build_planner_client_huggingface_deprecated():
    """build_planner_client('huggingface', ...) emits DeprecationWarning."""
    try:
        import openai as _  # noqa: F401
    except ImportError:
        pytest.skip("openai not installed")

    with patch("openai.OpenAI") as MockOAI:
        MockOAI.return_value = MagicMock()
        with pytest.warns(DeprecationWarning, match="deprecated"):
            client = build_planner_client(
                "huggingface",
                model="meta-llama/Llama-3.3-70B-Instruct",
                api_key="hf_test",
            )
    assert isinstance(client, HFOpenAIPlannerClient)


def test_build_planner_client_huggingface_requires_model():
    """Deprecated huggingface provider still validates that model is provided."""
    try:
        import openai as _  # noqa: F401
    except ImportError:
        pytest.skip("openai not installed")
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="model is required"):
            build_planner_client("huggingface", model=None, api_key="hf_test")


def test_build_planner_client_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        build_planner_client("bogus")


def test_build_planner_client_unknown_error_mentions_hf_local():
    """Error message for unknown provider lists hf-local as valid, not huggingface."""
    with pytest.raises(ValueError) as exc_info:
        build_planner_client("bogus")
    msg = str(exc_info.value)
    assert "hf-local" in msg
    assert "huggingface" not in msg  # cloud mode not advertised


# ---------------------------------------------------------------------------
# Integration: _plan_analysis_internal uses invoke_tool protocol
# ---------------------------------------------------------------------------


def _make_minimal_plan_args() -> dict:
    return {
        "analysis_type": "event_anchored_stat",
        "research_intent_summary": "average age at first AD diagnosis",
        "requires_longitudinal_data": True,
        "candidate_label": "average age at first AD diagnosis",
        "id_space": "A",
        "covariates": [],
        "unresolved_items": [],
        "value_variable": {
            "role": "value_variable",
            "researcher_label": "age",
            "semantic_definition": "participant age in years",
            "selected_raw_code": "NACCAGE",
            "resolution_status": "resolved",
            "required": True,
        },
        "event": {
            "role": "event",
            "researcher_label": "first AD diagnosis",
            "semantic_definition": "first visit with etiologic AD attribution",
            "selected_raw_code": "NACCALZD",
            "filter_expression": {"NACCALZD": 1},
            "resolution_status": "resolved",
            "required": True,
        },
        "visit_anchor": "first_matching_event_visit",
        "statistic": "mean",
    }


def test_plan_analysis_internal_uses_invoke_tool():
    """_plan_analysis_internal calls client.invoke_tool when the client is a PlannerClient."""
    from alz_agent.agent.intent import _plan_analysis_internal
    from alz_agent.schemas import CatalogEntry, IdSpace, Modality, SourceDataset

    catalog = [
        CatalogEntry(
            variable_id="NACC.NACCAGE",
            concept="age",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCAGE",
            release_version="nacc70",
            modality=Modality.DEMOGRAPHIC,
        ),
        CatalogEntry(
            variable_id="NACC.NACCALZD",
            concept="alzheimer_etiology",
            source_dataset=SourceDataset.NACC,
            id_space=IdSpace.A,
            raw_code="NACCALZD",
            release_version="nacc70",
            modality=Modality.OTHER,
        ),
    ]

    stub = _StubPlannerClient(invoke_tool_return=_make_minimal_plan_args())

    plan, debug = _plan_analysis_internal(
        "What is the average age at first AD diagnosis?",
        stub,
        catalog=catalog,
    )
    assert len(stub.invoke_tool_calls) == 1
    assert plan.analysis_type == "event_anchored_stat"


def test_plan_analysis_internal_no_tool_call_raises():
    """When invoke_tool returns None, _plan_analysis_internal raises RuntimeError."""
    from alz_agent.agent.intent import _plan_analysis_internal

    stub = _StubPlannerClient(invoke_tool_return=None)

    with pytest.raises(RuntimeError, match="did not invoke"):
        _plan_analysis_internal("some question", stub, catalog=[])


# ---------------------------------------------------------------------------
# Integration: JSON fallback via complete_text
# ---------------------------------------------------------------------------


_VALID_JSON_INTENT = json.dumps({
    "question_type": "cohort_size",
    "target_population": ["mci"],
    "target_etiology": None,
    "candidate_label": "cohort size",
    "horizon_years": None,
    "modalities": [],
    "stat_type": "mean",
    "analysis_type": None,
    "id_space": "A",
    "pre_index_only": True,
    "group_variable": None,
    "outcome_variable": None,
    "unresolved_fields": [],
    "covariate_labels": [],
    "covariate_columns": [],
})


def test_parse_with_json_prompt_uses_complete_text():
    """_parse_with_json_prompt calls complete_text when client is a PlannerClient."""
    from alz_agent.agent.intent import _parse_with_json_prompt

    stub = _StubPlannerClient(complete_text_return=_VALID_JSON_INTENT)

    intent = _parse_with_json_prompt("How many MCI participants?", stub, catalog=[])
    assert len(stub.complete_text_calls) == 1
    assert intent.parsing_mode == "llm"


def test_parse_with_json_prompt_invalid_json_raises():
    """If complete_text returns non-JSON, a RuntimeError is raised (not crash)."""
    from alz_agent.agent.intent import _parse_with_json_prompt

    stub = _StubPlannerClient(complete_text_return="not valid json at all")

    with pytest.raises(RuntimeError, match="invalid JSON"):
        _parse_with_json_prompt("Q?", stub, catalog=[])


# ---------------------------------------------------------------------------
# Integration: _parse_with_llm falls back when invoke_tool returns None
# ---------------------------------------------------------------------------


def test_parse_with_llm_falls_back_to_json_prompt_on_no_tool_call():
    """When invoke_tool returns None → RuntimeError → JSON fallback is triggered."""
    from alz_agent.agent.intent import _parse_with_llm

    stub = _StubPlannerClient(
        invoke_tool_return=None,
        complete_text_return=_VALID_JSON_INTENT,
    )

    intent = _parse_with_llm("How many MCI?", stub, catalog=[])
    assert intent.parsing_mode == "llm"
    assert len(stub.invoke_tool_calls) == 1
    assert len(stub.complete_text_calls) == 1


# ---------------------------------------------------------------------------
# Provider selection from ask_agent._build_client (no real keys / no download)
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import ask_agent  # noqa: E402


def test_build_client_heuristic_returns_none():
    result = ask_agent._build_client("heuristic", model=None, base_url=None)
    assert result is None


def test_build_client_hf_local_returns_local_hf_planner_client():
    """--mode hf-local constructs LocalHFPlannerClient, not HFOpenAIPlannerClient."""
    result = ask_agent._build_client(
        "hf-local", model="Qwen/Qwen2.5-7B-Instruct", base_url=None
    )
    assert isinstance(result, LocalHFPlannerClient)
    assert not isinstance(result, HFOpenAIPlannerClient)


def test_build_client_hf_local_no_api_key_needed():
    """hf-local does not require any environment variable."""
    # Should not raise even with no env vars set
    result = ask_agent._build_client(
        "hf-local", model="meta-llama/Llama-3.2-3B-Instruct", base_url=None
    )
    assert isinstance(result, LocalHFPlannerClient)


def test_build_client_hf_local_model_required_exits(capsys):
    """Missing --model for hf-local causes sys.exit(1)."""
    with pytest.raises(SystemExit) as exc_info:
        ask_agent._build_client("hf-local", model=None, base_url=None)
    assert exc_info.value.code == 1


def test_build_client_hf_local_forwards_cache_dir():
    """--hf-cache-dir is threaded through to LocalHFPlannerClient."""
    result = ask_agent._build_client(
        "hf-local", model="some/model", base_url=None,
        hf_cache_dir="D:/hf_cache",
    )
    assert isinstance(result, LocalHFPlannerClient)
    assert result._cache_dir == "D:/hf_cache"


def test_build_client_hf_local_forwards_4bit():
    result = ask_agent._build_client(
        "hf-local", model="some/model", base_url=None,
        hf_load_in_4bit=True,
    )
    assert result._load_in_4bit is True


def test_build_client_anthropic_missing_key_exits(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        ask_agent._build_client("anthropic", model=None, base_url=None)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# CLI help does not expose huggingface cloud mode as a normal option
# ---------------------------------------------------------------------------


def test_cli_mode_choices_do_not_include_huggingface():
    """--mode argparse choices must not list 'huggingface' as a valid user-facing option."""
    parser = ask_agent._build_arg_parser()
    mode_action = next(
        a for a in parser._actions if getattr(a, "dest", None) == "mode"
    )
    assert "huggingface" not in (mode_action.choices or [])


def test_cli_mode_choices_include_hf_local():
    """--mode argparse choices must include 'hf-local'."""
    parser = ask_agent._build_arg_parser()
    mode_action = next(
        a for a in parser._actions if getattr(a, "dest", None) == "mode"
    )
    assert "hf-local" in (mode_action.choices or [])


def test_cli_help_does_not_advertise_router_url(capsys):
    """CLI --help output must not contain router.huggingface.co."""
    parser = ask_agent._build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    captured = capsys.readouterr()
    assert "router.huggingface.co" not in captured.out
    assert "router.huggingface.co" not in captured.err


# ---------------------------------------------------------------------------
# hf-local compact JSON planner tests have moved to tests/test_hf_local_intent.py
# (schema expanded from the original 3-field intent="count"/population=
# "baseline_mci"-only shape to count/modality_overlap/aggregate_stat/unsupported
# with a semantic-mismatch guard — see docs/HF_LOCAL_TESTING_NOTE.md).
# ---------------------------------------------------------------------------
