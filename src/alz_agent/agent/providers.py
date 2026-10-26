"""Provider-neutral planner client protocol for intent.py and orchestrator.py.

Three classes implement the same interface so callers never depend on a
specific SDK's response format:

  AnthropicPlannerClient  — wraps anthropic.Anthropic (also handles DeepSeek
                            which uses the Anthropic SDK at a different base URL)
  LocalHFPlannerClient    — runs open-source models LOCALLY via transformers.
                            Weights are cached on disk after first download; all
                            subsequent inference is 100 % offline, no data leaves
                            the machine. This is the recommended Hugging Face path.
  HFOpenAIPlannerClient   — DEPRECATED. Wraps openai.OpenAI pointed at Hugging
                            Face's cloud router (router.huggingface.co). Uses
                            cloud inference, not local-only, may incur costs.
                            Kept for backward compatibility only; do not use in
                            new code and do not advertise in CLI help.

All three expose:
  invoke_tool(messages, tool_schema, max_tokens) -> dict | None
  complete_text(messages, max_tokens)            -> str
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Schema conversion helpers
# ---------------------------------------------------------------------------


def _anthropic_tool_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic tool schema (input_schema) to OpenAI function format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        },
    }


def _split_system(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract a leading system-role message for Anthropic-style API calls.

    The unified message format allows a leading {"role": "system", ...} entry.
    Anthropic's SDK takes ``system`` as a separate kwarg; this helper splits
    it out.  OpenAI-compatible providers receive the system message as-is.
    """
    if messages and messages[0].get("role") == "system":
        return str(messages[0]["content"]), messages[1:]
    return None, messages


# ---------------------------------------------------------------------------
# Marker base class
# ---------------------------------------------------------------------------


class PlannerClient:
    """Marker base class for all provider-neutral planner clients.

    isinstance(obj, PlannerClient) is the canonical dispatch check used by
    intent.py, orchestrator.py, and analysis/engine.py to distinguish real
    protocol-aware clients from raw anthropic.Anthropic objects and from
    unittest.mock.MagicMock instances in tests (which respond True to hasattr
    for any attribute, breaking hasattr-based dispatch).
    """


# ---------------------------------------------------------------------------
# AnthropicPlannerClient
# ---------------------------------------------------------------------------


class AnthropicPlannerClient(PlannerClient):
    """Planner client wrapping anthropic.Anthropic (or DeepSeek via Anthropic SDK).

    Exposes the unified invoke_tool / complete_text interface so callers in
    intent.py and orchestrator.py are provider-neutral.
    """

    def __init__(
        self, anthropic_client: Any, model: str | None = None, *, disable_thinking: bool = False
    ) -> None:
        self._client = anthropic_client
        self._model = model or "claude-haiku-4-5-20251001"
        # DeepSeek's reasoning models (deepseek-v4-pro and deepseek-v4-flash — the
        # only two model names this endpoint accepts) default to "thinking"
        # mode, which (a) rejects forced tool_choice with a 400 error and (b) can
        # consume the entire max_tokens budget on hidden reasoning before ever
        # emitting a text block, starving the JSON-fallback completion path.
        # Real Anthropic models are untested against this flag here, so it is
        # only ever set for the "deepseek" provider (see build_planner_client).
        self._disable_thinking = disable_thinking

    def invoke_tool(
        self,
        messages: list[dict[str, Any]],
        tool_schema: dict[str, Any],
        max_tokens: int = 1500,
    ) -> dict[str, Any] | None:
        """Call the model with forced tool use.

        Returns the parsed arguments dict if the model invoked the tool,
        or None if no tool call was found in the response.
        """
        tool_name = tool_schema["name"]
        system, user_messages = _split_system(messages)
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": tool_name},
            messages=user_messages,
        )
        if system:
            kwargs["system"] = system
        if self._disable_thinking:
            kwargs["thinking"] = {"type": "disabled"}
        response = self._client.messages.create(**kwargs)
        for block in response.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == tool_name
            ):
                return dict(block.input)
        return None

    def complete_text(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> str:
        """Call the model for free-text completion. Returns the response text."""
        system, user_messages = _split_system(messages)
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=user_messages,
        )
        if system:
            kwargs["system"] = system
        if self._disable_thinking:
            kwargs["thinking"] = {"type": "disabled"}
        response = self._client.messages.create(**kwargs)
        text_block = next(
            (b for b in response.content if getattr(b, "type", None) == "text"),
            None,
        )
        if text_block is None:
            raise RuntimeError("LLM returned no text content.")
        return text_block.text


# ---------------------------------------------------------------------------
# LocalHFPlannerClient
# ---------------------------------------------------------------------------


_LOCAL_HF_IMPORT_ERROR = (
    "Local Hugging Face mode requires transformers, torch, and accelerate.\n"
    "Install with: pip install transformers torch accelerate sentencepiece protobuf\n"
    "Optional for 4-bit quantization: pip install bitsandbytes"
)

_LOCAL_HF_4BIT_ERROR = (
    "--hf-local-load-in-4bit requires bitsandbytes.\n"
    "Install with: pip install bitsandbytes"
)


class LocalHFPlannerClient(PlannerClient):
    """Planner client that runs open-source models locally via transformers.

    On first call, model weights are downloaded from the Hugging Face Hub to
    the local cache directory (HF_HOME or --hf-cache-dir). All subsequent
    inference loads from disk and runs fully offline — no data ever leaves the
    machine during inference.

    Switch models by passing a different model_id string:
        LocalHFPlannerClient("Qwen/Qwen2.5-7B-Instruct")
        LocalHFPlannerClient("meta-llama/Llama-3.2-3B-Instruct")

    invoke_tool always returns None so callers fall back to the JSON-prompt
    path (complete_text). Most open-source models do not expose OpenAI-style
    function calling with the reliability needed here, so the JSON path is the
    correct fallback and produces equally good results for this workload.
    """

    def __init__(
        self,
        model_id: str,
        *,
        cache_dir: str | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
    ) -> None:
        self._model_id = model_id
        self._cache_dir = cache_dir
        self._device_map = device_map
        self._torch_dtype_str = torch_dtype
        self._load_in_4bit = load_in_4bit
        self._max_new_tokens = max_new_tokens
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None  # stored after import; lets tests inject a mock

    def _ensure_loaded(self) -> None:
        """Lazy-load tokenizer and model on first inference call."""
        if self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(_LOCAL_HF_IMPORT_ERROR) from exc

        self._torch = torch  # stored so complete_text never re-imports torch

        dtype_map: dict[str, Any] = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        resolved_dtype: Any = dtype_map.get(self._torch_dtype_str, "auto")

        load_kwargs: dict[str, Any] = {}
        if self._cache_dir:
            load_kwargs["cache_dir"] = self._cache_dir

        model_kwargs: dict[str, Any] = {
            "torch_dtype": resolved_dtype,
            "device_map": self._device_map,
            **load_kwargs,
        }

        if self._load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes as _  # noqa: F401
            except ImportError as exc:
                raise ImportError(_LOCAL_HF_4BIT_ERROR) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, **load_kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_id, **model_kwargs)

    def invoke_tool(
        self,
        messages: list[dict[str, Any]],
        tool_schema: dict[str, Any],
        max_tokens: int = 1500,
    ) -> dict[str, Any] | None:
        """Always returns None — callers fall back to the JSON-prompt path."""
        return None

    def complete_text(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> str:
        """Generate text using the locally loaded model."""
        self._ensure_loaded()
        torch = self._torch  # use the module reference stored during _ensure_loaded

        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback for models without a chat template
            prompt = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            ) + "\nASSISTANT:"

        model_device = next(self._model.parameters()).device
        inputs = self._tokenizer(prompt, return_tensors="pt").to(model_device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, self._max_new_tokens),
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# HFOpenAIPlannerClient  (DEPRECATED — cloud inference, not local-only)
# ---------------------------------------------------------------------------


class HFOpenAIPlannerClient(PlannerClient):
    """DEPRECATED: uses Hugging Face hosted inference; not local-only; may incur costs.

    This class wraps openai.OpenAI pointed at https://router.huggingface.co/v1
    and sends requests to Hugging Face's cloud API. Use LocalHFPlannerClient
    instead for privacy-sensitive / no-cloud use.

    Kept for backward compatibility only. Do not use in new code.
    """

    def __init__(self, openai_client: Any, model: str) -> None:
        self._client = openai_client
        self._model = model

    def invoke_tool(
        self,
        messages: list[dict[str, Any]],
        tool_schema: dict[str, Any],
        max_tokens: int = 1500,
    ) -> dict[str, Any] | None:
        """Attempt structured tool use via OpenAI function-calling format.

        Returns the parsed arguments dict, or None when:
        - the model doesn't support tool use
        - no tool call appears in the response
        - the arguments can't be parsed as JSON

        A None return triggers the JSON-prompt fallback in _plan_analysis_internal.
        """
        tool_name = tool_schema["name"]
        oai_tool = _anthropic_tool_to_openai(tool_schema)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                tools=[oai_tool],
                tool_choice={"type": "function", "function": {"name": tool_name}},
                messages=messages,
            )
        except Exception as exc:
            err = str(exc).lower()
            if any(kw in err for kw in ("tool", "function", "unsupported", "not support")):
                return None  # caller falls back to JSON prompt
            raise

        choice = response.choices[0] if response.choices else None
        if choice and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                if tc.function.name == tool_name:
                    try:
                        return json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        return None
        return None

    def complete_text(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> str:
        """Call the model for free-text completion. Returns the response text."""
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=messages,
        )
        choice = response.choices[0] if response.choices else None
        if choice is None or not choice.message.content:
            raise RuntimeError("HF model returned no text content.")
        return choice.message.content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_planner_client(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    hf_cache_dir: str | None = None,
    hf_device_map: str = "auto",
    hf_dtype: str = "auto",
    hf_load_in_4bit: bool = False,
    hf_max_new_tokens: int = 512,
) -> "AnthropicPlannerClient | LocalHFPlannerClient | HFOpenAIPlannerClient":
    """Construct the appropriate planner client for the given provider.

    Args:
        provider:  "anthropic" | "deepseek" | "hf-local" | "huggingface" (deprecated)
        model:     LLM model name. Required for "hf-local" and "huggingface";
                   optional for "anthropic" (default: claude-haiku-4-5-20251001)
                   and "deepseek" (default: deepseek-v4-flash). The DeepSeek
                   Anthropic-compatible endpoint (api.deepseek.com/anthropic) only
                   accepts the literal model names "deepseek-v4-pro" or
                   "deepseek-v4-flash" — "deepseek-chat"/"deepseek-v4-chat" are
                   rejected with a 400 here (those names route to a different,
                   OpenAI-protocol DeepSeek endpoint this codebase does not call).
        api_key:   Provider API key (not used for "hf-local").
        base_url:  Override API endpoint ("anthropic" and "deepseek" only).

        hf_cache_dir:       Local cache directory for model weights (hf-local only).
                            Sets HF_HOME semantics via cache_dir= kwarg.
        hf_device_map:      Device mapping string passed to from_pretrained (hf-local).
        hf_dtype:           Torch dtype: "auto" | "float16" | "bfloat16" | "float32".
        hf_load_in_4bit:    Enable 4-bit quantization via bitsandbytes (hf-local).
        hf_max_new_tokens:  Cap on tokens generated per call (hf-local).
    """
    if provider == "anthropic":
        import anthropic  # noqa: PLC0415

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        client = anthropic.Anthropic(**kwargs)
        return AnthropicPlannerClient(client, model=model or "claude-haiku-4-5-20251001")

    if provider == "deepseek":
        import anthropic  # noqa: PLC0415

        resolved_url = base_url or "https://api.deepseek.com/anthropic"
        client = anthropic.Anthropic(api_key=api_key, base_url=resolved_url)
        return AnthropicPlannerClient(
            client, model=model or "deepseek-v4-flash", disable_thinking=True
        )

    if provider == "hf-local":
        if not model:
            raise ValueError(
                "model is required for hf-local.\n"
                "Example: model='Qwen/Qwen2.5-7B-Instruct'"
            )
        return LocalHFPlannerClient(
            model,
            cache_dir=hf_cache_dir,
            device_map=hf_device_map,
            torch_dtype=hf_dtype,
            load_in_4bit=hf_load_in_4bit,
            max_new_tokens=hf_max_new_tokens,
        )

    if provider == "huggingface":
        # DEPRECATED: uses HF cloud router API, not local-only.
        # Use provider="hf-local" for local transformers execution.
        import warnings
        warnings.warn(
            "provider='huggingface' uses the Hugging Face cloud router API and is deprecated. "
            "Use provider='hf-local' for local transformers execution.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the deprecated huggingface provider.\n"
                "Install: pip install openai"
            ) from None
        if not model:
            raise ValueError(
                "model is required for the huggingface provider.\n"
                "Example: model='meta-llama/Llama-3.3-70B-Instruct'"
            )
        resolved_url = base_url or "https://router.huggingface.co/v1"
        oai_client = OpenAI(api_key=api_key or "", base_url=resolved_url)
        return HFOpenAIPlannerClient(oai_client, model=model)

    raise ValueError(
        f"Unknown provider {provider!r}. Valid: anthropic | deepseek | hf-local"
    )
