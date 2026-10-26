# Local Hugging Face testing note

Local model inference is optional and experimental. The CLI mode is hf-local
and requires a model name plus the hf-local dependency extra. Model weights
download on first use and are cached; subsequent inference can run locally.
The model parses the question, while the project's deterministic backend
computes counts.

In earlier fixture tests, Qwen2.5 1.5B and 3B models loaded and ran on a GPU.
The small models sometimes emitted malformed structured output or added filters
that the question did not request. The compact intent schema, retry, and
semantic-mismatch checks now reject several of those failures before tool
execution. Even with those checks, model latency was too high for comfortable
interactive use in the tested setup.

Start with synthetic data:

~~~powershell
python -m pip install -e ".[hf-local,catalog,rdd,retrieval]"
python scripts/ask_agent.py --mode hf-local --model Qwen/Qwen2.5-3B-Instruct --backend fixture -q "For NACC, count participants with baseline MCI." --show-intent --show-meta
~~~

Check the parsed intent and answer before trying --backend local. The current
compact schema does not support every question handled by the API planner,
including the multi-horizon worked example in the README. More capable
hardware or models may change latency and accuracy; the historical tests do
not establish performance on a new setup.

