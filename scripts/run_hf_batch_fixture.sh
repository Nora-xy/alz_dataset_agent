#!/usr/bin/env bash
set -u

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
BACKEND="${BACKEND:-fixture}"
CACHE_DIR="${HF_HUB_CACHE:-}"

if [ -z "$CACHE_DIR" ]; then
  echo "ERROR: HF_HUB_CACHE is not set."
  echo "Run: export HF_HUB_CACHE=/hpc/group/396-Alzheim26/xl461/.hf_home/hub"
  exit 1
fi

mkdir -p logs/hf_runs
BATCH_ID="$(date +%Y%m%d_%H%M%S)_hf_fixture_batch"
SUMMARY="logs/hf_runs/${BATCH_ID}_summary.txt"

run_one () {
  local name="$1"
  local max_tokens="$2"
  local question="$3"
  local log="logs/hf_runs/${BATCH_ID}_${name}.log"

  {
    echo "date: $(date)"
    echo "git_commit: $(git rev-parse --short HEAD)"
    echo "env: ${CONDA_DEFAULT_ENV:-unknown}"
    echo "model: $MODEL"
    echo "backend: $BACKEND"
    echo "max_new_tokens: $max_tokens"
    echo "question: $question"
    echo "----------------------------------------"

    START=$(date +%s)

    timeout 300s python scripts/ask_agent.py \
      --mode hf-local \
      --model "$MODEL" \
      --hf-cache-dir "$CACHE_DIR" \
      --hf-local-device-map auto \
      --hf-local-max-new-tokens "$max_tokens" \
      -q "$question" \
      --backend "$BACKEND" \
      --show-intent \
      --show-meta

    STATUS=$?
    END=$(date +%s)

    echo "----------------------------------------"
    echo "exit_status: $STATUS"
    echo "elapsed_seconds: $((END - START))"
  } 2>&1 | tee "$log"

  if grep -q "^ERROR:" "$log"; then
    RESULT="ERROR"
  elif grep -q "PARSED INTENT:" "$log" && grep -q "ANSWER:" "$log"; then
    RESULT="RAN"
  else
    RESULT="CHECK"
  fi

  {
    echo "[$RESULT] $name"
    echo "  question: $question"
    echo "  log: $log"
    grep -m 1 "elapsed_seconds:" "$log" | sed 's/^/  /'
    grep -m 1 "^ERROR:" "$log" | sed 's/^/  /'
    echo
  } | tee -a "$SUMMARY"
}

echo "Batch summary: $SUMMARY"
echo

run_one "count_baseline_mci" 128 \
"Count participants with baseline MCI. Do not infer AD etiology."

run_one "count_baseline_dementia_no_ad" 128 \
"Count participants with baseline dementia, regardless of Alzheimer etiology."

run_one "count_etiologic_ad_not_dementia" 128 \
"Count participants diagnosed with etiologic Alzheimer's disease, not all-cause dementia."

run_one "nacc_count_baseline_mci" 128 \
"In the NACC dataset, count participants with baseline MCI. Do not infer AD etiology."

run_one "adni_count_baseline_mci" 128 \
"In the ADNI dataset, count participants with baseline MCI."

run_one "cognition_any_imaging_overlap" 128 \
"How many participants have both cognition and any imaging data?"

run_one "mci_cognition_mri_overlap" 128 \
"Among baseline MCI participants, how many have both cognition and MRI data?"

run_one "nacc_ad_avg_age_first_dx" 256 \
"In NACC dataset, among those who were diagnosed as AD, what is the average age when they were first diagnosed?"

echo
echo "Done."
echo "Summary saved to: $SUMMARY"
