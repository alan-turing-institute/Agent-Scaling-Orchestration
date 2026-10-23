#!/usr/bin/env bash
# Agentic baselines: the 4B and the 35B working through the agentic pool alone (BrowseComp-Plus,
# Finance-Agent, Plancraft, WorkBench; 350 tasks), thinking off and on, 5 folds, with the same
# sampling as X0 (Qwen's recommended settings, no more temperature 0).
#
# Two streams, one per server, run side by side; each walks the folds in order, so a finished
# fold is a full replicate. Folds with a summary are skipped, so a rerun picks up where it stopped.
#
# Sampling is Qwen's "general tasks" setting for each mode (Qwen3.5/3.6 model cards):
#   thinking on   temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5
#   thinking off  temperature 0.7, top_p 0.8,  top_k 20, min_p 0, presence_penalty 1.5
# Every summary and prediction row records the settings it was run with. The judge (the 35B,
# for BrowseComp-Plus and Finance-Agent) stays at temperature 0.
#
# MAX_TOKENS is per turn. The 35B's server has a 32k context, and vLLM refuses a turn whose
# history plus MAX_TOKENS exceeds it (the episode then ends as a context limit), so 8192 leaves
# it about 24k of history; the 4B's server allows 131k. Both get the same per-turn limit.
#
# The 35B also answers the judge's calls, so its stream runs fewer workers.
#
# WAIT_FOR_PID, if set, holds the start until that process exits (X0's driver).
#
#   WAIT_FOR_PID=1234 ./scripts/agentic_baselines.sh
#   STREAMS=4b ./scripts/agentic_baselines.sh
set -u
cd "$(dirname "$0")/.." || exit 1

PYTHON=${PYTHON:-./env/bin/python}
DATASET=${DATASET:-data-claude/tagged_agentic}
OUT_ROOT=${OUT_ROOT:-data-claude/agentic-baselines}
LOG_DIR=${LOG_DIR:-data-claude/logs/agentic-baselines}
FOLDS=${FOLDS:-"0 1 2 3 4"}
MAX_TOKENS=${MAX_TOKENS:-8192}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-10800}
STREAMS=${STREAMS:-"4b 35b"}
JUDGE_MODEL=${JUDGE_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}
JUDGE_API=${JUDGE_API:-http://127.0.0.1:8002/v1}
mkdir -p "$LOG_DIR"

if [ -n "${WAIT_FOR_PID:-}" ]; then
    echo "$(date -u +%FT%TZ) waiting for process $WAIT_FOR_PID to exit"
    while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do sleep 300; done
    echo "$(date -u +%FT%TZ) process $WAIT_FOR_PID has exited; starting"
fi

ON="--thinking on --temperature 1.0 --top_p 0.95 --top_k 20 --min_p 0 --presence_penalty 1.5"
OFF="--thinking off --temperature 0.7 --top_p 0.8 --top_k 20 --min_p 0 --presence_penalty 1.5"

run() {  # label model api workers fold sampling...
    label=$1 model=$2 api=$3 workers=$4 fold=$5; shift 5
    out="$OUT_ROOT/$label/fold$fold/bare_model"
    if [ -f "$out/holdout_summary.json" ]; then echo "=== $out done, skipping ==="; return; fi
    until curl -s -m 60 "$api/models" | grep -q "$model"; do
        echo "$(date -u +%FT%TZ) $model at $api not answering; waiting"; sleep 60
    done
    echo "=== $out starting $(date -u +%FT%TZ) ==="
    # shellcheck disable=SC2068
    $PYTHON -u scripts/bare_model_baseline.py --dataset_path "$DATASET" --model_name "$model" \
        --api_base_url "$api" --split_seed 0 --n_folds 5 --fold "$fold" --workers "$workers" \
        --test_batch_size $((workers * 3)) --judge_model "$JUDGE_MODEL" --judge_api_base_url "$JUDGE_API" \
        --max_tokens "$MAX_TOKENS" --request_timeout "$REQUEST_TIMEOUT" --resume $@ --out_dir "$out" \
        > "$LOG_DIR/${label}_fold$fold.log" 2>&1
    status=$?
    echo "=== $out finished $(date -u +%FT%TZ) with status $status ==="
    grep -E "BARE MODEL" "$LOG_DIR/${label}_fold$fold.log"
}

stream_4b() {
    m=Qwen/Qwen3.5-4B a=http://127.0.0.1:8003/v1
    for fold in $FOLDS; do
        run qwen3.5-4b-off $m $a 12 "$fold" $OFF
        run qwen3.5-4b-on $m $a 12 "$fold" $ON
    done
}

stream_35b() {
    m=nvidia/Qwen3.6-35B-A3B-NVFP4 a=http://127.0.0.1:8002/v1
    for fold in $FOLDS; do
        run qwen3.6-35b-off $m $a 8 "$fold" $OFF
        run qwen3.6-35b-on $m $a 8 "$fold" $ON
    done
}

for s in $STREAMS; do "stream_$s" > "$LOG_DIR/stream_$s.log" 2>&1 & done
wait
for label in $(ls "$OUT_ROOT"); do $PYTHON scripts/report_crossval.py "$OUT_ROOT/$label" || true; done
