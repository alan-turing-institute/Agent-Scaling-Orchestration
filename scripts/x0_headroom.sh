#!/usr/bin/env bash
# X0, headroom on the hard pool: the 4B and the 35B answering alone, thinking off and on,
# 5 folds, each with Qwen's recommended sampling (no more temperature 0).
#
# Why: before an orchestrator can gain by choosing agents or by escalating from its own
# answer, some agent has to be right where another is wrong. These runs measure that,
# and give the first small answer matrix (4 agents x 780 questions) for replaying
# solve-first and escalation policies offline.
#
# Two streams, one per server, run side by side; each walks the folds in order, so a
# finished fold is a full replicate. The 4B with thinking off is sampled three times
# (s0-s2) for agreement-based escalation; the others once.
#
# Sampling is Qwen's "general tasks" setting for each mode (Qwen3.5/3.6 model cards):
#   thinking on   temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5
#   thinking off  temperature 0.7, top_p 0.8,  top_k 20, min_p 0, presence_penalty 1.5
# Every summary and prediction row records the settings it was run with.
#
# Questions go out in groups of three times the worker count: the script waits for a whole
# group before starting the next, and with answers up to 30k tokens a group of 5 left the
# servers running two or three requests behind one long answer. Grouping does not change
# answers or scores for one model answering alone.
#
# A 30k-token answer takes most of an hour with the server full (10-15 tokens a second per
# request), so the client waits up to REQUEST_TIMEOUT for each; at the default 300s every
# long answer timed out and was asked again from the start.
#
# MAX_TOKENS 30720 lets an answer run to the 32k context of the 35B server, less the
# prompt; the 4B gets the same, though its server allows 131k, so the two compare.
#
#   ./scripts/x0_headroom.sh            # both streams
#   STREAMS=4b ./scripts/x0_headroom.sh # one
set -u
cd "$(dirname "$0")/.." || exit 1

PYTHON=${PYTHON:-./env/bin/python}
DATASET=${DATASET:-data-claude/tagged_hard}
OUT_ROOT=${OUT_ROOT:-data-claude/x0-headroom}
LOG_DIR=${LOG_DIR:-data-claude/logs/x0-headroom}
FOLDS=${FOLDS:-"0 1 2 3 4"}
MAX_TOKENS=${MAX_TOKENS:-30720}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-10800}
STREAMS=${STREAMS:-"4b 35b"}
mkdir -p "$LOG_DIR"

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
        --test_batch_size $((workers * 3)) \
        --max_tokens "$MAX_TOKENS" --request_timeout "$REQUEST_TIMEOUT" --resume $@ --out_dir "$out" \
        > "$LOG_DIR/${label}_fold$fold.log" 2>&1
    status=$?
    echo "=== $out finished $(date -u +%FT%TZ) with status $status ==="
    grep -E "BARE MODEL" "$LOG_DIR/${label}_fold$fold.log"
}

stream_4b() {
    m=Qwen/Qwen3.5-4B a=http://127.0.0.1:8003/v1
    for fold in $FOLDS; do
        for s in 0 1 2; do run qwen3.5-4b-off-s$s $m $a 16 "$fold" $OFF; done
        run qwen3.5-4b-on $m $a 16 "$fold" $ON
    done
}

stream_35b() {
    m=nvidia/Qwen3.6-35B-A3B-NVFP4 a=http://127.0.0.1:8002/v1
    for fold in $FOLDS; do
        run qwen3.6-35b-off $m $a 12 "$fold" $OFF
        run qwen3.6-35b-on $m $a 12 "$fold" $ON
    done
}

for s in $STREAMS; do "stream_$s" > "$LOG_DIR/stream_$s.log" 2>&1 & done
wait
for label in $(ls "$OUT_ROOT"); do $PYTHON scripts/report_crossval.py "$OUT_ROOT/$label" || true; done
