#!/usr/bin/env bash
# Reruns of the X0 and agentic-baseline answers that hit a length limit, now that both servers
# allow 131,072 tokens of context:
#
#   x0      thinking-on answers cut off at 30,720 tokens (35B: 150, 4B: 174 of 780), again with
#           thinking on and the same sampling, at MAX_TOKENS_X0 (81,920, Qwen's setting for
#           competition maths). Lists: data-claude/reruns/x0_capped_<agent>.txt.
#   agentic the 35B's BrowseComp-Plus and Finance-Agent tasks (150), thinking off and on: on the
#           32k server 93 of its 150 thinking-off episodes ran out of context.
#
# A rerun answer is a fresh sample, not a continuation of the cut-off one; the merged arm takes
# the rerun answer for these questions and the original elsewhere.
#
# Three streams: the 4B's X0 rerun on :8003, and the 35B's X0 and agentic reruns side by side
# on :8002 (which also judges). Folds with a summary are skipped; interrupted folds resume.
#
#   ./scripts/long_reruns.sh
#   STREAMS="x0_4b" ./scripts/long_reruns.sh
set -u
cd "$(dirname "$0")/.." || exit 1

PYTHON=${PYTHON:-./env/bin/python}
LISTS=${LISTS:-data-claude/reruns}
OUT_ROOT=${OUT_ROOT:-data-claude/reruns}
LOG_DIR=${LOG_DIR:-data-claude/logs/reruns}
FOLDS=${FOLDS:-"0 1 2 3 4"}
MAX_TOKENS_X0=${MAX_TOKENS_X0:-81920}
MAX_TOKENS_AGENTIC=${MAX_TOKENS_AGENTIC:-8192}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-21600}
STREAMS=${STREAMS:-"x0_4b x0_35b agentic_35b"}
JUDGE_MODEL=${JUDGE_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}
JUDGE_API=${JUDGE_API:-http://127.0.0.1:8002/v1}
mkdir -p "$LOG_DIR"

ON="--thinking on --temperature 1.0 --top_p 0.95 --top_k 20 --min_p 0 --presence_penalty 1.5"
OFF="--thinking off --temperature 0.7 --top_p 0.8 --top_k 20 --min_p 0 --presence_penalty 1.5"

run() {  # out_group label model api workers fold dataset only max_tokens sampling...
    group=$1 label=$2 model=$3 api=$4 workers=$5 fold=$6 dataset=$7 only=$8 max_tokens=$9; shift 9
    out="$OUT_ROOT/$group/$label/fold$fold/bare_model"
    if [ -f "$out/holdout_summary.json" ]; then echo "=== $out done, skipping ==="; return; fi
    until curl -s -m 60 "$api/models" | grep -q "$model"; do
        echo "$(date -u +%FT%TZ) $model at $api not answering; waiting"; sleep 60
    done
    echo "=== $out starting $(date -u +%FT%TZ) ==="
    # shellcheck disable=SC2068
    $PYTHON -u scripts/bare_model_baseline.py --dataset_path "$dataset" --model_name "$model" \
        --api_base_url "$api" --split_seed 0 --n_folds 5 --fold "$fold" --workers "$workers" \
        --test_batch_size $((workers * 3)) --judge_model "$JUDGE_MODEL" --judge_api_base_url "$JUDGE_API" \
        --max_tokens "$max_tokens" --request_timeout "$REQUEST_TIMEOUT" --resume --only "$only" $@ \
        --out_dir "$out" > "$LOG_DIR/${group}_${label}_fold$fold.log" 2>&1
    status=$?
    echo "=== $out finished $(date -u +%FT%TZ) with status $status ==="
    grep -E "BARE MODEL" "$LOG_DIR/${group}_${label}_fold$fold.log"
}

stream_x0_4b() {
    for fold in $FOLDS; do
        run x0-long qwen3.5-4b-on Qwen/Qwen3.5-4B http://127.0.0.1:8003/v1 16 "$fold" \
            data-claude/tagged_hard "$LISTS/x0_capped_qwen3.5-4b-on.txt" "$MAX_TOKENS_X0" $ON
    done
}

stream_x0_35b() {
    for fold in $FOLDS; do
        run x0-long qwen3.6-35b-on nvidia/Qwen3.6-35B-A3B-NVFP4 http://127.0.0.1:8002/v1 6 "$fold" \
            data-claude/tagged_hard "$LISTS/x0_capped_qwen3.6-35b-on.txt" "$MAX_TOKENS_X0" $ON
    done
}

stream_agentic_35b() {
    for fold in $FOLDS; do
        for mode in off on; do
            [ $mode = on ] && s=$ON || s=$OFF
            # shellcheck disable=SC2086
            run agentic-131k qwen3.6-35b-$mode nvidia/Qwen3.6-35B-A3B-NVFP4 http://127.0.0.1:8002/v1 6 "$fold" \
                data-claude/tagged_agentic "$LISTS/agentic_long_context.txt" "$MAX_TOKENS_AGENTIC" $s
        done
    done
}

for s in $STREAMS; do "stream_$s" > "$LOG_DIR/stream_$s.log" 2>&1 & done
wait
