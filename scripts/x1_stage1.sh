#!/usr/bin/env bash
# X1 stage 1: serve one model at a time and run every worker run it has (scripts/x1_plan.py).
#
# For each model in serving order (personas on the base 4B first, then the specialists small
# to large): start its server (scripts/x1_serve.sh), run its pending runs PARALLEL at a time,
# tear it down. Runs with a summary are skipped and each run resumes by batch, so the whole
# script can be rerun after any interruption.
#
# A watchdog guards against the silent vLLM stall seen on 2026-10-23 (engine at 0 tokens/s with
# requests open, no error, every request then waiting out its timeout): when the engine logs
# no prompt or generation throughput with requests running for STALL_SECONDS, the runs are
# stopped, the server restarted and the pending runs started again, up to ATTEMPTS times.
#
#   ./scripts/x1_stage1.sh                        # every model
#   MODELS="rl-math bigbang" ./scripts/x1_stage1.sh
#   FOLDS=0 SAMPLES=0 ./scripts/x1_stage1.sh      # one replicate first
#   PLAN_ARGS="--teams T6,T7 --limit 2" ROOT=data-claude/x1-teams-test ./scripts/x1_stage1.sh   # a check
set -u
cd "$(dirname "$0")/.." || exit 1

PYTHON=${PYTHON:-./env/bin/python}
ROOT=${ROOT:-data-claude/x1-teams}
LOG_DIR=${LOG_DIR:-$ROOT/logs/stage1}
PORT=${PORT:-8001}
API=http://127.0.0.1:$PORT/v1
NAME=${NAME:-x1-serve}
PARALLEL=${PARALLEL:-2}
ATTEMPTS=${ATTEMPTS:-3}
STALL_SECONDS=${STALL_SECONDS:-300}
MODELS=${MODELS:-$($PYTHON scripts/x1_plan.py models | cut -f1)}
SELECT=()
[ -n "${FOLDS:-}" ] && SELECT+=(--folds "$FOLDS")
[ -n "${SAMPLES:-}" ] && SELECT+=(--samples "$SAMPLES")
read -r -a EXTRA <<< "${PLAN_ARGS:-}"
SELECT+=(--root "$ROOT" "${EXTRA[@]}")
mkdir -p "$LOG_DIR"

stalled() {  # every engine report in the window says nothing moved while requests were open
    docker logs --since "${STALL_SECONDS}s" "$NAME" 2>&1 | grep "Engine 000" > "$LOG_DIR/.engine" || return 1
    [ "$(wc -l < "$LOG_DIR/.engine")" -ge $((STALL_SECONDS / 10 - 2)) ] || return 1
    ! grep -qvE "prompt throughput: 0\.0 tokens/s, Avg generation throughput: 0\.0 tokens/s, Running: [1-9]" "$LOG_DIR/.engine"
}

serve() {  # model key
    line=$($PYTHON scripts/x1_plan.py models | awk -F'\t' -v m="$1" '$1 == m')
    profile=$(echo "$line" | cut -f2) quant=$(echo "$line" | cut -f3) handle=$(echo "$line" | cut -f4)
    [ "$quant" = "-" ] && quant=""
    # Requests set enable_thinking themselves; the server default only matters for the judge-free stage 1.
    NAME=$NAME PORT=$PORT PROFILE=$profile HANDLE=$handle QUANT=$quant THINKING=0 bash scripts/x1_serve.sh > /dev/null
    for _ in $(seq 1 180); do
        curl -sf "$API/models" | grep -q "$handle" && return 0
        [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ] || break
        sleep 10
    done
    echo "$(date -u +%FT%TZ) $1: server did not come up"; docker logs "$NAME" > "$LOG_DIR/$1.serve-fail.log" 2>&1
    return 1
}

for model in $MODELS; do
    for attempt in $(seq 1 "$ATTEMPTS"); do
        $PYTHON scripts/x1_plan.py commands --model "$model" --api "$API" --pending "${SELECT[@]}" > "$LOG_DIR/$model.pending"
        pending=$(wc -l < "$LOG_DIR/$model.pending")
        if [ "$pending" -eq 0 ]; then echo "=== $model: nothing pending"; break; fi
        echo "=== $model attempt $attempt: $pending runs $(date -u +%FT%TZ)"
        serve "$model" || break
        # Each run logs beside its results; xargs runs PARALLEL of them at once.
        tr '\n' '\0' < "$LOG_DIR/$model.pending" |
            xargs -0 -P "$PARALLEL" -I{} bash -c 'out=$(echo "$1" | sed -E "s/.*--out_dir ([^ ]+).*/\1/"); mkdir -p "$out"; eval "$1" > "$out/run.log" 2>&1; echo "$(date -u +%FT%TZ) $? $out"' _ {} \
            >> "$LOG_DIR/$model.done" &
        runs=$!
        result=done
        while kill -0 "$runs" 2>/dev/null; do
            sleep 60
            if stalled; then
                result=stalled
                echo "$(date -u +%FT%TZ) $model: engine stalled; restarting"
                pkill -f -- "--out_dir $ROOT/stage1/"  # this model's runs: one model is served at a time
                wait "$runs" 2>/dev/null
                break
            fi
        done
        docker logs "$NAME" > "$LOG_DIR/$model.attempt$attempt.server.log" 2>&1
        docker rm -f "$NAME" > /dev/null 2>&1
        [ "$result" = done ] && break
    done
    left=$($PYTHON scripts/x1_plan.py commands --model "$model" --pending "${SELECT[@]}" | wc -l)
    echo "=== $model finished $(date -u +%FT%TZ): $left runs still pending"
done
echo "STAGE1 DONE $(date -u +%FT%TZ)"
