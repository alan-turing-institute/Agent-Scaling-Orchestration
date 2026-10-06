#!/usr/bin/env bash
# E2: does choosing agents matter more under some topologies than others?
#
# Selection {random, no_memory, continual} x topology {vote, debate,
# centralized, synthesis}, on the 5 stratified folds the earlier
# cross-validation used. Teams of four; in centralized and synthesis the last
# agent picked leads the other three. Debate is one exchange round, full text.
# The random selector is seeded by --split_seed, so every topology gets the same
# random teams and the arms pair question for question.
#
# Reproducibility: the vLLM servers give different greedy text for the same
# request depending on what else is in the batch, so every arm runs with
# --max_inflight 1. To get throughput back, the agent model is served several
# times over (PORTS) and each worker sends one arm at a time to its own server.
# Several copies of the 0.8B fit on the box and barely compete for bandwidth.
#
# Jobs sit in a queue ordered fold by fold, longest first, and the workers pull
# from it. A job whose holdout_summary.json exists is skipped; a job that was
# interrupted is cleared and rerun from scratch, because the scoreboard of a
# half-finished continual run would carry over. Rerunning the script resumes.
#
#   PORTS="8001 8003 8004 8005" ./scripts/e2_topology_arms.sh
#
# Orchestrator: the 35B on 8002. Results: data-claude/e2-topology/<label>/
# fold<k>/<selection>-<topology>/; logs beside them under data-claude/logs/.
set -u

cd "$(dirname "$0")/.." || exit 1

PYTHON=${PYTHON:-./env/bin/python}
MODEL=${MODEL:-Qwen/Qwen3.5-0.8B}
LABEL=${LABEL:-qwen3.5-0.8b}
PORTS=${PORTS:-"8001 8003 8004 8005"}
ORCH_MODEL=${ORCH_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}
ORCH_API=${ORCH_API:-http://127.0.0.1:8002/v1}
ITERATIONS=${ITERATIONS:-30}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SELECTIONS=${SELECTIONS:-"continual no_memory random"}
TOPOLOGIES=${TOPOLOGIES:-"debate vote centralized synthesis"}
OUT_ROOT=${OUT_ROOT:-data-claude/e2-topology/$LABEL}
LOG_DIR=${LOG_DIR:-data-claude/logs/e2-topology/$LABEL}
mkdir -p "$OUT_ROOT" "$LOG_DIR"

QUEUE="$LOG_DIR/queue.txt"
: > "$QUEUE"
# Fold by fold so every finished fold is a full replicate; within a fold the
# order above puts the longest jobs (training, debate) first to balance workers.
for fold in $FOLDS; do
    for selection in $SELECTIONS; do
        for topology in $TOPOLOGIES; do
            echo "$fold $selection $topology" >> "$QUEUE"
        done
    done
done

pop() {  # prints the next job and removes it, or nothing when the queue is empty
    flock "$QUEUE.lock" sh -c 'head -n 1 "$1"; sed -i 1d "$1"' _ "$QUEUE"
}

wait_for() {  # url model
    until curl -s -m 60 "$1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}" \
        | grep -q '"choices"'; do
        echo "$(date -u +%FT%TZ) $2 at $1 not answering; waiting"
        sleep 60
    done
}

run_job() {  # port fold selection topology
    port=$1; fold=$2; selection=$3; topology=$4
    api="http://127.0.0.1:$port/v1"
    out="$OUT_ROOT/fold$fold/$selection-$topology"
    log="$LOG_DIR/fold${fold}_$selection-$topology.log"
    if [ -f "$out/holdout_summary.json" ]; then
        echo "=== $out already done, skipping ==="
        return
    fi
    rm -rf "$out"  # an interrupted run leaves a scoreboard that would carry over
    wait_for "$api" "$MODEL"
    case $selection in no_memory|continual) wait_for "$ORCH_API" "$ORCH_MODEL" ;; esac
    echo "=== $(date -u +%FT%TZ) :$port $out starting ==="

    common="--dataset_path data-claude/tagged_dataset --model_name $MODEL --api_base_url $api \
        --split_seed 0 --n_folds 5 --fold $fold --num_samples 5 --team_size 4 --seed 0 \
        --test_batch_size 5 --eval_workers 5 --summary_every 1 \
        --topology $topology --rounds 1 --handoff full --max_inflight 1"
    orch="--orchestrator_model $ORCH_MODEL --orchestrator_api_base_url $ORCH_API"
    case $selection in
        random)    args="--iterations 0 --selection random --memory none" ;;
        no_memory) args="$orch --iterations 0 --memory none" ;;
        continual) args="$orch --iterations $ITERATIONS" ;;
        *) echo "unknown selection $selection"; return ;;
    esac
    # shellcheck disable=SC2086
    $PYTHON -u src/train_orchestrator.py $common $args --out_dir "$out" > "$log" 2>&1
    status=$?
    echo "=== $(date -u +%FT%TZ) :$port $out finished (exit $status) ==="
}

worker() {  # port
    while true; do
        job=$(pop)
        [ -z "$job" ] && break
        # shellcheck disable=SC2086
        run_job "$1" $job
    done
    echo "=== $(date -u +%FT%TZ) worker :$1 done ==="
}

echo "=== $(date -u +%FT%TZ) $(wc -l < "$QUEUE") jobs, workers on ports $PORTS ==="
for port in $PORTS; do
    worker "$port" &
done
wait
echo "=== $(date -u +%FT%TZ) all done ==="
