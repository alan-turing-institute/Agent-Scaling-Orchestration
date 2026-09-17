#!/usr/bin/env bash
# Experiment 1: what the orchestrator's memory is worth, and whether its update
# schedule matters. Four arms over the same split and seeds:
#
#   continual   scoreboard refreshed after every task
#   batched     scoreboard refreshed every 10 tasks
#   no_memory   scoreboard written but never read
#   random      no orchestrator at all; teams drawn uniformly from the pool
#
# The first three ask what the memory is worth given that a team is being
# chosen. The fourth asks what the choosing is worth, and is the floor the
# other three have to clear before any difference between them is interesting.
#
# The arms run one after another rather than together. Two earlier attempts ran
# them concurrently and the vLLM engine wedged both times, roughly ten minutes
# in, with requests in flight and generation throughput at zero - once with
# speculative decoding and once without. A single run keeps about twenty
# requests in flight, well inside what this server has been benchmarked at.
#
# ARMS overrides which arms run, so a single arm can be added to a finished
# round without repeating the ones already scored:
#
#   ARMS=random ./scripts/experiment1_schedules.sh
set -u

cd "$(dirname "$0")/.." || exit 1

PYTHON=${PYTHON:-./env/bin/python}
OUT_ROOT=${OUT_ROOT:-data-claude/orchestrator}
LOG_DIR=${LOG_DIR:-data-claude/logs}
ITERATIONS=${ITERATIONS:-30}
BATCH_EVERY=${BATCH_EVERY:-10}
ARMS=${ARMS:-"no_memory continual batched random"}

mkdir -p "$LOG_DIR"

COMMON="--dataset_path data-claude/tagged_dataset \
    --model_name nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --api_base_url http://127.0.0.1:8001/v1 \
    --iterations $ITERATIONS --num_samples 5 --team_size 4 \
    --seed 0 --split_seed 0 --test_fraction 0.2 --test_batch_size 5 --eval_workers 5"

run_arm() {
    name=$1
    shift
    echo "=== $name starting $(date -u +%FT%TZ) ==="
    # shellcheck disable=SC2086
    $PYTHON -u src/train_orchestrator.py $COMMON "$@" --out_dir "$OUT_ROOT/$name" \
        > "$LOG_DIR/orch_$name.log" 2>&1
    echo "=== $name finished $(date -u +%FT%TZ) with status $? ==="
}

for arm in $ARMS; do
    case $arm in
        no_memory) run_arm no_memory --memory none --summary_every 1 ;;
        continual) run_arm continual --summary_every 1 ;;
        batched)   run_arm batched --summary_every "$BATCH_EVERY" ;;
        # The random arm never reads a scoreboard, so its schedule is moot; it
        # still writes one, which keeps its outputs readable by report_run.py.
        random)    run_arm random --selection random --memory none --summary_every 1 ;;
        *)         echo "unknown arm: $arm" ;;
    esac
done

echo "ALL ARMS DONE $(date -u +%FT%TZ)"
