#!/usr/bin/env bash
# Cross validation over the whole experiment: every arm, every fold, one at a time.
#
# The first round ran each arm once on a single held-out split of 140 questions.
# Every multi-agent arm landed between 90.0% and 93.6% there, a spread of five
# questions, with confidence intervals that overlap almost entirely - so nothing
# was separable and a repeat was the only way to tell an effect from noise.
#
# Folds are disjoint and stratified by source dataset (src/splits.py), so across a
# full sweep every one of the 699 questions is tested exactly once per arm, and
# each fold holds 20 questions from each of the seven sources.
#
# ARM ORDER AND FOLD ORDER. The outer loop is the fold, not the arm, so each
# completed fold is a full replicate of the experiment. Stopping after any fold
# leaves a usable result rather than a few arms measured more often than others.
#
# NOTHING RUNS CONCURRENTLY. One vLLM server, one run at a time. Three separate
# engine wedges have already been traced to concurrent load, and a fourth cost a
# whole held-out split. The watchdog restarts a wedged engine; the loops wait out
# the restart.
#
#   ./scripts/crossval.sh                 # 5 folds, every arm, about 3 days
#   FOLDS="0 1" ./scripts/crossval.sh     # the first two folds only
#   ARMS="random bare_model" ./scripts/crossval.sh
set -u

cd "$(dirname "$0")/.." || exit 1

PYTHON=${PYTHON:-./env/bin/python}
OUT_ROOT=${OUT_ROOT:-data-claude/crossval}
LOG_DIR=${LOG_DIR:-data-claude/logs/crossval}
ITERATIONS=${ITERATIONS:-30}
BATCH_EVERY=${BATCH_EVERY:-10}
N_FOLDS=${N_FOLDS:-5}
FOLDS=${FOLDS:-"0 1 2 3 4"}
ARMS=${ARMS:-"bare_model paper_personas paper_personas_4 random no_memory continual batched"}
MODEL=${MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}
API=${API:-http://127.0.0.1:8001/v1}

mkdir -p "$LOG_DIR"

# Cheapest arms first within a fold, so a fold that is interrupted still leaves the
# baselines the expensive arms are measured against.
orchestrator_common() {
    echo "--dataset_path data-claude/tagged_dataset \
        --model_name $MODEL --api_base_url $API \
        --iterations $ITERATIONS --num_samples 5 --team_size 4 \
        --seed 0 --split_seed 0 --n_folds $N_FOLDS --fold $1 \
        --test_batch_size 5 --eval_workers 5"
}

baseline_common() {
    echo "--dataset_path data-claude/tagged_dataset \
        --model_name $MODEL --api_base_url $API \
        --split_seed 0 --n_folds $N_FOLDS --fold $1"
}

wait_for_server() {
    until curl -s -m 60 "$API/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}" \
        | grep -q '"choices"'; do
        echo "$(date -u +%FT%TZ) server not answering; waiting"
        sleep 60
    done
}

run_arm() {
    fold=$1
    arm=$2
    out="$OUT_ROOT/fold$fold/$arm"
    log="$LOG_DIR/fold${fold}_${arm}.log"

    if [ -f "$out/holdout_summary.json" ]; then
        echo "=== fold $fold $arm already done, skipping ==="
        return
    fi

    wait_for_server
    echo "=== fold $fold $arm starting $(date -u +%FT%TZ) ==="
    # shellcheck disable=SC2086
    case $arm in
        bare_model)
            $PYTHON -u scripts/bare_model_baseline.py $(baseline_common "$fold") \
                --out_dir "$out" > "$log" 2>&1 ;;
        paper_personas)
            $PYTHON -u scripts/paper_persona_baseline.py $(baseline_common "$fold") \
                --team_size 0 --out_dir "$out" > "$log" 2>&1 ;;
        paper_personas_4)
            $PYTHON -u scripts/paper_persona_baseline.py $(baseline_common "$fold") \
                --team_size 4 --out_dir "$out" > "$log" 2>&1 ;;
        random)
            $PYTHON -u src/train_orchestrator.py $(orchestrator_common "$fold") \
                --selection random --memory none --summary_every 1 --out_dir "$out" > "$log" 2>&1 ;;
        no_memory)
            $PYTHON -u src/train_orchestrator.py $(orchestrator_common "$fold") \
                --memory none --summary_every 1 --out_dir "$out" > "$log" 2>&1 ;;
        continual)
            $PYTHON -u src/train_orchestrator.py $(orchestrator_common "$fold") \
                --summary_every 1 --out_dir "$out" > "$log" 2>&1 ;;
        batched)
            $PYTHON -u src/train_orchestrator.py $(orchestrator_common "$fold") \
                --summary_every "$BATCH_EVERY" --out_dir "$out" > "$log" 2>&1 ;;
        *)
            echo "unknown arm: $arm"; return ;;
    esac
    echo "=== fold $fold $arm finished $(date -u +%FT%TZ) with status $? ==="
}

for fold in $FOLDS; do
    echo "##### FOLD $fold of $N_FOLDS starting $(date -u +%FT%TZ) #####"
    for arm in $ARMS; do
        run_arm "$fold" "$arm"
    done
    echo "##### FOLD $fold complete $(date -u +%FT%TZ) #####"
    $PYTHON scripts/report_crossval.py "$OUT_ROOT" || true
done

echo "CROSSVAL DONE $(date -u +%FT%TZ)"
