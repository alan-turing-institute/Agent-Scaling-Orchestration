#!/usr/bin/env bash
# The orchestrator experiment with small agents: a small model answers, and
# (for the orchestrator arms) the 35B-A3B chooses the team from a second server.
#
# Why: with the 35B as agent and orchestrator, every multi-agent arm landed
# above 90% (first round: random 92.9%, bare 71.4%), too close to the ceiling to
# separate the arms. Small agents leave room for team choice to matter.
#
# Arms, cheapest first:
#   bare_model  the agent model alone, no persona, no team
#   random      teams of four drawn uniformly from the pool; no orchestrator call
#   no_memory   the orchestrator picks each held-out batch's team from its tags
#               alone. --iterations 0: with --memory none the held-out selections
#               never read the scoreboard, so training would change nothing there
#   continual   30 training iterations, scoreboard rewritten after every one and
#               read back before every selection, then frozen for the holdout
#   canonical   no selection at all: each question goes to the paper's persona set
#               for its source dataset, cut to the first TEAM_SIZE personas
#               (scripts/paper_persona_baseline.py). It batches by source dataset,
#               so it cannot be paired batch-for-batch with the other arms.
# Only continual trains; the others only ever see the test set.
#
# TEAM_SIZE (default 4) is the number of agents per team. TEAM_SIZE=1 removes the
# vote, so the run measures the choice of agent directly; its results go to a
# separate root (crossval-small-team1, small-models-team1) so they never mix.
#
# Two split schemes, both shared with the 35B runs so the numbers line up:
#   FOLDS unset  the first round's single 20% holdout (split_seed 0, 140
#                questions), written to data-claude/small-models/<label>/<arm>
#   FOLDS="0 1 2 3 4"  the 5-fold cross validation of scripts/crossval.sh
#                (stratified by source, split_seed 0, 140 per fold), written to
#                data-claude/crossval-small/<label>/fold<k>/<arm>. The outer loop
#                is the fold, so every finished fold is a full replicate.
#
# Serve the agents on 8001 (../vllm/<model>/run_docker.sh) and, for the
# orchestrator arms, the 35B on 8002 (HOST_PORT=8002 sh ../vllm/qwen3.6/run_docker_nvfp4.sh), then:
#
#   MODEL=Qwen/Qwen3.5-0.8B LABEL=qwen3.5-0.8b ./scripts/small_model_arms.sh
#   MODEL=mistralai/Ministral-3-3B-Instruct-2512 LABEL=ministral3-3b FOLDS="0 1 2 3 4" ./scripts/small_model_arms.sh
#
# DATASET (default data-claude/tagged_dataset) picks the question pool; any other
# pool, e.g. DATASET=data-claude/tagged_hard from scripts/build_pool.sh, adds its
# name to the output root (crossval-small-hard/...) so pools never mix. The
# canonical arm needs the paper's persona sets, so it refuses pools whose
# benchmarks have none. MAX_TOKENS (default 4096, what every run so far used)
# is the agents' generation budget; long-solution sets such as AIME want more.
#
# Finished arms are skipped, so a stopped sweep resumes where it left off.
set -u

cd "$(dirname "$0")/.." || exit 1

PYTHON=${PYTHON:-./env/bin/python}
MODEL=${MODEL:?set MODEL to the served agent model id}
LABEL=${LABEL:?set LABEL, used as the output directory name}
API=${API:-http://127.0.0.1:8001/v1}
ORCH_MODEL=${ORCH_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}
ORCH_API=${ORCH_API:-http://127.0.0.1:8002/v1}
ITERATIONS=${ITERATIONS:-30}
ARMS=${ARMS:-"bare_model random no_memory continual"}
TEAM_SIZE=${TEAM_SIZE:-4}
DATASET=${DATASET:-data-claude/tagged_dataset}
MAX_TOKENS=${MAX_TOKENS:-4096}
SUFFIX=""
[ "$TEAM_SIZE" != "4" ] && SUFFIX="-team$TEAM_SIZE"
POOL=$(basename "$DATASET"); POOL=${POOL#tagged_}
[ "$POOL" != "dataset" ] && SUFFIX="$SUFFIX-$POOL"
FOLDS=${FOLDS:-}
N_FOLDS=${N_FOLDS:-5}
if [ -n "$FOLDS" ]; then
    OUT_ROOT=${OUT_ROOT:-data-claude/crossval-small$SUFFIX}
    LOG_DIR=${LOG_DIR:-data-claude/logs/crossval-small$SUFFIX}
else
    OUT_ROOT=${OUT_ROOT:-data-claude/small-models$SUFFIX}
    LOG_DIR=${LOG_DIR:-data-claude/logs/small-models$SUFFIX}
fi
mkdir -p "$LOG_DIR"

wait_for() {  # url model
    until curl -s -m 60 "$1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}" \
        | grep -q '"choices"'; do
        echo "$(date -u +%FT%TZ) $2 at $1 not answering; waiting"
        sleep 60
    done
}

run_arm() {  # arm split out log
    arm=$1; split=$2; out=$3; log=$4
    if [ -f "$out/holdout_summary.json" ]; then
        echo "=== $out already done, skipping ==="
        return
    fi
    wait_for "$API" "$MODEL"
    case $arm in no_memory|continual) wait_for "$ORCH_API" "$ORCH_MODEL" ;; esac
    echo "=== $out starting $(date -u +%FT%TZ) ==="
    loop="--dataset_path $DATASET --model_name $MODEL --api_base_url $API $split \
        --num_samples 5 --team_size $TEAM_SIZE --seed 0 --test_batch_size 5 --eval_workers 5 \
        --max_new_tokens $MAX_TOKENS"
    orch="--orchestrator_model $ORCH_MODEL --orchestrator_api_base_url $ORCH_API"
    # shellcheck disable=SC2086
    case $arm in
        bare_model)
            $PYTHON -u scripts/bare_model_baseline.py --dataset_path "$DATASET" \
                --model_name "$MODEL" --api_base_url "$API" $split --max_tokens "$MAX_TOKENS" \
                --out_dir "$out" > "$log" 2>&1 ;;
        random)
            $PYTHON -u src/train_orchestrator.py $loop --iterations 0 \
                --selection random --memory none --summary_every 1 \
                --out_dir "$out" > "$log" 2>&1 ;;
        no_memory)
            $PYTHON -u src/train_orchestrator.py $loop $orch --iterations 0 \
                --memory none --summary_every 1 \
                --out_dir "$out" > "$log" 2>&1 ;;
        continual)
            $PYTHON -u src/train_orchestrator.py $loop $orch --iterations "$ITERATIONS" \
                --summary_every 1 \
                --out_dir "$out" > "$log" 2>&1 ;;
        canonical)
            $PYTHON -u scripts/paper_persona_baseline.py --dataset_path "$DATASET" \
                --model_name "$MODEL" --api_base_url "$API" $split \
                --team_size "$TEAM_SIZE" --test_batch_size 5 --eval_workers 5 \
                --out_dir "$out" > "$log" 2>&1 ;;
        *)
            echo "unknown arm: $arm"; return ;;
    esac
    echo "=== $out finished $(date -u +%FT%TZ) with status $? ==="
    grep -E -A2 "BARE MODEL|HELD-OUT RESULT" "$log" | grep -v "^=*$" | head -3
}

if [ -z "$FOLDS" ]; then
    for arm in $ARMS; do
        run_arm "$arm" "--split_seed 0 --test_fraction 0.2" \
            "$OUT_ROOT/$LABEL/$arm" "$LOG_DIR/${LABEL}_${arm}.log"
    done
    exit 0
fi

for fold in $FOLDS; do
    echo "##### $LABEL fold $fold of $N_FOLDS starting $(date -u +%FT%TZ) #####"
    for arm in $ARMS; do
        run_arm "$arm" "--split_seed 0 --n_folds $N_FOLDS --fold $fold" \
            "$OUT_ROOT/$LABEL/fold$fold/$arm" "$LOG_DIR/${LABEL}_fold${fold}_${arm}.log"
    done
    echo "##### $LABEL fold $fold complete $(date -u +%FT%TZ) #####"
    $PYTHON scripts/report_crossval.py "$OUT_ROOT/$LABEL" || true
done
