#!/usr/bin/env bash
# X1 stage 2: serve the hub (Qwen3.6-35B, thinking off) and run it over stage 1's saved answers.
#
# The hub also judges: stage 1's BrowseComp-Plus and Finance-Agent episodes are graded here.
# Each fold and sample is run with the headline thinking modes first, then with every specialist
# on and every specialist off (MODES), for the thinking-mode comparisons. Finished questions are
# skipped, so a rerun resumes.
#
#   ./scripts/x1_stage2.sh
#   FOLDS=0 SAMPLES=0 MODES=headline ./scripts/x1_stage2.sh
set -u
cd "$(dirname "$0")/.." || exit 1

PYTHON=${PYTHON:-./env/bin/python}
ROOT=${ROOT:-data-claude/x1-teams}
LOG_DIR=${LOG_DIR:-$ROOT/logs/stage2}
PORT=${PORT:-8001}
API=http://127.0.0.1:$PORT/v1
NAME=${NAME:-x1-serve}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SAMPLES=${SAMPLES:-"0 1 2"}
MODES=${MODES:-"headline on off"}
mkdir -p "$LOG_DIR"

handle=$($PYTHON -c "import json; print(json.load(open('configs/x1_teams.json'))['hub']['handle'])")
NAME=$NAME PORT=$PORT PROFILE=qwen35b HANDLE=$handle QUANT= THINKING=0 bash scripts/x1_serve.sh > /dev/null
for _ in $(seq 1 180); do curl -sf "$API/models" | grep -q "$handle" && break; sleep 10; done
curl -sf "$API/models" | grep -q "$handle" || { echo "hub server did not come up"; exit 1; }

for fold in $FOLDS; do
    for sample in $SAMPLES; do
        for modes in $MODES; do
            echo "=== fold $fold sample $sample modes $modes $(date -u +%FT%TZ)"
            $PYTHON -u scripts/x1_hub.py --root "$ROOT" --api "$API" --fold "$fold" --sample "$sample" --modes "$modes" \
                > "$LOG_DIR/fold${fold}_s${sample}_${modes}.log" 2>&1
            echo "status $?"; grep -E "^T[0-9]+: hub" "$LOG_DIR/fold${fold}_s${sample}_${modes}.log"
        done
    done
done
docker logs "$NAME" > "$LOG_DIR/server.log" 2>&1
docker rm -f "$NAME" > /dev/null
echo "STAGE2 DONE $(date -u +%FT%TZ)"
