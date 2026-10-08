#!/usr/bin/env bash
# End-to-end check of a question pool against the stand-in server, no real model.
#
# Starts tests/mock_openai_server.py on a spare port, builds a small pool from
# the benchmarks named in DATA with scripts/build_pool.sh, runs every arm of
# scripts/small_model_arms.sh on one fold, and checks each arm wrote its summary
# and one prediction row per held-out question. Everything is written under a
# throwaway pool name and removed afterwards, unless KEEP=1.
#
#   DATA="gpqa_diamond mmlu_pro aime" tests/mock_e2e.sh
#
# Never points at :8001/:8002: a sweep's answers change with any extra load.
# Needs the benchmarks' data already downloaded (scripts/fetch_benchmarks.py).
set -eu

cd "$(dirname "$0")/.." || exit 1

PYTHON=${PYTHON:-./env/bin/python}
DATA=${DATA:?set DATA, the space-separated benchmark names}
PORT=${PORT:-18011}
DATA_SIZE=${DATA_SIZE:-20}
POOL=${POOL:-mocke2e}
ARMS=${ARMS:-"bare_model random no_memory continual"}
case $PORT in 8000|8001|8002) echo "port $PORT is a real model server's"; exit 1 ;; esac

WORK=$(mktemp -d)
API=http://127.0.0.1:$PORT/v1
# MOCK_ARGS passes options to the server, e.g. MOCK_ARGS="--context_chars 30000".
# shellcheck disable=SC2086
$PYTHON tests/mock_openai_server.py --port "$PORT" --log "$WORK/requests.jsonl" ${MOCK_ARGS:-} > "$WORK/server.log" 2>&1 &
SERVER=$!
cleanup() {
    kill "$SERVER" 2>/dev/null || true
    if [ "${KEEP:-0}" != "1" ]; then
        rm -rf "data-claude/tagged_$POOL" "data-claude/tag_mapping_$POOL.json" \
            "data-claude/question_tags/${POOL}_tags.jsonl" "data-claude/question_tags/${POOL}_tags.log" \
            "data-claude/tag_frequencies_$POOL.png" "$WORK"
    else
        echo "kept: data-claude/tagged_$POOL and $WORK"
    fi
}
trap cleanup EXIT
until curl -s -m 2 "$API/models" | grep -q mock; do sleep 0.5; done

echo "### build pool $POOL from: $DATA ($DATA_SIZE each)"
POOL=$POOL DATA=$DATA DATA_SIZE=$DATA_SIZE MODEL=mock API=$API ./scripts/build_pool.sh | tail -3

echo "### arms: $ARMS"
MODEL=mock LABEL=mock API=$API ORCH_MODEL=mock ORCH_API=$API DATASET="data-claude/tagged_$POOL" \
    FOLDS=0 ITERATIONS=2 ARMS="$ARMS" OUT_ROOT="$WORK/out" LOG_DIR="$WORK/logs" \
    ./scripts/small_model_arms.sh > "$WORK/arms.log" 2>&1 || { tail -30 "$WORK/arms.log"; exit 1; }

$PYTHON - "$WORK/out/mock/fold0" "data-claude/tagged_$POOL" $ARMS <<'EOF'
import json, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, "src")
from datasets import load_from_disk
from splits import make_split

root, pool, arms = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
_, test = make_split(load_from_disk(pool), SimpleNamespace(n_folds=5, fold=0, split_seed=0, test_fraction=0.2))
problems = []
for arm in arms:
    summary = root / arm / "holdout_summary.json"
    preds = root / arm / "holdout_predictions.jsonl"
    if not summary.exists() or not preds.exists():
        problems.append(f"{arm}: missing {summary.name} or {preds.name}")
        continue
    rows = [json.loads(line) for line in preds.open()]
    if len(rows) != len(test):
        problems.append(f"{arm}: {len(rows)} prediction rows for {len(test)} held-out questions")
    unparsed = sum(1 for r in rows for s in (r.get("stages") or []) if not s.get("parsed"))
    limits = json.loads(summary.read_text())
    print(f"  {arm:11s} {len(rows)} rows, {unparsed} unparsed stage answers, "
          f"{limits.get('context_limited_questions', 0)} hit the context limit, "
          f"{limits.get('max_tokens_limited_questions', 0)} cut off by max_tokens")
if problems:
    print("FAILED:\n  " + "\n  ".join(problems))
    raise SystemExit(1)
print(f"ok - {len(arms)} arms, {len(test)} held-out questions each")
EOF
echo "requests served: $(wc -l < "$WORK/requests.jsonl")"
