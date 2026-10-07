#!/usr/bin/env bash
# Build a tagged question pool from registered benchmarks, sharing the existing
# tag vocabulary.
#
#   1. tag_questions.py asks the model for capability tags per question.
#   2. canonicalise_tags.py --extend grows data-claude/tag_mapping.json to cover
#      the new tags without changing any existing entry, so the 699-question pool
#      keeps its tags exactly and the new pool reuses them where they overlap.
#   3. tag_dataset.py applies the mapping, drops rare tags and saves the pool.
#
#   POOL=hard DATA="gpqa_diamond mmlu_pro aime" ./scripts/build_pool.sh
#
# Writes data-claude/question_tags/<POOL>_tags.jsonl,
# data-claude/tag_mapping_<POOL>.json and data-claude/tagged_<POOL>/.
# Point the experiment drivers at the pool with DATASET=data-claude/tagged_<POOL>.
# Tagging needs the model server, so do not run it while a sweep is using it.
set -eu

cd "$(dirname "$0")/.." || exit 1

PYTHON=${PYTHON:-./env/bin/python}
POOL=${POOL:?set POOL, the pool name, e.g. hard}
DATA=${DATA:?set DATA, the space-separated benchmark names}
MODEL=${MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}
API=${API:-http://127.0.0.1:8002/v1}
BASE_MAPPING=${BASE_MAPPING:-data-claude/tag_mapping.json}
DATA_DIR=${DATA_DIR:-data-claude/benchmarks}
THRESHOLD=${THRESHOLD:-5}
DATA_SIZE=${DATA_SIZE:-0}   # questions per benchmark; 0 = each loader's full set

TAGS=data-claude/question_tags/${POOL}_tags.jsonl
MAPPING=data-claude/tag_mapping_${POOL}.json
OUT=data-claude/tagged_${POOL}

echo "=== 1/3 tagging: $DATA ==="
# shellcheck disable=SC2086
$PYTHON src/tag_questions.py --use_vllm --vllm_base_url "$API" --model "$MODEL" \
    --data $DATA --data_dir "$DATA_DIR" --split test --data_size "$DATA_SIZE" \
    --out_dir data-claude/question_tags --output_file "${POOL}_tags.jsonl" > "data-claude/question_tags/${POOL}_tags.log" 2>&1
echo "    $(wc -l < "$TAGS") questions tagged"

echo "=== 2/3 extending the tag vocabulary from $BASE_MAPPING ==="
$PYTHON src/canonicalise_tags.py --tags_file "$TAGS" --extend "$BASE_MAPPING" \
    --out_file "$MAPPING" --model "$MODEL" --api_base_url "$API" | tail -4

echo "=== 3/3 saving the pool ==="
$PYTHON src/tag_dataset.py --tags_file "$TAGS" --tag_mapping "$MAPPING" --threshold "$THRESHOLD" \
    --out_dir "$OUT" --plot_path "data-claude/tag_frequencies_${POOL}.png" | grep -E "Total number|Unique tags|^  - [a-z_]+: [0-9]+$" | head -12
echo "pool written to $OUT"
