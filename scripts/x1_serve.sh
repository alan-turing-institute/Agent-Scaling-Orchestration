#!/bin/bash
# Serve one X1 model on vLLM 0.21 (nvcr.io/nvidia/vllm:26.05.post1-py3), as the 2026-10-22
# smoke tests on the second Spark did. Flags follow ../vllm's qwen3.5-4b, qwen3.6 NVFP4,
# gemma4-e4b and ministral3-3b scripts, plus --reasoning-config so thinking_token_budget works.
# The model must already be in the HF cache (HF_HUB_OFFLINE=1). No MTP: it breaks the
# thinking budget (docs/x1-plan.md section 7). QUANT=fp8 quantises BF16 weights at load.
#
# Profiles: qwen4b | qwen35b | gemma | ministral.  Usage:
#   PROFILE=qwen4b HANDLE=wflying/Qwen3.5-4B-RL-MATH THINKING=0 bash scripts/x1_serve.sh
set -e
PROFILE=${PROFILE:?}
HANDLE=${HANDLE:?}
NAME=${NAME:-x1-serve}
PORT=${PORT:-8001}
MAXLEN=${MAXLEN:-131072}
THINKING=${THINKING:-1}
[ "$THINKING" = "1" ] && TKW='{"enable_thinking": true}' || TKW='{"enable_thinking": false}'

case "$PROFILE" in
qwen4b)
	UTIL=${UTIL:-0.5}
	QUANT=${QUANT-fp8}
	FLAGS="--kv-cache-dtype fp8 --language-model-only --enable-auto-tool-choice --tool-call-parser ${TOOLP:-qwen3_xml}
	       --reasoning-parser qwen3 --max-num-seqs 16"
	RCFG='{"reasoning_start_str": "<think>", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.</think>"}'
	;;
qwen35b)
	UTIL=${UTIL:-0.8}
	QUANT=${QUANT-}
	# BF16 KV: community NVFP4 builds ship no calibrated FP8 KV scales (TheShed card).
	FLAGS="--kv-cache-dtype auto --language-model-only --enable-auto-tool-choice --tool-call-parser ${TOOLP:-qwen3_xml}
	       --reasoning-parser qwen3 --max-num-seqs 8"
	RCFG='{"reasoning_start_str": "<think>", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.</think>"}'
	;;
gemma)
	UTIL=${UTIL:-0.5}
	QUANT=${QUANT-fp8}
	# Eager: with CUDA graphs Gemma hangs vLLM 0.21 within minutes on thinking-on requests (GPU busy,
	# 0 tokens/s; 5 of 5 attempts on 2026-10-29); eager ran the same 55 questions without a stall.
	FLAGS="--kv-cache-dtype fp8 --language-model-only --enable-auto-tool-choice --tool-call-parser gemma4
	       --reasoning-parser gemma4 --max-num-seqs 16 --enforce-eager"
	RCFG='{"reasoning_start_str": "<|channel>", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.<channel|>"}'
	;;
ministral)
	UTIL=${UTIL:-0.5}
	QUANT=${QUANT-}
	FLAGS="--tokenizer-mode mistral --config-format mistral --load-format mistral --kv-cache-dtype fp8
	       --language-model-only --enable-auto-tool-choice --tool-call-parser mistral --reasoning-parser mistral
	       --max-num-seqs 16"
	RCFG='{"reasoning_start_str": "[THINK]", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.[/THINK]"}'
	;;
*) echo "unknown PROFILE $PROFILE"; exit 1 ;;
esac
[ -n "$NO_RCFG" ] && RCFG=""
QUANTIZATION=""
[ -n "$QUANT" ] && QUANTIZATION="--quantization ${QUANT}"
TKW_FLAG=(--default-chat-template-kwargs "$TKW")
[ "$PROFILE" = "ministral" ] && TKW_FLAG=()
RCFG_FLAG=()
[ -n "$RCFG" ] && RCFG_FLAG=(--reasoning-config "$RCFG")

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --gpus all -p ${PORT}:8000 --name "$NAME" \
	-e HF_HUB_OFFLINE=1 \
	-v ~/.cache/huggingface:/root/.cache/huggingface \
	nvcr.io/nvidia/vllm:26.05.post1-py3 \
	vllm serve ${HANDLE} \
	${QUANTIZATION} \
	--max-model-len ${MAXLEN} \
	--max-num-batched-tokens 8192 \
	--gpu-memory-utilization ${UTIL} \
	--enable-prefix-caching \
	${FLAGS} \
	"${RCFG_FLAG[@]}" \
	"${TKW_FLAG[@]}" \
	${EXTRA_ARGS}
