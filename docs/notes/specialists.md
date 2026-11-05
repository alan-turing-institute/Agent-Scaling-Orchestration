# Specialist checkpoints: how they were trained and how to serve them (2026-10-19)

Sources: HF model cards, config.json, the safetensors index, tokenizer and generation configs, the HF discussions, the Ministral 3 paper (arXiv 2601.08584), the BigBang paper (endlessfrontier.tech/assets/paper.pdf), and the vLLM registry inside our image. Download counts are for the last 30 days, with the all-time figure where HF gives one.

**Local stack.** Every small-model script uses `nvcr.io/nvidia/vllm:26.05.post1-py3`, whose image label says **vLLM 0.21.0** (transformers 5.6.0, mistral_common 1.11.2). The registry in 0.21 has `Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration` but **not `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM`**. Upstream added those in **v0.27.0**; the current release is v0.32.0. So any *text-only* Qwen3.5/3.6 checkpoint will not load on our image as shipped.

---

## Part 1: the four chosen specialists

### 1. wflying/Qwen3.5-4B-RL-MATH
| | |
|---|---|
| Base | `Qwen/Qwen3.5-4B` (the instruct/thinking model we already serve) |
| Method | GRPO-style RLVR with a rule-verified maths reward. Full weights (BF16, 4.66B params including the vision tower), not an adapter. Trained in **non-thinking mode** (`enable_thinking=False`) |
| Data | `wflying/math-rlvr-15k`: 15k prompts with `\boxed{}` answers, taken from `allenai/RLVR-MATH` (MATH *train*) and the maths part of `allenai/Dolci-RL-Zero-Mix-7B`. MATH-500 is drawn from the test split, so no direct overlap. Dolci provenance is not itemised |
| Who / licence | Individual HF user `wflying` (who also has 9B and 35B RL variants); Apache-2.0 |
| Config | `Qwen3_5ForConditionalGeneration` (multimodal), BF16. Ships the vision and **MTP** tensors (738 tensors, same layout as base). Chat template is byte-identical to the base (md5 match). No generation_config.json |
| Claimed evals (non-thinking, harness and sampling not stated) | Base → RL: AIME24 58.3→**74.4**, AIME25 38.3→**58.5**, HMMT25 30.0→**50.4**, GPQA-D 71.2→**76.9**, ZebraLogic 82.6→89.8, LCB-v6 52.1→62.7 |
| Reproduced? | No. 0 likes, 152 dl (194 all-time), no discussions |
| Caveat | Qwen's official *thinking-mode* Qwen3.5-4B already scores GPQA 76.2 and HMMT-Feb25 74.0. The gain probably comes from RL teaching the model to reason at length inside the answer, so expect **much longer non-thinking outputs**, more often hitting the 4096-token cap. The MTP head was presumably not updated, so speculative acceptance may drop (it is still lossless) |

**vLLM plan.** Swap the handle into the existing script, which already uses thinking off, FP8, MTP k=3 and `--reasoning-config`:
`HF_MODEL_HANDLE=wflying/Qwen3.5-4B-RL-MATH CONTAINER_NAME=qwen-rlmath HOST_PORT=8003 sh vllm/qwen3.5-4b/run_docker.sh`. The download is ~9.3 GB and the memory share is the same as the 4B (0.4). Pitfalls:
- FP8 is applied at load time, the same as our base 4B, so the comparison is fair.
- Watch the MTP acceptance rate in the logs. Use `NO_SPEC=1` if it collapses.
- Raise the client `max_tokens`.

### 2. mistralai/Ministral-3-3B-Reasoning-2512
| | |
|---|---|
| Base | `mistralai/Ministral-3-3B-Base-2512`, the long-context pretrained checkpoint. It is a **sibling** of the Instruct-2512 we serve, not a fine-tune of it |
| Method (paper §3.3) | Full post-training in three stages: (1) reasoning SFT on short and long CoT, with **logit distillation from Magistral Small 1.2** for the 3B (plain SFT made it verbose and loop forever); (2) GRPO, first STEM RL (maths, code, visual) and then general RL with an LLM judge and rubrics, generation length 32K→80K; (3) ODPO. The parent model was itself made by "Cascade Distillation" (prune + distil) from Mistral Small 3.1 |
| Data | Proprietary plus open, not itemised |
| Who / licence | Mistral AI; Apache-2.0 |
| Config | `Mistral3ForConditionalGeneration` (3.4B LM + 0.4B Pixtral vision), **BF16**. The Instruct sibling ships as FP8. Also ships mistral-format `consolidated.safetensors`/`params.json`/`tekken.json` |
| Sampling / template | temp **0.7**, top_p 0.95. **Always thinks**; there is no off switch. Thinking is triggered by a **mandatory system prompt** (SYSTEM_PROMPT.txt) containing a `[THINK]…[/THINK]` template. The HF jinja injects it only when *no* system message is present. Mistral say to *append* it to custom system prompts and keep traces across turns |
| Claimed evals | AIME25 72.1, AIME24 77.5, GPQA-D 53.4, LCB 54.8; paper Table 5 adds HMMT25 51.7 and PhyBench 15.0 (reported as "pass@16", which is ambiguous). Card comparison: Qwen3-VL-4B-Thinking scores 69.7 / 72.9 / 60.1 |
| Reproduced? | No independent numbers found. 16,379 dl (514k all-time), 135 likes; the discussions raise no technical issues |

**vLLM plan.** Reuse `vllm/ministral3-3b/run_docker.sh` (mistral tokenizer/config/load formats, `--language-model-only`, mistral tool parser) and **add** `--reasoning-parser mistral`. The script has no `EXTRA_ARGS`, so this needs a new sibling script, which still has to be written. To match the FP8 Instruct, add `--quantization fp8`. Online FP8 together with `--load-format mistral` is untested here; if it fails, serve BF16 (~7.7 GB, still small). Pitfalls:
- **System prompt**: our agents send their own system prompts, so the thinking prompt is not injected. We must prepend SYSTEM_PROMPT.txt ourselves. Sent as a plain string, `[THINK]` may tokenize as text rather than the control token; Mistral's example sends it as a `{"type":"thinking"}` content part.
- **Thinking budget**: our `--reasoning-config` uses `<think>` strings, and the Mistral equivalents are `[THINK]`/`[/THINK]` control tokens (untested). Per CLAUDE.md, `THINKING_TOKEN_BUDGET` must not reach a server without `--reasoning-config`. Because it is a *global* env var in the main repo, a mixed team that includes this server breaks unless the budget becomes per-URL.
- Long reasoning (trained to 80K) against our 32K/4096 caps means truncation risk.
- `--language-model-only` with mistral format worked for Instruct, so it is expected to work here.

### 3. endless-frontier/BigBang-v1 (+ the NVFP4 build `TheShed/BigBang-v1-W4A16-NVFP4-MTP`)
| | |
|---|---|
| Base | `Qwen/Qwen3.6-35B-A3B` |
| Method | **SFT** (the paper figure reads "35B-A3B · SFT"). Full weights released (BF16, 35.95B), so either full fine-tuning or a merge; hyperparameters not given. Data comes from an adversarial generator/critic agent loop of "verifiable frontier tasks", calibrated on held-out real research tasks |
| Data | About **10,000** synthetic, long-horizon *agentic* trajectories (search/visit/code_exec, up to 500 tool calls, 256K context). Not released; issue #8 asks for it |
| Who / licence | "The Endless Frontier" / BigBang Team, led by **Siheng Chen and Linfeng Zhang (SJTU)**, with Weinan E among the advisors. Apache-2.0. The paper is self-hosted, not on arXiv |
| Config | `Qwen3_5MoeForConditionalGeneration` (multimodal), BF16, MTP 1 layer. Chat template identical to Qwen3.6. generation_config: temp 1.0, top_k 20, top_p 0.95, eos [<\|im_end\|>, <\|endoftext\|>]. Recommends **≥128K context** "to preserve thinking" |
| Claimed evals (all agentic, with tools) | Base → BigBang: BrowseComp 67.9→76.5, SWE-Bench Pro 43.6→54.2, FrontierScience-R 11.9→46.2, **HLE 36.2→50.3 (with search+code tools)**, MLE-Bench-Lite 31.8→59.1. **No MATH/GSM8K/GPQA/MMLU-Pro numbers.** The paper admits there are no per-component ablations |
| Reproduced? | No. Issue #3 asks for Artificial Analysis listing. 2,221 dl (20k all-time), 233 likes. HF #4 reports infinite loops in an agent harness (Q8 GGUF; attributed to bad sampling). #6 reports MTP acceptance ~69% and prefix-cache hit 0% on SM120 |
| NVFP4 build | `TheShed/BigBang-v1-W4A16-NVFP4-MTP` (individual uploader; 295 dl / 1,223 all-time). ModelOpt 0.45 MIXED_PRECISION: MoE and shared experts and lm_head in W4A16 NVFP4 (g16), attention in FP8, vision and MTP in BF16. That is the **same recipe as `nvidia/Qwen3.6-35B-A3B-NVFP4`** (161 NVFP4 + 130 FP8 layer entries in both). 21.8 GiB. Tested on vLLM **0.25.1** on a DGX Spark-class box with Marlin MoE, **BF16 KV** (the config advertises FP8 KV but has **no calibrated KV scales**), and the qwen3 reasoning/tool parsers. "No formal quality comparison vs BF16." A community member has posted FP8-KV scales (discussion #1) |

**vLLM plan.** `HF_MODEL_HANDLE=TheShed/BigBang-v1-W4A16-NVFP4-MTP sh vllm/qwen3.6/run_docker_nvfp4.sh` gives `qwen3_5_mtp` k=1, auto→Marlin, and `qwen3_xml`/`qwen3`. Pitfalls:
- The script hard-codes `--kv-cache-dtype fp8`, which with uncalibrated scales means a quality risk. A variant using `auto`/bf16 KV is needed.
- Our image is 0.21 against their 0.25.1. It should still be fine, because the nvidia checkpoint with the identical structure runs on 0.21.
- **Memory**: ~23 GB of weights at util 0.45, the same as `qwen3.6-orch`, so one of the two must go, or both drop to ~0.4 with fewer sequences.
- Neither the script nor `qwen3.6-orch` sets `--reasoning-config`, so the global thinking budget cannot be used, and Qwen3.6 thinks by default.
- Agents run at temperature 0, whereas the model is tuned for 1.0; greedy decoding raises the risk of the loop reports.
- The model is optimised for tool-using research, so gains on closed-book MCQ are unproven.

### 4. adi274903/Med-Qwen-3.5-4B-Instruct
| | |
|---|---|
| Base | **`Qwen/Qwen3.5-4B-Base`** (pretrained, *not* the instruct model we serve) |
| Method | (1) "multi-stream biomedical" continued pretraining (DAPT); (2) SFT with **DoRA r64/α128** for 2,700 steps, merged. 11-stream mixture of ~276M tokens (long-CoT medical, board vignettes, evidence grounding, STEM replay). No dataset names, no code, no paper |
| Who / licence | Individual user (one model, account active since 2026-10-14); Apache-2.0 |
| Config | **`Qwen3_5ForCausalLM`, `qwen3_5_text` (text-only)**, BF16, 4.21B. 426 tensors (`model.language_model.*`); **no MTP weights** even though `mtp_num_hidden_layers: 1`. Tokenizer `eos_token` and generation_config eos = **`<|endoftext|>` (248044) only**, whereas the instruct model uses `<|im_end|>`. Chat template = Qwen3.5 instruct template (thinking **on** by default) |
| Claimed evals (log-prob scoring) | vs its own Base: PubMedQA 45.6→**66.2**, MedQA 68.3→68.0, MedMCQA 26.5→33.2, GPQA-D 36.9→50.5, MMLU-med 82.9→80.7, AfriMed 67.0→64.8 |
| Credibility | **Low.** MedMCQA at 26–33% for every model is chance level for a 4-way task, and Gemma-4-E4B-it at 31.7% on MMLU-med is implausible, which together point to a broken harness. A careful paired evaluation of *our* base Qwen3.5-4B (helloimsaif/medjev card, n=300) gives MedQA 71.7, MedMCQA 67.3, PubMedQA 79.5, all above Med-Qwen's claims. Not reproduced; 980 dl, 0 likes, no discussions |

**vLLM plan.** It **will not load on vLLM 0.21** because the architecture is unregistered. There are two options:
- Use a ≥0.27 image. That means pulling a new image, which was not done.
- Build a local wrapper directory: Med-Qwen safetensors, plus `Qwen/Qwen3.5-4B`'s `config.json` (ConditionalGeneration; the weight prefix `model.language_model.` already matches its mapper), plus the base's `tokenizer_config.json` (eos `<|im_end|>`) and preprocessor jsons. Mount it, then serve with `--language-model-only` (vision weights absent; verify vLLM skips the tower) and **`NO_SPEC=1`** (no MTP weights).

Otherwise, generation will not stop at `<|im_end|>`; as a workaround, the client can send `stop_token_ids:[248046]`. It is unknown whether the SFT used `<think>` blocks, so smoke-test both `THINKING=0` and `THINKING=1`. Memory share is the same as the 4B (0.4; ~4.2 GB at FP8).

### Minimal smoke test (all four)
1. Check `docker logs` for: the resolved architecture, "Loading weights took", no missing/unexpected-key warnings, the MTP acceptance line, and the KV-cache size. Then `curl :PORT/v1/models`.
2. Use five fixed prompts from the cached datasets: GSM8K, MATH-500, GPQA-D, MMLU-Pro (health or law), TruthfulQA-MC. Send each to **the specialist and its comparator** under identical server flags: RL-MATH and Med-Qwen vs `Qwen/Qwen3.5-4B`; Ministral-R vs Ministral-Instruct; BigBang vs `nvidia/Qwen3.6-35B-A3B-NVFP4`. Use temp 0, the same `max_tokens`, and the same system prompt as the orchestrator.
3. Record correctness, `completion_tokens`, `finish_reason` (`length` means a cap problem), whether `reasoning_content` is populated, and any template leakage (`<|im_end|>`, `[THINK]` or `</think>` in `content`).
4. Pass criteria: 5/5 stop cleanly, answers parse, and accuracy is no worse than the comparator. Then run a ≥200-item paired run before admitting a model to the pool.

---

## Part 2: further candidates on the same five bases

There are few credible STEM or knowledge specialists on these bases. Most children are decision/classifier heads ("Jev"-style), agents, language ports, or quantisations. No org-backed maths, science, medical or law specialists exist for Qwen3.5-0.8B/4B, Gemma-4-E4B or Ministral-3-3B, except Mistral's own Reasoning model. OpenMed has nothing on these sizes, and MedGemma is still Gemma-3. Credibility is scored 1–3, where 3 means org-backed with paired evals.

| Repo | Base | Method / data | Claimed gains | Cred | Verdict |
|---|---|---|---|---|---|
| ValiantLabs/gemma-4-E4B-it-ShiningValiant3 | gemma-4-E4B-it | Full SFT (single safetensors) on DeepSeek-R1-0528 science-reasoning traces (Celestia3), AI-design (Mitakihara) and general (Raiden) data | None published (12 dl) | 2 | **Best GPQA/science candidate on Gemma.** An established series; evaluate it ourselves. Serve with the gemma4-e4b script plus the handle swap |
| Jackrong/Qwen3.5-4B-Neo | Qwen3.5-4B | LoRA SFT, merged (ConditionalGeneration). Data: Opus-4.6 distilled reasoning 3k, Qwen3.5-reasoning-700x, Nemotron competitive programming/SWE/IF | MMLU-Pro 250-item subset 82.0 vs 80.4 (+1.6, within noise); thinking length −43% | 1–2 | Possible "cheap thinker" arm. Claude-distilled data carries a ToS caveat |
| DavidOKB/MathThink-Qwen-3.5-4B | Qwen3.5-4B | LoRA r64 SFT on `nvidia/Nemotron-SFT-Math-v3`; merged safetensors in a subfolder (α32) | No numbers | 1 | Only if we evaluate it; second-choice maths arm |
| Kerassy/Qwen3.5-4B-Medical-Reasoning | Qwen3.5-4B | LoRA SFT (merged) on `FreedomIntelligence/medical-o1-reasoning-SFT` | MedQA 68.3% (full 1,273), no base comparison; base scores ~71.7 | 1 | Skip; no evidence of gain |
| helloimsaif/medjev-4b-lora | Qwen3.5-4B | LoRA decision model; 3.4k examples | Careful paired test: MedQA +1.7±4.1, MedMCQA −1.7, PubMedQA +0.5 | 3 (for the eval) | Not a generator. Useful as **evidence that base Qwen3.5-4B is already strong on medicine** |
| OrionLLM/GRM-2.5 | Qwen3.5-4B | Undisclosed | MMLU-Pro 80.1, GPQA 76.7, HMMT 75.2; each is ≈ Qwen's official 4B score +0.3–1.2 | 1 | Skip; essentially the base |
| Merlin-Research/Qwen3.5-4B-Safety-Thinking | Qwen3.5-4B | LoRA SFT merged; private safety/reasoning data | No numbers (1,259 dl, 20 likes) | 1 | Truthfulness arm only if evaluated |
| celestialcreator/Qwen3.5-0.8B-GRPO-Math (zosmaai/… is a near-duplicate) | Qwen3.5-0.8B | GRPO on GSM8K, full weights, **text-only arch** | GSM8K zero-shot 52.1→58.0 | 2 | Optional 0.8B maths arm. Needs vLLM ≥0.27 or a config wrapper |
| kai-os/Grug-35B-A3B | Qwen3.6-35B-A3B | QLoRA merged; ~5.7k "compact reasoning" traces (672 maths) | Fewer reasoning tokens; no accuracy table | 1–2 | Low priority. BF16 text-only arch, so we would have to quantise to NVFP4 ourselves (ModelOpt) and need a newer vLLM |
| zait-ai/toffee-3b-beta | Ministral-3-3B-Base | SFT (UltraChat, OpenMathInstruct-2, CodeFeedback); `MistralForCausalLM` | MMLU 63.1, HellaSwag 72.0, BoolQ 85.0, Winogrande 68.1; no Ministral comparison | 1 | Skip |
| ertghiu256/ministral-3-3b-glm-distillation | Ministral-3-3B-Reasoning | LoRA r8: SFT on 919 GLM logic puzzles + 50 GRPO steps; custom `[SOLUTION]` format; 4K context | None | 1 | Skip |
| XYZAILab/XYZ-Aquila-mini, FrontisAI/Frontis-MA1-35B (CC-BY-NC), endless-frontier/Fx-Work, Accio-Lab/occamy-1.0 | Qwen3.6-35B-A3B | Org-backed SFT/RL for deep search, ML engineering and co-work | Agentic benchmarks only | 2–3 | Credible but **off-domain** for MCQ/maths benchmarks. Occamy ships an official NVFP4 (23.9 GB) |
| rajveer43/gemma-4-E4B-medical-legal-finance-qa; epicmajorman/Gemma4-Biomedical-E4B | gemma-4-E4B-it | QLoRA on a Kaggle T4; the second is **abliterated** | Weak or none | 1 | Skip |

**Recommendation.**
- Among the four chosen, **Ministral-3-3B-Reasoning** has the strongest provenance and **RL-MATH** is the easiest to serve.
- **BigBang** is credible but tuned for agentic, tool-using research, and it competes with the base for memory.
- **Med-Qwen** has the weakest evidence and needs workarounds to load. Consider replacing it with a ShiningValiant3 (science) evaluation, or simply with Qwen3.5-4B in thinking mode as the "medical" arm.
