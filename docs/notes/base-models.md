# Candidate base-model families for the agent pool (survey, 2026-10-19)

**Scope.** The machine is one DGX Spark (GB10, aarch64, 121 GB unified memory) running vLLM 0.21.0 from `nvcr.io/nvidia/vllm:26.05.post1-py3`, which ships transformers 5.6.0. The pool already has Qwen3.5-0.8B/4B, Gemma-4-E4B, Ministral-3-3B (+Reasoning) and Qwen3.6-35B-A3B NVFP4.

**How I checked.**
- **Architecture support:** I grepped `registry.py` inside the image with a throwaway `docker run --rm`.
- **Repo facts:** I spot-checked repo ids, gating, licences and `architectures` through the HF API.
- **Scores:** these come from model cards or tech reports. "card" means the vendor's own figure; ‡ means a high self-reported figure that is worth reproducing.

## 0. Key facts up front
- **None of the candidates is in the local HF cache** (`~/.cache/huggingface/hub`, mounted by every serve script).
  - The only related item is `models--allenai--dolma2-tokenizer`, which is OLMo's tokenizer and not a model.
  - So every addition needs a download.
- **Meta has released no new small Llama since Llama 3.2 (Sep 2024).**
  - Reports say Meta paused open weights after Llama 4 Scout/Maverick (Apr 2025), which are too big anyway.
  - "Small Llama" therefore means Llama 3.2 1B/3B and Llama 3.1 8B.
  - These repos are **gated with manual approval** (`gated: manual`), and I don't know whether our token has access.
  - Ungated FP8 mirrors exist: `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic`, `RedHatAI/Llama-3.2-{1B,3B}-Instruct-FP8-dynamic`, `nvidia/Llama-3.1-8B-Instruct-FP8` and `nvidia/Llama-3.1-8B-Instruct-NVFP4`.
- **Not loadable on vLLM 0.21:**
  - These are in 0.21's `_PREVIOUSLY_SUPPORTED_MODELS` list (removed when the V0 engine was retired):
    - **Phi-4-mini-flash-reasoning** (`Phi4FlashForCausalLM`, removed in 0.10.2).
    - **Motif-2-12.7B** (`MotifForCausalLM`, removed in 0.10.2).
  - These use architectures 0.21 doesn't have:
    - Gemma-4-12B (`Gemma4UnifiedForConditionalGeneration`).
    - Apertus-v1.5.
    - K2-Horizon-7B.
    - Nemotron-Labs-Dory-4B.
    - granite-switch-4.1.
- **SmolLM3** loads only through vLLM's Transformers backend (`SmolLM3ForCausalLM → TransformersForCausalLM`). It works but is slower.
- **Licence blockers:**
  - The **Tencent Hunyuan** small-model licence excludes the UK and EU, which rules it out for UK use.
  - **EXAONE 4.x** is non-commercial only.

## 1. Candidate bases

These are the main candidates. Smaller families are listed afterwards. Memory share assumes our serve settings (FP8 weights, FP8 KV cache, 32k context, 20 sequences).
- Weights alone are about 1.1 GB per billion parameters at FP8.
- The "util" column is the `GPU_MEM_UTIL` I suggest; our existing small servers use 0.2–0.4.
- Hybrid Mamba models (NemotronH, FalconH1, GraniteMoeHybrid, LFM2) have much smaller KV per token.

| Base (HF repo) | Params total/active | Arch (vLLM 0.21) | Licence / gated | Thinking toggle | Ctx | Published scores (source) | GB10 quant | Suggested util (≈GB) |
|---|---|---|---|---|---|---|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | 1.2B | LlamaForCausalLM ✓ | Llama 3.2 Community, **gated (manual)** | none | 128K | GSM8K 44.4, MATH 30.6, GPQA 27.2 (Meta card) | RedHatAI FP8-dynamic | 0.10 (~12) |
| `meta-llama/Llama-3.2-3B-Instruct` | 3.2B | LlamaForCausalLM ✓ | Llama 3.2, **gated** | none | 128K | GSM8K 77.7, MATH 48.0, GPQA 32.8 (Meta card) | RedHatAI FP8-dynamic | 0.15 (~18) |
| `meta-llama/Llama-3.1-8B-Instruct` | 8.0B | LlamaForCausalLM ✓ | Llama 3.1, **gated** | none | 128K | MMLU-Pro 48.3, GPQA 30.4, MATH 51.9, GSM8K 84.5 (Meta card) | nvidia FP8 and **NVFP4**; RedHatAI FP8 | 0.25 (~30) |
| `ibm-granite/granite-4.2-8b` (Aug 2026) | 8.8B dense | GraniteForCausalLM ✓ | Apache-2.0, open | `enable_thinking` True/False plus `low_effort=True`; uses `<think>` tags | 128K | MMLU-Pro 74.0, GPQA 64.1, AIME25 86.7 (card) | official `-fp8`, **`-nvfp4`**, `-mxfp4` | 0.25 (~30) |
| `ibm-granite/granite-4.2-3b` | 3.7B dense | GraniteForCausalLM ✓ | Apache-2.0, open | same as 8B | 128K | MMLU-Pro 67.8, GPQA 54.8, AIME25 78.3 (card) | `-fp8`, `-nvfp4` | 0.15 (~18) |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | 8.9B, Mamba2-hybrid | NemotronHForCausalLM ✓ | NVIDIA Open Model, open | `/think` or `/no_think` in the system prompt; budget control | 128K | MATH500 97.8, GPQA 64.0, AIME25 72.1 (card) | `-FP8`, **`-NVFP4`** | 0.20 (~24) |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (Mar 2026) | 4.0B hybrid | NemotronHForCausalLM ✓ (needs ≥0.15.1) | NVIDIA Open, open | `enable_thinking` | 262K | MATH500 95.4, GPQA 53.2, AIME25 78.5 (card) | `-FP8` | 0.15 (~18) |
| `openai/gpt-oss-20b` | 20.9B / 3.6B MoE | GptOssForCausalLM ✓ | Apache-2.0, open | `reasoning_effort` low/medium/high (Harmony format; **not** `<think>`) | 128K | MMLU-Pro 75.0, GPQA 71.5, AIME25 91.7 (NVIDIA's evaluation, from the Nemotron-3 card) | native MXFP4 (~13 GB) | 0.25 (~30) |
| `tiiuae/Falcon-H1R-7B` | 7.6B hybrid | FalconH1ForCausalLM ✓ | Falcon LLM licence (Apache-based), open | always reasons | 256K | MATH500 97.4, AIME25 83.1, GPQA-D 61.3, MMLU-Pro 72.1 (card) | official `-FP8` | 0.20 (~24) |
| `allenai/Olmo-3-7B-Think` / `-7B-Instruct` | 7.3B | Olmo3ForCausalLM ✓ | Apache-2.0, open, **fully open data** | separate repos | 64K | Think: MATH 95.1, AIME25 64.6, GPQA 46.2. Instruct: 87.3 / 32.5 / 40.4 (card) | none (FP8 at load) | 0.20 each (~24) |
| `LiquidAI/LFM2.5-1.2B-Thinking` / `-Instruct` | 1.2B hybrid conv | Lfm2ForCausalLM ✓ | LFM Open v1.0, open | separate repos | 128K | Thinking: GSM8K 85.6, MATH500 88.0, GPQA-D 37.9, MMLU-Pro 49.7, AIME25 31.7. Instruct: 64.5 / 63.2 / 38.9 / 44.4 / 14.0 (card) | none | 0.08 (~10) |
| `LiquidAI/LFM2.5-8B-A1B` | 8.5B / 1.5B MoE | Lfm2MoeForCausalLM ✓ | LFM Open v1.0 | always thinks | 128K | MATH500 88.8, AIME25 42.5 (card) | none | 0.15 |
| `microsoft/Phi-4-mini-instruct` / `-mini-reasoning` | 3.8B | Phi3ForCausalLM ✓ | MIT, open | separate repos | 128K | Instruct: MMLU-Pro 52.8, GPQA 25.2, GSM8K 88.6. Reasoning: MATH500 91.2, AIME24 48.1, GPQA-D 44.5 (card) | RedHatAI FP8-dynamic | 0.15 |
| `HuggingFaceTB/SmolLM3-3B` | 3.1B | SmolLM3 via **Transformers backend** | Apache-2.0 | `enable_thinking` or `/think` `/no_think` | 64K | think: AIME25 36.7, GPQA-D 41.7. no_think: 9.3 / 35.7 (card) | RedHatAI FP8-dynamic | 0.15 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` / `-7B` | 1.8B / 7.6B | Qwen2ForCausalLM ✓ | MIT | always thinks | 128K | 7B: MATH500 92.8, AIME24 55.5, GPQA-D 49.1 (DeepSeek paper) | community FP8 | 0.10 / 0.20 |
| `LGAI-EXAONE/EXAONE-4.0-1.2B` | 1.3B | Exaone4ForCausalLM ✓ | **non-commercial** | `enable_thinking` | 64K | reasoning: MMLU-Pro 59.3, GPQA-D 52.0, MATH500 88.8. Non-reasoning: 52.0 / 40.1 / 71.2 (card) | `-FP8` | 0.08 |
| `tencent/Hunyuan-{0.5,1.8,4,7}B-Instruct` | 0.5–7.5B | HunYuanDenseV1ForCausalLM ✓ | **excludes UK/EU** | `enable_thinking` or `/no_think` | 256K | 7B: MATH 93.7, AIME25 75.3, GPQA-D 60.1 (card) | official FP8 | skip |

Other families that load on 0.21 but are less compelling:
- `zai-org/GLM-4.7-Flash`: 31B/3B MoE, MIT, AIME25 91.6, GPQA 75.2. It overlaps Qwen3.6-35B-A3B in the cost tier.
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`: NVFP4 available, MMLU-Pro 78.3, GPQA 73.0.
- `openbmb/MiniCPM5-2B`: LlamaForCausalLM, Apache, card claims AIME25 86.5‡ and GPQA-D 70.2‡, which are implausible for 2B. Reproduce before trusting.
- `EssentialAI/rnj-1-instruct`: 8.3B, uses the Gemma3 architecture, no thinking mode.
- `ai21labs/AI21-Jamba-Reasoning-3B`: JambaForCausalLM, MMLU-Pro 61.0.
- `arcee-ai/Trinity-Nano-Preview`: 6B/1B AfmoeForCausalLM, NVFP4 available, scores unverified.
- `XiaomiMiMo/MiMo-7B-RL-0530`: MATH500 97.2.
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`: built on Qwen3, so it isn't a new family.

Older **Granite 4.0/4.1** (no thinking mode) are much weaker than 4.2, e.g. Granite 4.0 Micro has MMLU-Pro 44.5 and GPQA 30.1. **Granite 4.2 supersedes them**.

**Thinking-budget compatibility.** Our `THINKING_TOKEN_BUDGET` and `--reasoning-config` mechanism needs `<think>`-style tags.
- **Should work** (all emit `<think>`; check each one's reasoning parser): Granite 4.2, Nemotron Nano, Falcon-H1R, OLMo-3-Think, SmolLM3, R1-Distill and LFM2.5-Thinking.
- **Does not apply to gpt-oss:** it uses Harmony channels, so control it with `reasoning_effort` instead. That needs per-model handling in `OpenAICompatChatWrapper`.
- **Llama 3.x and Phi-4-mini-instruct** have no thinking mode. That is useful as a cheap, deliberately non-reasoning tier.

## 2. Specialists each base unlocks
Credibility (Cred): 3 = reputable org plus paper plus benchmarks; 2 = real org with thin or self-reported evidence; 1 = hobbyist. Exact base lineage matters for the "specialist must be on a pool base" rule. **Llama-3-8B and Llama-3.1-8B are different bases**, so strictly you'd need both in the pool.

| Base | Specialist (HF repo) | Domain | Evidence | Cred |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | `FreedomIntelligence/HuatuoGPT-o1-8B` | Medicine, reasoning | MedQA 72.6 vs 58.7 base (arXiv 2412.18925) | 3 |
| Llama-3.1-8B-Instruct | `EPFLiGHT/Meditron3-8B` | Medicine | EPFL; benchmarks "coming soon"; gated (auto) | 2 |
| Llama-3.1-8B (merge) | `HPAI-BSC/Llama3.1-Aloe-Beta-8B` | Medicine | only +0.8 pt over Instruct (arXiv 2505.04388); CC-BY-NC | 2 |
| Llama-3.1-8B-Instruct | `TheFinAI/Fino1-8B` | **Finance** reasoning | FinQA, DocMath, TATQA (arXiv 2502.08127) | 2–3 |
| Llama-3.1-8B base | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Maths, reasoning | MATH500 89.1, AIME24 50.4 | 3 |
| Llama-3.1-8B-Instruct | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | Maths, science (thinking on/off) | NVIDIA card | 3 |
| Llama-3.1-8B (pruned) | `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` | Reasoning | NVIDIA card | 3 |
| Llama-3-8B-Instruct | `m42-health/Llama3-Med42-8B` | Medicine | MedQA 62.8, USMLE 67.0 (arXiv 2408.06142) | 3 |
| Llama-3-8B-Instruct | `ricdomolm/lawma-8b` | Law (classification only; outputs labels) | ICLR 2025, SCOTUS tasks | 3, but unusable as a debater |
| Llama-3-8B-Instruct | `jujipotle/honest_llama3_8B_instruct` (ITI) | Truthfulness | **fit on TruthfulQA, so contaminated** | 1–2 |
| Llama-3 / 3.1-8B | `PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct(-v1.1)` | Hallucination **judge** | HaluBench | 2–3 (verifier role) |
| Mistral-7B v0.1 | `Equall/Saul-7B-Instruct-v1` | **Law** (the only credible small English legal generalist) | LegalBench-Instruct (arXiv 2403.03883), MIT | 3 (2024, weak base) |
| Qwen2.5-7B-Instruct | `SUFE-AIFLM-Lab/Fin-R1` | **Finance** | FinQA 76 vs 60, ConvFinQA 85 vs 66 (arXiv 2503.16252) | 3 |
| Qwen2.5-7B-Instruct | `DianJin/DianJin-R1-7B` | Finance | Alibaba Cloud; only the 32B repo was confirmed to exist | 3 (verify repo) |
| Qwen2.5-7B-Instruct | `FreedomIntelligence/HuatuoGPT-o1-7B` | Medicine | arXiv 2412.18925 | 3 |
| Phi-4-mini 3.8B | `microsoft/Phi-4-mini-reasoning` | Maths | MATH500 91.2 | 3 |
| Phi-4 14B | `microsoft/Phi-4-reasoning(-plus)`, `EPFLiGHT/Meditron3-Phi4-14B` | Maths and science; medicine | AIME24 75.3 / 81.3; Meditron3 thin | 3 / 2 |
| Granite 3.3 (not 4.x) | `ibm-granite/granite-guardian-3.3-8b` | Groundedness and hallucination judge | #1 on REVEAL | 3 (needs a Granite 3.3 base) |

**No credible domain specialists** on Granite 4.x, OLMo 2/3, SmolLM3, Nemotron-Nano-9B-v2/3, Falcon-H1, LFM2 or gpt-oss-20b. The gpt-oss medical SFTs are hobbyist (Cred 1). Add these only as generalists.

**Gaps that remain:**
- **Truthfulness:** there is no credible truthful QA agent on any modern base. The credible factuality models are judges (Lynx, MiniCheck, Granite Guardian).
- **Law:** apart from Saul-7B, which is on Mistral-7B v0.1 rather than the Ministral-3 already in the pool, the only Llama-3 law model is Lawma, which is a classifier.

## 3. Recommendation

Add four bases chosen for spread across family, architecture and cost, plus one optional specialist host.

| # | Base to add | Why (spread) | Serving notes | Specialists unlocked |
|---|---|---|---|---|
| 1 | **Llama 3.1 8B Instruct** (+ optionally **Llama 3.2 3B** as the cheap end) | Meta family. A deliberately *weak, non-thinking* tier: MMLU-Pro 48, GPQA 30. It gives the orchestrator a real capability and cost gradient. | **Gated (manual)**: check token access, or use ungated `nvidia/Llama-3.1-8B-Instruct-NVFP4` or `RedHatAI/...-FP8-dynamic`. Plain dense model; serving risk is low. | HuatuoGPT-o1-8B (med, 3), Fino1-8B (**finance**, 2–3), R1-Distill-Llama-8B (maths, 3), Nemotron-Nano-8B-v1 (reasoning, 3), Meditron3-8B (2). Adding Llama-3-8B too unlocks Med42-v2 (3). Lynx-8B fits as a judge. |
| 2 | **IBM Granite 4.2 8B** (and/or **3B**) | The strongest new small dense family: MMLU-Pro 74 and GPQA 64 at 8B. Apache, ungated. Three thinking modes (on, off, low-effort) in one model, a good test of thinking on/off. | Official **NVFP4** and FP8. `<think>` tags, so THINKING_TOKEN_BUDGET should work. Verify the reasoning parser on 0.21; the card suggests `auto`/`nemotron_v3`. | none (Guardian is on Granite 3.3) |
| 3 | **gpt-oss-20b** | OpenAI family. MoE with 3.6B active, so it is cheap per token. Strongest on the hard set (GPQA ~71, AIME25 ~92). The only effort-level thinking control. | MXFP4 native (~13 GB). Needs Harmony / `reasoning_effort` handling in the wrapper; the existing budget mechanism doesn't apply. NVIDIA's Spark playbooks run it; smoke-test on 0.21. | none credible |
| 4 | **NVIDIA Nemotron Nano 9B v2** (or **Nemotron-3-Nano-4B** for a cheaper slot) | NVIDIA family on a **Mamba2-hybrid** architecture, which is structurally unlike anything in the pool. Small KV cache. `/think` and `/no_think`. | Official **NVFP4**/FP8 suit GB10. The hybrid models need prefix caching and other settings checked; see the vllm README. | none (Llama-Nemotron-Nano-8B sits under Llama, #1) |
| 5 (optional) | **Qwen2.5-7B-Instruct** as a *specialist host* | Same vendor as the pool, but an older generation with a different base. Its value is the specialists, mainly finance. | Plain Qwen2 architecture; trivial to serve. | Fin-R1 (**finance**, 3), DianJin-R1-7B (finance, 3, verify), HuatuoGPT-o1-7B (med, 3). R1-Distill-Qwen-7B is on Qwen2.5-Math, so it is strictly a different base. |

**Alternates:**
- **OLMo-3-7B Think/Instruct:** fully open and Apache, with a clean think/no-think pair, but no specialists. Choose it over #4 if openness matters more than architectural diversity.
- **Falcon-H1R-7B:** strong hybrid reasoner.
- **LFM2.5-1.2B Thinking/Instruct:** a tiny third family at the cost floor, though Qwen3.5-0.8B already covers that tier.
- **For law coverage:** the only credible option is adding **Mistral-7B-Instruct-v0.1** as a weak base, purely to host Saul-7B. Treat it as an explicit, documented exception rather than a general-purpose agent.

**Memory budget.** Llama-8B (0.25) + Granite-4.2-8B (0.25) + gpt-oss-20b (0.25) + Nemotron-9B (0.2) ≈ 0.95 of 121 GB. That doesn't fit alongside the existing servers on one Spark, so lower `MAX_NUM_SEQS`/util per server (0.12–0.15 is enough for the ≤4B and NVFP4 models). Alternatively, spread the servers over the second Spark when it arrives.

**Download needs.** Nothing is cached. NVFP4/FP8 checkpoints are about 5–13 GB each; the Llama repos need gated access.
