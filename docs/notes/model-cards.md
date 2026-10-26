# Model cards of candidate specialists (fetched 2026-10-22)

What each fine-tune's Hugging Face card says, condensed, with the datasets it should be used for.
Part A covers fine-tunes for the hard pool's domains; Part B covers tool-use and agentic fine-tunes
for the agentic pool; Part C covers general-purpose and adjacent-domain fine-tunes for the subjects
with no exact match (law, biology, engineering, Plancraft). All claims are the authors' own and none is reproduced; credibility (1–3) is
our judgement of how far to trust them. "Loads on 0.21" is judged from the config's architecture
only, until a smoke test. Training details beyond the cards are in [`specialists.md`](specialists.md)
and [`base-models.md`](base-models.md).

## Which models for which dataset

"Served base" means the model is a fine-tune of a base we already serve (Qwen3.5-4B/0.8B,
Qwen3.6-35B, Gemma-4-E4B, Ministral-3-3B); the others need a new base family served first (#32).
Teams have 4 workers, so at most two specialists per source (x1-plan Q20/Q23).

| Dataset | On a served base | Needs a new base family | Not loadable on vLLM 0.21 |
|---|---|---|---|
| AIME, MATH-500 L4–5 | RL-MATH (4B), Ministral-3B-Reasoning, AgentMercury-4B (AIME claim only, cred 1) | R1-Distill-Llama-8B | — |
| GPQA-D | ShiningValiant3 (E4B), RL-MATH (claims GPQA), Ministral-3B-Reasoning, BigBang (science, no GPQA number) | R1-Distill-Llama-8B | — |
| GPQA biology (adjacent) | BigBang (BioMystery-HS 45→58); Reasoning-Medical0.1-E4B (MMLU-Pro bio 0.75, no base row, cred 1); Qwen3.5-4B-Neo (within noise, part Claude data) | BioMed-R1-8B, Gene-R1-8B, MegaScience-8B (Llama) | — |
| MMLU-Pro physics, chemistry | ShiningValiant3, RL-MATH, Ministral-3B-Reasoning, BigBang | — | — |
| MMLU-Pro engineering (adjacent) | 0GM-1.0-35B-A3B (engineering 42.0→46.4 at 4k context, BF16 only, cred 1–2); RL-MATH; DeepSeek-V4-Pro-Qwen3.5-4B (STEM distill, no base row); Ministral-3B-Reasoning | OpenThinker3-7B (Qwen2.5, cred 3) | — |
| MMLU-Pro law (generalist) | No credible law tune. Generalists: BigBang; 0GM-1.0-35B-A3B (law flat 67.5→66.0); weak exact: gemma-4-E4B medical-legal-finance LoRA (ROUGE only, cred 1) | Saul-7B (Mistral-7B v0.1); calme-2.3-legalkit-8b (Llama, cred 1) | Personal-Jurisdiction-GRPO (one LegalBench task) |
| BrowseComp-Plus | Iris-mini, XYZ-Aquila-mini, BigBang, Agents-A1 (base unstated), T-Search (retriever only) (35B); Agents-A1-4B, AREX-Turbo, Apodex-1.0-4B (4B); ContextPilot-E4B | Harness-1, Context-1 (gpt-oss-20b); PokeeResearch-7B (weak) | — |
| Finance-Agent | XYZ-Aquila-mini, Occamy-1.0, Fx-Work (35B) | Harness-1 (SEC retrieval); Fin-R1, Fino1-8B (finance reasoning, no tool training) | — |
| WorkBench | CompoWorld, Occamy-1.0, Skill2Env, Fx-Work (35B); AgentMercury-4B (cred 1); CloudSurf-4B-FC (E4B) | xLAM-2-8b-fc-r (non-commercial), ToolACE-2.5, watt-tool-8B, Hermes-3 (Llama-8B); Toucan-7B (Qwen2.5) | NeoHorse-1-4B |
| Plancraft (adjacent) | R2S ALFWorld student (4B, ALFWorld 43→82, cred 2); SkillGym-Qwen3.5-4B (cred 2); CompoWorld (35B, DeepPlanning); TextCraft GRPO LoRAs (closest domain, no numbers, cred 1) | Plancraft author's Llama-8B LoRA (cred 1); early-experience-textcraft (Qwen2.5, cred 2) | — |
| (no matching source) | — | HuatuoGPT-o1-8B/-7B (medicine) | Med-Qwen-3.5-4B (medicine) |

No agentic fine-tunes exist for Ministral-3-3B, Granite-4.2-8B or Nemotron-Nano-9B-v2.

---

# Part A: domain fine-tunes (hard pool)

| Model | Base | Licence | Gated | Downloads / likes | Card updated |
|---|---|---|---|---|---|
| `wflying/Qwen3.5-4B-RL-MATH` | Qwen3.5-4B | Apache-2.0 | no | 149 / 0 | 2026-08-12 |
| `mistralai/Ministral-3-3B-Reasoning-2512` | Ministral-3-3B-Base-2512 | Apache-2.0 | no | 16,770 / 139 | 2026-07-15 |
| `ValiantLabs/gemma-4-E4B-it-ShiningValiant3` | gemma-4-E4B-it | Apache-2.0 | no | 11 / 4 | 2026-04-13 |
| `endless-frontier/BigBang-v1` | Qwen3.6-35B-A3B | Apache-2.0 | no | 2,230 / 233 | 2026-08-06 |
| `TheShed/BigBang-v1-W4A16-NVFP4-MTP` | BigBang-v1 (quantised) | Apache-2.0 | no | 286 / 1 | 2026-08-12 |
| `adi274903/Med-Qwen-3.5-4B-Instruct` | Qwen3.5-4B-Base | Apache-2.0 | no | 1,000 / 0 | 2026-10-14 |
| `FreedomIntelligence/HuatuoGPT-o1-8B` | Llama-3.1-8B-Instruct | Apache-2.0 | no | 2,971 / 62 | 2024-12-30 |
| `FreedomIntelligence/HuatuoGPT-o1-7B` | Qwen2.5-7B-Instruct | Apache-2.0 | no | 779 / 68 | 2025-01-06 |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Llama-3.1-8B (base) | MIT | no | 120,261 / 912 | 2025-02-24 |
| `TheFinAI/Fino1-8B` | Llama-3.1-8B-Instruct | Llama 3.1 | **yes (auto-approve)** | 42 / 35 | 2026-10-07 |
| `SUFE-AIFLM-Lab/Fin-R1` | Qwen2.5-7B-Instruct | not stated | no | 1,041 / 230 | 2025-03-21 |
| `Equall/Saul-7B-Instruct-v1` | Mistral-7B | MIT | no | 1,260 / 124 | 2024-03-10 |

---

## Qwen3.5-4B-RL-MATH

- **Tuned from:** `Qwen/Qwen3.5-4B`
- **Training:** GRPO-style RLVR with rule-verified maths rewards, in **non-thinking mode**
  (`enable_thinking=False`). Data: `wflying/math-rlvr-15k`, 15,000 English maths problems.
- **Architecture:** Qwen3.5 multimodal ConditionalGeneration, ~4.66B, BF16.
- **Claims (non-thinking; harness not stated):**

  | Model | AIME24 | AIME25 | HMMT25 | GPQA-D | ZebraLogic | LCB-v6 |
  |---|---|---|---|---|---|---|
  | Qwen3.5-35B-A3B | 64.8 | 60.0 | 48.8 | 82.8 | 90.6 | 71.9 |
  | Qwen3.5-4B | 58.3 | 38.3 | 30.0 | 71.2 | 82.6 | 52.1 |
  | RL-MATH | 74.4 | 58.5 | 50.4 | 76.9 | 89.8 | 62.7 |

- **Limitations stated:** specialised for maths; scores may not transfer to other prompts or
  decoding; multimodal behaviour not evaluated.
- **Use for:** AIME, MATH-500 L4–5 (trained on maths); GPQA-D and MMLU-Pro physics, chemistry and engineering (claims GPQA +5.7). Not agentic: no tool training.

## Ministral-3-3B-Reasoning-2512

- **Tuned from:** `mistralai/Ministral-3-3B-Base-2512` (a sibling of `Ministral-3-3B-Instruct-2512`, not a fine-tune of it)
- **Training:** Mistral's reasoning post-train of the 3B base (paper arXiv 2601.08584). It's a sibling
  of the Instruct model, not a fine-tune of it. 3.4B LM + 0.4B vision encoder, BF16, 256k context.
- **Recommended use:** prepend Mistral's `SYSTEM_PROMPT.txt` to any system prompt, temperature 0.7,
  keep reasoning traces in context; native function calling.
- **Serve:** vLLM ≥ 0.12, `--tokenizer_mode mistral --config_format mistral --load_format mistral
  --enable-auto-tool-choice --tool-call-parser mistral --reasoning-parser mistral`.
- **Claims:** AIME25 72.1, AIME24 77.5, GPQA-D 53.4, LiveCodeBench 54.8 (vs Qwen3-VL-4B-Thinking
  69.7 / 72.9 / 60.1 / 51.3). The Instruct sibling: MATH maj@1 83.0.
- **Use for:** AIME, MATH-500, GPQA-D, MMLU-Pro physics, chemistry and engineering. Possibly agentic tasks (claims native function calling), untested.

## ShiningValiant3 (gemma-4-E4B-it)

- **Tuned from:** `google/gemma-4-E4B-it`
- **Training:** SFT on DeepSeek-R1-0528-generated data: `sequelbox/Celestia3-DeepSeek-R1-0528`
  (science reasoning), `sequelbox/Mitakihara-DeepSeek-R1-0528` (AI/ML reasoning),
  `sequelbox/Raiden-DeepSeek-R1` (general and creative reasoning).
- **Use:** Gemma-4-E4B-it prompt format; the example enables thinking (`enable_thinking=True`).
- **Claims:** none. The card has no benchmark numbers. The same recipe exists on Qwen3-4B, Qwen3-8B,
  Ministral-3-14B-Reasoning and gpt-oss-20b.
- **Use for:** GPQA-D (chemistry, physics, biology), MMLU-Pro physics and chemistry. Not agentic.

## BigBang-v1 (and its NVFP4 build)

- **Tuned from:** `Qwen/Qwen3.6-35B-A3B` (the NVFP4 build is a quantisation of BigBang-v1)
- **Training:** a general-purpose post-train of Qwen3.6-35B-A3B on ~10,000 high-difficulty
  synthetic examples across scientific and technical domains. Generator agents propose and solve
  verifiable frontier problems; critic agents judge correctness, difficulty and diversity, calibrated
  against held-out real research tasks. Data not released.
- **Claims (vs Qwen3.6-35B-A3B):** BrowseComp 67.9→76.5, XBench 32.6→58.4, SWE-Bench Pro 43.6→54.2,
  FrontierScience-R 11.9→46.2, HLE 36.2→50.3, BioMystery-HS 44.8→57.5, MLE-Bench-Lite 31.8→59.1,
  PaperBench 30.7→53.6. No MATH, AIME or GPQA numbers.
- **Serve:** vLLM ≥ 0.19, `--reasoning-parser qwen3`, tools via `--tool-call-parser qwen3_coder`;
  keep ≥ 128K context "to preserve thinking capabilities"; `--language-model-only` drops the vision
  encoder.
- **NVFP4 build** (`TheShed/…`): ModelOpt 0.45 (MoE weights W4A16 NVFP4, attention FP8, MTP BF16),
  21.8 GiB; tested on vLLM 0.25.1 on a DGX Spark-class box with `--kv-cache-dtype bfloat16` (no
  calibrated FP8 KV scales), FlashInfer, Marlin MoE, `qwen3_xml` tool parser. No quality comparison
  against BF16.
- **Correction to `specialists.md`:** the card presents BigBang as general-purpose (science,
  reasoning, coding, tool use), not agentic-only, so it is a candidate for GPQA too.
- **Use for:** BrowseComp-Plus (BrowseComp claim), Finance-Agent and WorkBench (general tool use); GPQA-D and MMLU-Pro science (scientific-research claims). Plancraft untested.

## Med-Qwen-3.5-4B-Instruct

- **Tuned from:** `Qwen/Qwen3.5-4B-Base`
- **Training:** domain-adaptive pretraining on multi-stream biomedical text over Qwen3.5-4B-**Base**,
  then 2,700-step DoRA SFT (r 64, α 128) on an 11-stream clinical mixture (~276M tokens): long-CoT
  medical deduction, board vignettes, evidence grounding, STEM replay.
- **Claims (greedy, normalised choice log-probs):** PubMedQA 45.6→66.2, GPQA-D 36.9→50.5, MedQA
  68.3→68.0, MedMCQA 26.5→33.2, MMLU-med 82.9→80.7. MedMCQA near chance for every model points to a
  broken harness.
- **Note:** text-only `CausalLM`; not loadable on our vLLM 0.21 without a wrapper.
- **Use for:** none in our pools (medicine only; GPQA biology at most). Not loadable on vLLM 0.21.

## HuatuoGPT-o1-8B and -7B

- **Tuned from:** 8B: `meta-llama/Llama-3.1-8B-Instruct`; 7B: `Qwen/Qwen2.5-7B-Instruct`
- **Training** (paper arXiv 2412.18925): medical complex reasoning. Data:
  `FreedomIntelligence/medical-o1-reasoning-SFT` and `medical-o1-verifiable-problem`.
- **Use:** same as its base (Llama-3.1-8B-Instruct / Qwen2.5-7B-Instruct); vLLM supported. Output
  is formatted as `## Thinking` … `## Final Response`. The 7B is English and Chinese.
- **Claims:** none on the card; see the paper.
- **Use for:** none in our pools (medicine only; GPQA biology at most).

## DeepSeek-R1-Distill-Llama-8B

- **Tuned from:** `meta-llama/Llama-3.1-8B` (base, not Instruct)
- **Training:** SFT of Llama-3.1-8B (base) on samples generated by DeepSeek-R1.
- **Claims (max 32,768 tokens, T 0.6, top_p 0.95, pass@1 over 64 samples):** AIME24 50.4 (cons@64
  80.0), MATH-500 89.1, GPQA-D 49.0, LiveCodeBench 39.6, Codeforces 1205.
- **Usage recommendations:** temperature 0.5–0.7 (0.6), no system prompt (put instructions in the
  user turn), for maths ask for a step-by-step answer in `\boxed{}`, force the reply to start with
  `<think>\n`.
- **Use for:** AIME, MATH-500, GPQA-D. Needs a Llama-3.1-8B host (#32). Not agentic.

## Fino1-8B

- **Tuned from:** `meta-llama/Llama-3.1-8B-Instruct`
- **Training** (paper arXiv 2502.08127): SFT of Llama-3.1-8B-Instruct on GPT-4o reasoning paths
  derived from FinQA (`TheFinAI/Fino1_Reasoning_Path_FinQA`), then RL. 131k context.
- **Claims:** evaluated on FinQA, DocMath-Eval, XBRL-Math; numbers in the paper and leaderboard only.
- **Note:** gated (auto-approve; accept the terms on HF before download). The card recommends the
  newer Fin-o1 family for new work, but Fin-o1-8B is on Qwen3-8B, which isn't in our pool.
- **Use for:** Finance-Agent, only if it can call tools (trained on FinQA reasoning, not tool use). Needs a Llama host.

## Fin-R1

- **Tuned from:** `Qwen/Qwen2.5-7B-Instruct`
- **Training:** Qwen2.5-7B-Instruct, SFT on R1-distilled financial CoT filtered by Qwen2.5-72B
  (answer and reasoning checks), then GRPO with a format reward and a model-based verifier. 60,091
  examples (Chinese and English) from FinCorpus, Finance-Instruct-500K, ConvFinQA, FinQA, TFNS,
  FinanceIQ, Ant_Finance, FinCUGE, FinanceQT and FinPEE.
- **Claims:** FinQA 60→76, ConvFinQA 66→85, Ant_Finance 85→81, TFNS 68→71, Finance-Instruct-500K
  49→62.9; average 65.6→75.2 vs base.
- **Use for:** Finance-Agent, only if it can call tools (finance QA training, not tool use). Needs a Qwen2.5-7B host.

## Saul-7B-Instruct-v1

- **Tuned from:** `mistralai/Mistral-7B-v0.1`
- **Training** (paper arXiv 2403.03883): continued pretraining of Mistral-7B on legal text, then
  instruction tuning. English.
- **Claims:** none on the card; LegalBench-Instruct in the paper.
- **Note:** the card is the auto-generated template; the example uses greedy decoding.
- **Use for:** MMLU-Pro law (50 questions). Excluded so far: needs a Mistral-7B v0.1 host.


---

# Part B: tool-use and agentic fine-tunes (agentic pool)

Method: HF API `base_model:finetune:` / `adapter:` / `quantized:` listings for every pool base (Qwen3.5-4B 1000+, 0.8B 563, Qwen3.6-35B-A3B 389, gemma-4-E4B-it 424, Ministral-3-3B x3 ~108, Llama-3.1-8B(-Instruct) 2x1000+, Llama-3.2-3B-Instruct 1000+, Qwen2.5-7B-Instruct 1000+, granite-4.2-8b 20, gpt-oss-20b 568, Nemotron-Nano-9B-v2 27), keyword plus top-likes scan, plus name searches for known families. Cards, API metadata and config.json were fetched anonymously. "dl" means HF `downloads` (last 30 days). "Loads on 0.21" is judged only from `architectures` in config.json against the stated 0.21 support list. Anything not from the card is marked (not from card).

### Summary

| model | base | in pool? | matches which agentic benchmark | loads on 0.21 | cred |
|---|---|---|---|---|---|
| AllSpark-Research/Iris-mini | Qwen3.6-35B-A3B | yes | BrowseComp-Plus (deep search); Finance-Agent web part | yes | 2 |
| XYZAILab/XYZ-Aquila-mini | Qwen3.6-35B-A3B | yes | BrowseComp-Plus; Finance-Agent (search/scrape/python tool set) | yes | 2 |
| t-tech/T-Search | Qwen3.6-35B-A3B | yes | BrowseComp-Plus (retrieval only; reports BC-Plus Recall@10) | yes | 3 |
| Accio-Lab/occamy-1.0 | Qwen3.6-35B-A3B | yes | WorkBench (AutomationBench, BFCL, τ³-Banking); Finance-Agent (OfficeQA Pro) | yes | 2 |
| AllSpark-Research/CompoWorld | Qwen3.6-35B-A3B | yes | WorkBench (multi-service tools, AutomationBench); Plancraft-adjacent (DeepPlanning) | yes | 2 |
| AllSpark-Research/Skill2Env | Qwen3.6-35B-A3B | yes | WorkBench (weaker than CompoWorld) | yes | 2 |
| endless-frontier/Fx-Work | Qwen3.6-35B-A3B | yes | WorkBench / Finance-Agent (office-work deliverables) | yes | 2 |
| endless-frontier/BigBang-v1 (card in Part A) | Qwen3.6-35B-A3B | yes | BrowseComp-Plus (BrowseComp 76.5); hard pool (HLE) | yes | 2 |
| InternScience/Agents-A1 | not stated on card (Qwen3.5/3.6-35B-A3B arch) | unclear | BrowseComp-Plus; general tool use | yes | 2 |
| InternScience/Agents-A1-4B | not stated (config = Qwen3.5-4B dims; compared vs Qwen3.5-4B) | yes, probably | BrowseComp-Plus; general tool use | yes | 2 |
| BAAI/AREX-Turbo | Qwen3.5-4B | yes | BrowseComp-Plus | yes | 2 |
| apodex/Apodex-1.0-4B-SFT | Qwen3.5-4B | yes | BrowseComp-Plus (small gain) | yes | 2 |
| apodex/Apodex-1.0-0.8B-SFT | Qwen3.5-0.8B | yes | BrowseComp-Plus (weak) | yes | 2 |
| TokenRhythm/NeoHorse-1-4B | Qwen3.5-4B | yes | WorkBench (τ2, BFCL, office-style claw benches) | **no** (Qwen3_5ForCausalLM) | 2 |
| Minbyul/AgentMercury-Qwen3.5-4B | Qwen3.5-4B (card text) | yes | WorkBench (MCP RL with DB-state reward) | yes | 1 |
| Pepps233/Qwen3.5-0.8B-ToolCalling-OPD | Qwen3.5-0.8B | yes | WorkBench (single-turn FC only) | **no** (Qwen3_5ForCausalLM) | 1 |
| cloudsurf-software/CloudSurf-4B-FC | gemma-4-E4B-it | yes | WorkBench (BFCL multi-turn, τ²) | yes (BF16 in `merged/`) | 2 |
| tencent/ContextPilot-E4B | gemma-4-E4B-it | yes | BrowseComp-Plus (deep-search context mgmt) | yes | 2 |
| pat-jj/harness-1 | gpt-oss-20b | yes | BrowseComp-Plus (BC+ corpus); Finance-Agent (RL on SEC retrieval) | yes | 2 |
| chromadb/context-1 | gpt-oss-20b | yes | BrowseComp-Plus (retrieval subagent; harness unreleased) | yes | 2 |
| Salesforce/Llama-xLAM-2-8b-fc-r | Llama-3.1-8B-Instruct (not from card) | yes | WorkBench | yes | 3 |
| Team-ACE/ToolACE-2.5-Llama-3.1-8B (and ToolACE-2) | Llama-3.1-8B-Instruct | yes | WorkBench | yes | 3 |
| watt-ai/watt-tool-8B | Llama-3.1-8B-Instruct | yes | WorkBench | yes | 2 |
| gautierdag/oa-llama3-r64-a32 | Llama-3.1-8B-Instruct (LoRA adapter config) | yes | Plancraft (inferred) | probably (LoRA saves embed/lm_head, so merge first) | 1 |
| NousResearch/Hermes-3-Llama-3.1-8B | Llama-3.1-8B (base) | yes (base variant) | WorkBench (generic FC) | yes | 2 |
| PokeeAI/pokee_research_7b | Qwen2.5-7B-Instruct | yes | BrowseComp-Plus (weak on BrowseComp) | yes | 2 |
| Agent-Ark/Toucan-Qwen2.5-7B-Instruct-v0.1 | Qwen2.5-7B-Instruct | yes | WorkBench (MCP multi-tool) | yes | 2 |
| SUFE-AIFLM-Lab/Fin-R1 (card in Part A) | Qwen2.5-7B-Instruct | yes | Finance-Agent (reasoning only, no tools) | yes | 2 |
| Mindcraft-CE/Andy-4.2-Air | Qwen3.5-4B | yes | Plancraft (Minecraft, but a different interface) | yes | 1 |
| *Off-pool:* Tongyi-DeepResearch-30B-A3B, MiroThinker-v1.5-30B, WebExplorer-8B, Jan-nano, II-Search-4B, Nex-N2-mini, Apodex-1.1-mini, Holo3, Ornith-1.5, Hammer2.1-7b, xLAM-2-3b-fc-r, Qwen-AgentWorld | Qwen3-4B/8B/30B-A3B, Qwen3.5-35B-A3B(-Base), Qwen2.5-Coder | no | see the off-pool section | n/a | |

Screened out as not relevant (GUI/vision, decision classifiers, coding, memory archs): Holo4-35B-A3B and Holo-3.1-* (computer use), microsoft/Fara1.5-4B (browser CUA from screenshots), Frontis-MA1-35B (MLE-Bench), juspay/xor, nace-ai/drex-v2.0, internlm/Intern-Decision-*, pplx-decider-flash, h2o-lightning-4b, Tev1-* (typed-decision classifiers), MemT-1.0-Nano (custom arch), SIQ-1-35B (Qwen3_5MoeForCausalLM, so no), Polaris-V1, zira-researcher, Noema-1.5-4B-Deep (all Qwen3_5ForCausalLM), CogEvol-4B, DMind-3-mini, GameScaling (keyboard/mouse game control).

There are **no on-pool candidates** for Ministral-3-3B (Instruct/Base/Reasoning), granite-4.2-8b or NVIDIA-Nemotron-Nano-9B-v2. Their fine-tune lists contain only quantisations, abliterations, RP and translation tunes. For Llama-3.2-3B-Instruct, the only agentic tune is `Alibaba-NLP/ZeroSearch_*_Llama_3.2_3B_Instruct`, which has an empty card.

---

## Iris-mini
- **Tuned from:** `Qwen/Qwen3.6-35B-A3B`
- **Repo:** `AllSpark-Research/Iris-mini` · base `Qwen/Qwen3.6-35B-A3B` · apache-2.0 · not gated · 714 dl / 18 likes · 2026-09-02 · `Qwen3_5MoeForConditionalGeneration` (loads on 0.21: yes). No official FP8/NVFP4. A community `felkf/Iris-mini-oQ6e-fp16-mtp` exists.
- **Training:** "alternating rounds of supervised fine-tuning and reinforcement learning" for the search loop. No data details on the card.
- **Claims:** BrowseComp 82.2, BrowseComp-ZH 84.8, DeepSearchQA 86.9 (F1), HLE (text-only, 2,158 Q) 52.3, all under the `discard-all` context setting. The same setting without context management gives BrowseComp 64.7 and HLE 43.2. Card has no base-model row. Peer rows from the card: XYZ-Aquila-mini 78.8 and Agents-A1 75.5 on BrowseComp. Harness: Iris-Harness, with one tool set, one context limit and one judge.
- **Serving notes:** SGLang `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`, context 262144. Uses OpenAI function calling. Final answer goes in `\boxed{}`. "Prior reasoning is carried back into the conversation on each turn" (the harness config handles this). Eval uses `--context-discard-threshold 131072`.
- **Use for:** BrowseComp-Plus is the strongest fit (search, read, decide when evidence is enough). The card has no BC-Plus number. Could also cover the web-search half of Finance-Agent. Hard pool: HLE only.
- **Credibility:** 2. The org is newer. Numbers come with a released harness and an honest context-management ablation, but they are self-reported and not reproduced.

## XYZ-Aquila-mini
- **Tuned from:** `Qwen/Qwen3.6-35B-A3B`
- **Repo:** `XYZAILab/XYZ-Aquila-mini` · base `Qwen/Qwen3.6-35B-A3B` · apache-2.0 · not gated · 550 dl / 443 likes · 2026-07-29 · `Qwen3_5MoeForConditionalGeneration` (yes). Community `r0b0tlab/XYZ-Aquila-mini-NVFP4` exists.
- **Training:** post-trained through a "bounded-exploration AI4AI pipeline", in which AI agents propose interventions across data, post-training, runtime and context management. SFT data is released as `XYZAILab/XYZ-Aquila-SFT`. No algorithm details on the card beyond the tech report.
- **Claims:** BrowseComp 78.8, BrowseComp-ZH 82.9, DeepSearchQA 89.5, GAIA 97.1, LiveBrowseComp 48.7, HLE 51.1, WideSearch 80.8. No base-model row. Harness: "ReAct-style search harness with web search, webpage extraction, stateful Python execution, and a maximum 256K context". Card says the external benchmarks "held out from routine AI4AI optimization".
- **Serving notes:** thinking model with "Qwen-compatible reasoning and tool-call formats". SGLang ≥0.5.14 `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`, context 262144. Example system prompt asks for one tool call per turn and a final `\boxed{}`. Tools are `web_search`, `scrape_and_extract_info` and `python`, via the AxisAgentic harness.
- **Use for:** BrowseComp-Plus. Also Finance-Agent, because its trained tool set (search, scrape/extract, python) closely matches Finance-Agent's web search, parse HTML and run python (not from card: this mapping is my inference). Hard pool: HLE only.
- **Credibility:** 2. Data is released and the harness is open, but there is no base comparison and the scores are self-reported.

## T-Search
- **Tuned from:** `Qwen/Qwen3.6-35B-A3B`
- **Repo:** `t-tech/T-Search` · base `Qwen/Qwen3.6-35B-A3B` · apache-2.0 · not gated · 372 dl / 56 likes · 2026-07-16 · `Qwen3_5MoeForConditionalGeneration` (yes). **Official `t-tech/T-Search-NVFP4` and `-FP8` exist.**
- **Training:** SFT on 11K fully synthetic search trajectories (8K questions per language, EN and RU), then GSPO RL with a recall-over-gold-chunk reward. EN and RU experts are merged.
- **Claims (Recall@10, own harness, fixed indices):** BrowseComp-Plus 65.35 (N=1) and 72.65 (N=3 RRF), against Qwen3.6-35B-A3B 43.66. With the Qwen3-Embedding-8B + LLM rerank retriever, BC-Plus is 75.04. With Qwen3-Embedding-0.6B it is 51.70, and with BM25 39.49. SealQA is 61.16 vs 46.07. Card says there is no noticeable difference from base on general benchmarks.
- **Serving notes:** temp 0.7, top_p 1.0. SGLang `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`, context 65536, MTP (NEXTN) speculative decoding. The model expects its own harness: `search_corpus`, `save_and_advance` and `finalize_ranking` in rounds with a ~32K budget. **It returns a ranked evidence set, not an answer.**
- **Use for:** BrowseComp-Plus. It is the only on-pool model with a reported BC-Plus number, but it is a retriever, so it needs an answerer downstream. The BC-Plus tool interface (search + get_document) would need adapting to its 3-tool round protocol.
- **Credibility:** 3. Established org (T-Bank), clear harness, base comparison on the target benchmark, and a retriever ablation.

## Occamy-1.0
- **Tuned from:** `Qwen/Qwen3.6-35B-A3B`
- **Repo:** `Accio-Lab/occamy-1.0` · base `Qwen/Qwen3.6-35B-A3B` · apache-2.0 · not gated · 2293 dl / 236 likes · 2026-10-02 · `Qwen3_5MoeForConditionalGeneration` (yes). **Official `-NVFP4` (23.9 GB) and `-FP8`.** The card says it passed a 9-case vLLM check, including tool calls.
- **Training:** full-parameter SFT across two experts. "Marathon" gets SFT then HDPO, "Sprint" gets SFT. The experts are uniformly merged, then SAO RL is applied. SFT union is 14,998 trajectories / 403M tokens: general agentic, long-horizon interactive, terminal/SWE and tool-call grounding. Uses the Dressage RL stack.
- **Claims (vs Qwen3.6-35B-A3B):** Claw-Eval 82.20 vs 69.50. AutomationBench Pass¹ 27.60 vs 7.50, partial 69.10 vs 39.40. BFCL v4 65.40 vs 63.19. τ³-Bench Banking 37.10 vs 11.90. OfficeQA Pro 48.10 vs 39.10. VitaBench 41.75 vs 34.25. Terminal-Bench 2.1 59.00 vs 49.50. IFEval 91.53 vs 86.90. The card notes that "Retrieval-heavy and simulated-user tasks still have headroom".
- **Serving notes:** vLLM ≥0.19 `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`. SFT used 131K sequences.
- **Use for:** WorkBench (structured API and productivity tool use with state, AutomationBench/BFCL gains). Finance-Agent (OfficeQA Pro and banking gains; python and tool use).
- **Credibility:** 2. Thorough card with base rows and official quants. Self-reported.

## CompoWorld
- **Tuned from:** `Qwen/Qwen3.6-35B-A3B`
- **Repo:** `AllSpark-Research/CompoWorld` · base `Qwen/Qwen3.6-35B-A3B` · apache-2.0 · not gated · 113 dl / 3 likes · 2026-09-29 · `Qwen3_5MoeForConditionalGeneration` (yes). No quants.
- **Training:** 3K verified SFT trajectories plus 1K RL tasks over 448 composed services (10,130 tools), with a "Completion-Focused Rubric Reward".
- **Claims (vs Qwen3.6-35B-A3B):** AutomationBench 32.33 vs 10.33. τ³-Banking 16.49 vs 10.65. DeepPlanning 35.21 vs 26.04. VitaBench 49.44 vs 38.94. SkillsBench 47.71 vs 32.52. WildClawBench 47.46 vs 44.28. ALE 13.59 vs 7.77. Harnesses: OpenHands for SkillsBench, Claude Code for ALE, OpenClaw for WildClawBench.
- **Serving notes:** vLLM `--tool-call-parser qwen3_coder --reasoning-parser qwen3`, context 262144.
- **Use for:** WorkBench. Its cross-service composed tool tasks with verified state are the closest training distribution to WorkBench's email/calendar/CRM/analytics/PM tools (inference, not from card). Its DeepPlanning gain is a weak Plancraft signal.
- **Credibility:** 2. Paper and base rows are present. Low adoption.

## Skill2Env
- **Tuned from:** `Qwen/Qwen3.6-35B-A3B`
- **Repo:** `AllSpark-Research/Skill2Env` · base `Qwen/Qwen3.6-35B-A3B` · apache-2.0 · not gated · 77 dl / 5 likes · 2026-10-03 · `Qwen3_5MoeForConditionalGeneration` (yes).
- **Training:** 1.5K SFT trajectories from 2,963 synthesized, skill-derived environments.
- **Claims (vs base):** AutomationBench 17.7 vs 10.3. τ³-Banking 15.1 vs 10.7. Claw-Eval 61.3 vs 55.8. Terminal-Bench 2.1 58.4 vs 44.9. SkillsBench 46.9 vs 32.5. Average 45.0 vs 36.6.
- **Serving notes:** vLLM `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`.
- **Use for:** WorkBench backup. CompoWorld and Occamy are stronger on AutomationBench.
- **Credibility:** 2. Same lab as CompoWorld. SFT only.

## Fx-Work
- **Tuned from:** `Qwen/Qwen3.6-35B-A3B`
- **Repo:** `endless-frontier/Fx-Work` · base `Qwen/Qwen3.6-35B-A3B` · apache-2.0 · not gated · 2671 dl / 8 likes · 2026-10-09 · `Qwen3_5MoeForConditionalGeneration` (yes). No quants.
- **Training:** post-trained on 20K tool-interaction trajectories from occupational work tasks (real artifacts, rubrics, execution-guided verification). Method (SFT/RL) is not stated.
- **Claims (vs Qwen3.6-35B-A3B):** GDPval Rubric 86.66 vs 82.61. GDPval Elo 1352 vs 1053. APEX-Agents pass@1 25.2 vs 13.4. JobBench Main 25.19 vs 14.99.
- **Serving notes:** vLLM `--reasoning-parser qwen3`. The card's vLLM command has no tool parser. Tool execution comes from the harness. Recommends ≥128K context.
- **Use for:** Finance-Agent or WorkBench as a generalist office-work agent. Its training is about file deliverables, not API/DB state, so it is a weaker WorkBench fit than CompoWorld or Occamy.
- **Credibility:** 2.

## Agents-A1 (35B)
- **Tuned from:** not stated on the card; Qwen3.5/3.6-35B-A3B architecture (compared against both)
- **Repo:** `InternScience/Agents-A1` · base **not stated on card** (`Qwen3_5MoeForConditionalGeneration`; the table compares against both Qwen3.5- and Qwen3.6-35B-A3B) · apache-2.0 · not gated · 13,033 dl / 658 likes · 2026-09-21 · loads: yes. Official `-FP8`; community `r0b0tlab/Agents-A1-NVFP4`.
- **Training:** full-domain SFT, then domain teacher models, then multi-teacher multi-domain on-policy distillation.
- **Claims:** BrowseComp 75.51 (Qwen3.5-35B-A3B 61.0, Qwen3.6 67.93). GAIA 96.04. HLE w/ tools 47.6. τ2-Bench 79.81 (Qwen3.5 81.2). VitaBench 38.75.
- **Serving notes:** vLLM `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`, context 262144.
- **Use for:** BrowseComp-Plus and general tool use. Pool membership is uncertain until the base is confirmed.
- **Credibility:** 2. Large lab, released evaluation code. The base is unstated.

## Agents-A1-4B
- **Tuned from:** not stated on the card; almost certainly `Qwen/Qwen3.5-4B` (identical config dimensions; the card compares against it)
- **Repo:** `InternScience/Agents-A1-4B` · base not in metadata. Config matches Qwen3.5-4B (hidden 2560, 32 layers, vocab 248320) and the card compares against Qwen3.5-4B · apache-2.0 · not gated · 102,131 dl / 103 likes · 2026-07-14 · `Qwen3_5ForConditionalGeneration` (yes). **Official `-FP8`.** Community `sakamakismile/Agents-A1-4B-NVFP4`.
- **Training:** same 3-stage recipe as Agents-A1 (SFT, domain teachers, multi-teacher OPD).
- **Claims (vs Qwen3.5-4B):** BrowseComp 66.8 vs 47.2. XBench-DS-2510 90.0 vs 73.0. Seal0 45.8 vs 31.5. GAIA 95.1 vs 58.3. VitaBench 40.3 vs 22.0. IFEval 94.8 vs 89.8. **τ2-Bench 78.2 vs 79.9 (slightly worse)**. LiveCodeBench-v6 59.6 vs 55.8.
- **Serving notes:** vLLM `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`. `--language-model-only` for text-only. Sampling: temp 0.85, top_p 0.95, top_k 20, presence_penalty 1.1. Recommended system prompt is a deep-research persona with a `tavily_search` tool.
- **Use for:** BrowseComp-Plus is the best 4B on-pool option. It is a general tool-use generalist, but the τ2 regression is a caution for WorkBench.
- **Credibility:** 2. Widely downloaded, with base rows on the card. Self-reported.

## AREX-Turbo
- **Tuned from:** `Qwen/Qwen3.5-4B`
- **Repo:** `BAAI/AREX-Turbo` · base `Qwen/Qwen3.5-4B` · apache-2.0 · not gated · 388 dl / 50 likes · 2026-08-11 · `Qwen3_5ForConditionalGeneration` (yes). No FP8/NVFP4.
- **Training:** card describes the method (an inner research loop plus an outer self-improvement loop, with an `update_context` tool). No data or algorithm details on the card (arXiv 2607.21461).
- **Claims:** BrowseComp 70.7, GAIA 81.6, xbench-2510 57.0, DeepSearchQA 78.5, WideSearch-en 68.5, HLE w/ tools 40.6 (text-only subset). No Qwen3.5-4B row. Unified interface with `search`, `visit`, `update_context` and `finish` tools.
- **Serving notes:** `inference/README.md` uses vLLM `--reasoning-parser qwen3 --language-model-only` and **an XML `<tool_call>` protocol with tools embedded in the system prompt**. There is no server-side tool parser, and results go back as `<tool_response>`. BrowseComp prompts ship in `inference/prompts.py`.
- **Use for:** BrowseComp-Plus. Its protocol needs a custom loop rather than OpenAI tools.
- **Credibility:** 2. Strong org (BAAI). No base comparison.

## Apodex-1.0-4B-SFT / 0.8B-SFT
- **Tuned from:** 4B: `Qwen/Qwen3.5-4B`; 0.8B: `Qwen/Qwen3.5-0.8B`
- **Repo:** `apodex/Apodex-1.0-4B-SFT` (110 dl / 33 likes) and `apodex/Apodex-1.0-0.8B-SFT` (175 dl / 19 likes) · bases `Qwen/Qwen3.5-4B` / `Qwen/Qwen3.5-0.8B` · apache-2.0 · not gated · 2026-06-08 · `Qwen3_5ForConditionalGeneration` (yes).
- **Training:** SFT-only small variants of a recipe whose full version is SFT, agentic DPO and RL on long rollouts.
- **Claims:** 4B gets BrowseComp 48.8, BC-ZH 63.5, HLE-Text 32.9, DeepSearchQA 69.9. 0.8B gets 13.9, 10.7, 11.2 and 25.8. No base rows. For reference, Qwen3.5-4B BrowseComp is 47.2 per the Agents-A1-4B card, so the 4B gain looks marginal (not from Apodex card).
- **Serving notes:** vLLM `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`. temp 1.0, top_p 0.95, rep 1.05, max_tokens 32768. Pass tools via `tools=`.
- **Use for:** BrowseComp-Plus. The 0.8B is the only 0.8B search agent found.
- **Credibility:** 2.

## NeoHorse-1-4B
- **Tuned from:** `Qwen/Qwen3.5-4B`
- **Repo:** `TokenRhythm/NeoHorse-1-4B` · base `Qwen/Qwen3.5-4B` · apache-2.0 · not gated · 6,113 dl / 2,553 likes · 2026-09-10 · **`Qwen3_5ForCausalLM`, text-only repackage. Does not load on 0.21.** It would need the vision tower re-attached into ConditionalGeneration (not from card). Community `Codeflux/NeoHorse-1-4B-FP8`.
- **Training:** routing-guided curriculum SFT plus routing-guided on-policy distillation from a heterogeneous model pool.
- **Claims (vs Qwen3.5-4B):** QwenClawBench 44.68 vs 38.47. WorkBuddy Bench 34.41 vs 24.62. PinchBench 77.33 vs 71.19. VitaBench 32.00 vs 21.50. BFCL v4 61.79 vs 61.02. tau2-Bench 88.46 vs 84.29. IFEval 88.35 vs 87.06. Harness: SGLang 0.5.17, temp 1.0, thinking on.
- **Serving notes:** vLLM `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`.
- **Use for:** WorkBench. It is the best-claimed 4B workplace-tool model, but it is blocked on 0.21.
- **Credibility:** 2.

## AgentMercury-Qwen3.5-4B
- **Tuned from:** `Qwen/Qwen3.5-4B` (stated in the card text; not in its metadata)
- **Repo:** `Minbyul/AgentMercury-Qwen3.5-4B` · base Qwen3.5-4B (card text; no metadata) · apache-2.0 · not gated · 34 dl / 2 likes · 2026-08-14 · `Qwen3_5ForConditionalGeneration` (yes).
- **Training:** 200-step on-policy GRPO on about 2.3k MCP tool environments. Reward is final environment/DB-state verifiers. Checkpoint chosen at peak reward with zero degeneration.
- **Claims (only gains listed):** BFCL 31.93 vs 30.35. τ³-bench 0.747 vs 0.706. τ²-bench 0.457 vs 0.448. AIME 2026 0.553 vs 0.459. GPQA-D 0.770 vs 0.765. Finance-Reasoning 0.571 vs 0.563. SGLang, 32k context.
- **Serving notes:** SGLang `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`, context 32768.
- **Use for:** WorkBench, since its DB-state reward objective matches WorkBench's outcome metric, though the tool gains are tiny. Hard pool: AIME (+9 pts).
- **Credibility:** 1. Individual author. Card lists only the improved benchmarks.

## Qwen3.5-0.8B-ToolCalling-OPD
- **Tuned from:** `Qwen/Qwen3.5-0.8B`
- **Repo:** `Pepps233/Qwen3.5-0.8B-ToolCalling-OPD` · base `Qwen/Qwen3.5-0.8B` · apache-2.0 · 177 dl / 0 likes · 2026-10-16 · **`Qwen3_5ForCausalLM` (no)**.
- **Training:** on-policy distillation from a Qwen3.5-4B teacher on 1,477 single-turn xLAM/ToolACE prompts, one epoch.
- **Claims:** BFCL single-turn sample (500 examples) 71.07% mean. No base number on the card excerpt.
- **Use for:** WorkBench, but only single-turn. Not loadable.
- **Credibility:** 1.

## CloudSurf-4B-FC
- **Tuned from:** `google/gemma-4-E4B-it`
- **Repo:** `cloudsurf-software/CloudSurf-4B-FC` · base `google/gemma-4-E4B-it` · apache-2.0 · not gated · 65 dl / 3 likes · 2026-08-19 · `Gemma4ForConditionalGeneration` (yes). Merged BF16 is in `merged/`, plus 3 LoRA seeds. The card notes an earlier export bug fixed for vLLM 0.25.1, so it is worth verifying on 0.21.
- **Training:** QLoRA r8 on a 2,747-row proprietary mix: gpt-oss-120b teacher trajectories in state-verified, partially observable look-then-act environments, plus memory and irrelevance rows.
- **Claims (BFCL V4 full, 3-seed, vs stock E4B):** Overall 55.73 vs 34.81. Multi-Turn 43.46 vs 19.46. Web 48.00 vs 7.00. Memory 41.65 vs 16.56. Irrelevance 80.63 vs 84.37. Under the "matched best-serving frame" it is 54.91 vs 39.40. Format sensitivity regresses. Raw results are public, and τ²-bench trajectories are public.
- **Serving notes:** native interface is prompt mode (Gemma-4 turn format, bracket tool-call lists). OpenAI tools API needs their `serving/fc_tau2_bridge.py` in front of a /v1/completions server.
- **Use for:** WorkBench (multi-turn stateful FC). The gemma pool slot has no other tool tune.
- **Credibility:** 2. Unusually transparent (seeds, raw results, a gorilla PR), but a small, new org.

## ContextPilot-E4B
- **Tuned from:** `google/gemma-4-E4B-it`
- **Repo:** `tencent/ContextPilot-E4B` · base `google/gemma-4-E4B-it` · licence "other" · not gated · 838 dl / 9 likes · 2026-08-31 · `Gemma4ForConditionalGeneration` (yes).
- **Training:** RL for proactive context management (planning, memory, offloading tools) with fine-grained credit assignment (EMNLP 2026).
- **Claims:** card silent on numbers. It is evaluated on long-context QA and deep search.
- **Serving notes:** needs the ContextPilot repo's tool definitions and runtime.
- **Use for:** BrowseComp-Plus (gemma slot), conditional on the harness.
- **Credibility:** 2.

## Harness-1
- **Tuned from:** `openai/gpt-oss-20b`
- **Repo:** `pat-jj/harness-1` · base `openai/gpt-oss-20b` · licence not set · not gated · 307 dl / 107 likes · 2026-06-15 · `GptOssForCausalLM` (yes). BF16 merged weights.
- **Training:** 899 GPT-5.4-generated SFT trajectories, then RL on 3,453 **SEC** retrieval queries. Data and corpora are released, including BrowseComp+ (1.14M chunks) and SEC (2.1M chunks).
- **Claims:** card silent on numbers. arXiv 2606.02373 abstract: "0.730 average curated recall, outperforming the next strongest open search subagent by +11.4 points" (not from card).
- **Serving notes:** vLLM with GPT-OSS support. The repo has `docs/run_vllm_browsecompplus.md`. Its BC+ eval needs a Chroma collection and "OpenAI credentials for retrieval support used by the harness" (GitHub README).
- **Use for:** BrowseComp-Plus, with an explicit BC+ path, though it is a retrieval subagent. Finance-Agent EDGAR retrieval, since it was RL-trained on SEC filings.
- **Credibility:** 2. Paper, data and code are released. Numbers are not on the card.

## Context-1
- **Tuned from:** `openai/gpt-oss-20b`
- **Repo:** `chromadb/context-1` · base `openai/gpt-oss-20b` · apache-2.0 · not gated · 273 dl / 423 likes · 2026-03-30 · `GptOssForCausalLM` (yes). Community MXFP4 build exists.
- **Training:** SFT plus RL (CISPO) with a staged curriculum on web, legal and **finance** multi-hop retrieval tasks.
- **Claims:** card names BrowseComp-Plus, SealQA, FRAMES and HLE as held-out generalization benchmarks, with no numbers on the card.
- **Serving notes:** "The harness is not yet public. Running the model without it will not reproduce the results." It averages 2.56 parallel tool calls per turn and self-prunes its context.
- **Use for:** BrowseComp-Plus retrieval subagent, if the harness is released.
- **Credibility:** 2.

## Llama-xLAM-2-8b-fc-r
- **Tuned from:** `meta-llama/Llama-3.1-8B-Instruct` (implied by the name; not stated on the card)
- **Repo:** `Salesforce/Llama-xLAM-2-8b-fc-r` · base Llama-3.1-8B-Instruct (not from card; no base metadata, implied by name) · **cc-by-nc-4.0** · not gated · 98,133 dl / 66 likes · 2025-05-06 · `LlamaForCausalLM` (yes).
- **Training:** APIGen-MT simulated agent-human multi-turn data (`APIGen-MT-5k`, `xlam-function-calling-60k`).
- **Claims:** "state-of-the-art performance on BFCL and τ-bench, outperforming frontier models like GPT-4o and Claude 3.5". Numbers are in figures only (BFCL v3 as of 04/02/2025).
- **Serving notes:** vLLM ≥0.6.5 `--enable-auto-tool-choice --tool-parser-plugin ./xlam_tool_call_parser.py --tool-call-parser xlam`. The parser file is on the HF repo. 128k context.
- **Use for:** WorkBench, as a multi-turn stateful function-calling specialist for the Llama-8B slot.
- **Credibility:** 3. Salesforce research, widely reproduced on BFCL. Research-only licence.

## ToolACE-2.5-Llama-3.1-8B (and ToolACE-2-Llama-3.1-8B)
- **Tuned from:** `meta-llama/Llama-3.1-8B-Instruct`
- **Repo:** `Team-ACE/ToolACE-2.5-Llama-3.1-8B` (42 dl / 4 likes) and `Team-ACE/ToolACE-2-Llama-3.1-8B` (344 dl / 53 likes) · base Llama-3.1-8B-Instruct · apache-2.0 · not gated · 2025-08 · `LlamaForCausalLM` (yes).
- **Training:** ToolACE synthetic data (26,507 APIs, multi-agent dialogs, dual verification). ToolACE-2 adds self-refinement and decomposition. 2.5 adds ToolACE-R/DEV/MT for multi-turn.
- **Claims:** ToolACE-2 "state-of-the-art ... on BFCL ... rivaling the latest GPT-4 models", "highest scores among 8B-scale models" (BFCL-v3, figure only). ToolACE-2.5 "enhances the multi-turn tool-usage ability" (no numbers).
- **Serving notes:** outputs Python-style `[func(a=..)]` lists in the chat template. No vLLM parser named; the `pythonic` parser is the likely fit (not from card).
- **Use for:** WorkBench.
- **Credibility:** 3 (ICLR 2025, Huawei).

## watt-tool-8B
- **Tuned from:** `meta-llama/Llama-3.1-8B-Instruct`
- **Repo:** `watt-ai/watt-tool-8B` · base Llama-3.1-8B-Instruct · apache-2.0 · not gated · 1,101 dl / 119 likes · 2024-12-20 · `LlamaForCausalLM` (yes).
- **Training:** SFT on CoT-synthesized multi-turn tool dialogs plus DMPO.
- **Claims:** "state-of-the-art performance on BFCL". No numbers on the card.
- **Serving notes:** system prompt asks for `[func_name1(params...)]` output.
- **Use for:** WorkBench.
- **Credibility:** 2.

## Plancraft-author LoRA (Llama-3.1-8B)
- **Tuned from:** `meta-llama/Llama-3.1-8B-Instruct` (per its adapter config)
- **Repo:** `gautierdag/oa-llama3-r64-a32` · base per `adapter_config.json` is `meta-llama/Meta-Llama-3.1-8B-Instruct` · no licence · not gated · ~0 dl · 2024-09-20 · PEFT LoRA r64/α32 on q/k/v with `modules_to_save` embed_tokens and lm_head, so merge it before serving. Probably loadable after merge. The sibling `gautierdag/oam-llama3-r64-a32` has the custom `PlancraftOAM` architecture, so it does not load.
- **Training:** card is an empty template. (not from card) The uploader is the Plancraft first author, and r=64/α=32 matches the paper's "Llama 8B FT" (LoRA on 1,145 expert examples). Paper: 0.40 success vs 0.04 for untuned 8B on the move/smelt set, and 0.41 with the full action set. The paper also says the FT model "almost never uses think and search actions, and never emits impossible".
- **Use for:** Plancraft. This attribution is inferred, and the model never declares impossible.
- **Credibility:** 1 for the repo itself, given the empty card and inferred identity.

## Hermes-3-Llama-3.1-8B
- **Tuned from:** `meta-llama/Llama-3.1-8B` (base, not Instruct)
- **Repo:** `NousResearch/Hermes-3-Llama-3.1-8B` · base Llama-3.1-8B (base model) · llama3 · 466,906 dl / 521 likes · 2024-09-08 · `LlamaForCausalLM` (yes). The related `NousResearch/DeepHermes-ToolCalling-Specialist-Atropos` claims BFCL Parallel 0.46 vs 0.10 and Simple 0.5175 vs 0.21 vs DeepHermes-3 in reasoning mode only (experimental).
- **Use for:** a generic Hermes-format function-calling fallback for WorkBench (vLLM `hermes` parser, not from card). Credibility 2.

## PokeeResearch-7B
- **Tuned from:** `Qwen/Qwen2.5-7B-Instruct`
- **Repo:** `PokeeAI/pokee_research_7b` · base Qwen2.5-7B-Instruct · apache-2.0 · 69 dl / 99 likes · 2025-10-23 · `Qwen2ForCausalLM` (yes).
- **Training:** RLOO with AI-feedback reward on MiroRL-GenQA, 140 steps, 32K context.
- **Claims (mean@4):** BrowseComp 5.4 (PR+ 8.4), GAIA 36.9, HLE 15.2. All other 7B search agents on the card (Search-R1, ZeroSearch, ASearcher, DeepResearcher) get ≤3.2 on BrowseComp.
- **Use for:** BrowseComp-Plus only as the "best" Qwen2.5-7B option. The card's own table shows the whole Qwen2.5-7B search-agent family is near zero on BrowseComp-style questions.
- **Credibility:** 2.

## Toucan-Qwen2.5-7B-Instruct-v0.1
- **Tuned from:** `Qwen/Qwen2.5-7B-Instruct`
- **Repo:** `Agent-Ark/Toucan-Qwen2.5-7B-Instruct-v0.1` · base Qwen2.5-7B-Instruct · apache-2.0 · 54 dl / 6 likes · 2025-10-02 · `Qwen2ForCausalLM` (yes).
- **Training:** SFT on 119.3K Toucan-1.5M trajectories from real MCP servers (multi-turn, parallel, irrelevance). Hermes template.
- **Claims:** BFCL V3 and MCP-Universe gains (figures only).
- **Serving notes:** Hermes prompt template, so the `hermes` parser is the likely fit (not from card).
- **Use for:** WorkBench in the Qwen2.5-7B slot.
- **Credibility:** 2 (UW/IBM paper).

## Andy-4.2-Air
- **Tuned from:** `Qwen/Qwen3.5-4B`
- **Repo:** `Mindcraft-CE/Andy-4.2-Air` · base Qwen3.5-4B · licence "Andy" (custom) · 36 dl / 7 likes · `Qwen3_5ForConditionalGeneration` (yes). `Andy-4.2-Micro` is the 0.8B version.
- **Training:** 8-bit LoRA, 2,748 examples, 1 epoch.
- **Claims:** "first local model capable of getting a full set of diamond armour". No benchmark numbers.
- **Use for:** Plancraft only tangentially. It plays Minecraft through Mindcraft `!commands`, not crafting-grid move/smelt, and has no planning or crafting evaluation.
- **Credibility:** 1.

## Off-pool (brief)
- `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B` (Qwen3-30B-A3B family arch, 827 likes) is a deep-research agent.
- `miromind-ai/MiroThinker-v1.5-30B` (Qwen3-30B-A3B-Thinking-2507, MIT) is a deep-research agent.
- `hkust-nlp/WebExplorer-8B` (Qwen3-8B), `Menlo/Jan-nano` (Qwen3-4B) and `Intelligent-Internet/II-Search-4B` (Qwen3-4B) are small search agents.
- `nex-agi/Nex-N2-mini` / `Nex-N2.5-mini` (Qwen3.5-35B-A3B-Base) are generalist agents.
- `apodex/Apodex-1.1-mini` (Qwen3.5-35B-A3B; BrowseComp 71.5 for 1.0-mini) has official NVFP4/FP8.
- `ornith-ai/Ornith-1.5-35B-A3B` is "developed on top of Qwen3.5 and Gemma4", exact base unstated. It claims BrowseComp 67.6, GPQA-D 89.2 and MCP-Atlas 70.2.
- `Hcompany/Holo3-35B-A3B` (Qwen3.5-35B-A3B) is GUI-focused.
- `MadeAgents/Hammer2.1-7b` (Qwen2.5-Coder-7B-Instruct, cc-by-nc) and `Salesforce/xLAM-2-3b-fc-r` (Qwen2.5-3B per xLAM family; cc-by-nc) are function calling.
- `Qwen/Qwen-AgentWorld-35B-A3B` (Qwen3.5-35B-A3B-Base) is a world model or environment simulator, not an agent.
- `cindermond/world-model-plancraft-qwen3-4b-filtered` is a Plancraft world model, not a policy.
- Qwen3.5/3.6-family models with a different size from the pool (9B, 27B) were ignored.

---

# Part C: general-purpose and adjacent fine-tunes (law, biology, engineering, Plancraft)

**Method.** HF API `base_model:finetune:` and `base_model:adapter:` listings for every served base, sorted
by downloads and by likes (Qwen3.5-4B, 3.5-4B-Base, 0.8B, 0.8B-Base, Qwen3.6-35B-A3B, gemma-4-E4B(-it),
Ministral-3-3B Instruct/Reasoning/Base). The same listings were pulled for the `unsloth/` mirrors of those
bases, because many community tunes declare the mirror as their base. Qwen3.5-4B fine-tunes number more than
1,000, so I also ran per-keyword `search=` queries over it (law, legal, bio, med, chem, engineer, physics,
circuit, plan, craft, minecraft, game, world, reason, think, distill, opus, claude, gemini, deep, gpqa, hle,
mmlu and others). Global searches for biology, legal, law-llm, lawyer, engineering, electrical, plancraft,
textcraft, minecraft, gpqa, mmlu-pro, molecular and genetics were filtered to served bases.

For about 80 candidates I fetched the README, `/api/models/<id>` and `config.json` anonymously.
"dl" means HF `downloads` over the last 30 days. "Loads on 0.21" is judged only from `architectures`
in config.json: the `*ForConditionalGeneration` architectures load and `Qwen3_5(Moe)ForCausalLM` does not.
No smoke test has been run. Anything not taken from the card is marked (not from card).

### Summary

| subject | best exact | best adjacent / general | loads on 0.21 | cred |
|---|---|---|---|---|
| Law (MMLU-Pro law) | **none credible.** DoodDood/Personal-Jurisdiction-GRPO is real US-law RL but covers one LegalBench task and does not load. rajveer43 E4B legal QA LoRA loads but has only ROUGE numbers. | BigBang-v1 (in model-cards.md; general knowledge, HLE +14). 0GM-1.0-35B-A3B is the only card with law base rows, and its law score is flat (67.5→66.0). | exact: no / yes(LoRA); adjacent: yes | exact 1–2; adjacent 2 / 1–2 |
| Biology (GPQA-D mol-bio, genetics) | **none credible.** EpistemeAI/Reasoning-Medical0.1-E4B-sft claims MMLU-Pro biology 0.75 with no base row. | BigBang-v1 (BioMystery-HS 44.8→57.5). On 4B, Jackrong/Qwen3.5-4B-Neo (MMLU-Pro subset +1.6, biology up). | yes / yes | 1 / 2, 1–2 |
| Engineering (MMLU-Pro EE) | **none.** No EE or circuits fine-tune with weights exists on a served base. | 0GM-1.0-35B-A3B (engineering 42.0→46.4 at 4k context). RL-MATH (4B, in model-cards.md). | yes | 1–2 / 2 |
| Plancraft | **none credible.** The CMU-course TextCraft GRPO LoRAs (`*/scottyrl-a3-p5_textcraft_grpo`) cover Minecraft-style text crafting but report no numbers. | **TreezzZ/r2s-alfworld-qwen3.5-4b-student** (ALFWorld 43.4→81.9). CompoWorld (35B, DeepPlanning; in model-cards.md). | yes (LoRA) / yes | 1 / 2 |
| General-purpose slot | — | BigBang-v1 (35B, cred 2). Then 0GM-1.0-35B-A3B (cred 1–2). Snowdon1.1-Small has full base rows but is ≈ base. On 4B: RL-MATH (in model-cards.md), then Qwen3.5-4B-Neo. | yes | 2 / 1–2 |

**Overall.** No served-base checkpoint makes a credible claim of improving US law, molecular biology or
electrical engineering MCQ. The best evidence for the generalist slot remains BigBang-v1 on 35B and
RL-MATH on 4B. Most popular "reasoning" tunes on these bases are distilled from Claude, Gemini or
DeepSeek outputs. Their benchmarks are either missing, limited to 70–500 question subsets, or come from
visibly broken harnesses.

---

## Law

### Personal-Jurisdiction-GRPO
- **Tuned from:** `Qwen/Qwen3.5-4B`
- **Repo:** `DoodDood/Personal-Jurisdiction-GRPO` · base Qwen/Qwen3.5-4B · Apache-2.0 · not gated · 17 dl / 0 likes · 2026-08-03 · `Qwen3_5ForCausalLM` (**no**, text-only export).
- **Training:** GRPO, LoRA r16/α16 merged. Trained on a 2,000-row synthetic corpus (`DoodDood/personal-jurisdiction-grpo`) in a Prime Intellect environment. The model decomposes the question into three doctrinal yes/no sub-questions (domicile, contacts, nexus).
- **Claims:** LegalBench `personal_jurisdiction` (50 rows, non-thinking, greedy): base 66.0% → **74.0%**. Gains are in the domicile and contacts-plus-nexus slices.
- **Serving notes:** only works with the card's system prompt and `enable_thinking=False`, and emits a strict 4-line Yes/No format.
- **Use for:** Law, exact domain but far too narrow (one civil-procedure question type). It will not answer 10-option MMLU-Pro law without re-prompting and does not load on 0.21.
- **Credibility:** 2 for the claim (base row, public data, honest scope); 1 as a team member.

### gemma-4-E4B medical-legal-finance-qa
- **Tuned from:** `google/gemma-4-E4B-it`
- **Repo:** `rajveer43/gemma-4-E4B-medical-legal-finance-qa` · base google/gemma-4-E4B-it · Apache-2.0 · not gated · 271 dl / 13 likes · 2026-04-12 · LoRA adapter plus a Q4_K_M GGUF of the merge. config says `Gemma4ForConditionalGeneration` (yes after merge, or via vLLM LoRA if 0.21 supports Gemma4 LoRA, not from card).
- **Training:** QLoRA (NF4, r16/α32) on a Kaggle T4. Data: `medalpaca/medical_meadow_medqa`, `nguyen-brat/legal_case_qa` and `gbharti/finance-alpaca`, with the domain set by the system prompt.
- **Claims:** ROUGE-1 0.224 on its own mix. No accuracy numbers and no base comparison.
- **Serving notes:** Gemma-4 chat template; pass the domain in the system prompt.
- **Use for:** Law, exact but weak. It is the only legal-QA tune on a served base that loads.
- **Credibility:** 1.

### gemma4-e4b-legal-grpo
- **Tuned from:** `google/gemma-4-E4B-it`
- **Repo:** `Semaj90/gemma4-e4b-legal-grpo` · base gemma-4-E4B-it (trained on `unsloth/...-bnb-4bit`) · Apache-2.0 · 11 dl / 1 like · 2026-04-05 · LoRA adapter only (language tensors), no config.
- **Training:** GRPO, 250 steps and 1,000 prompts drawn from FineTome, GSM8K, Pile-of-Law, LexGLUE CaseHOLD and SCOTUS. The rewards check citation format, "logical connector" words, list formatting and length.
- **Claims:** none.
- **Use for:** Law, but the rewards train style, not correctness.
- **Credibility:** 1.

### Others screened (law)
- `CATI-AI/Qwen3.6-35B-A3B-Legal-v2` is a LoRA with manual gating, an empty card and 3 dl. Cred 1, unknown content.
- `flywheel-ai/legal-intake` is a Qwen3.6-35B LoRA merge (`Qwen3_5MoeForConditionalGeneration`, loads) for law-firm client intake and scheduling. Its card says it "never gives legal advice" and is "roughly on par with the base". Not a reasoning tune.
- `cs-552-2026-Clanker-Scientists/` and `OzairFZN/Qwen3.5-4B-Legal-Agent-US` (plus `-FR`) are `Qwen3_5ForCausalLM` (no) with empty Trainer cards. They look like a course project.
- `KitakitaQaQ/qwen35-4b-law-lora-r2` is a LoRA with an empty TRL card.
- `Nebulaw1/jev-qwen3.5-4b-legal-lora` is a Chinese finite-choice legal scorer that reads letter logits and does not generate.
- The Vietnamese, Thai, Czech, Brazilian, Indonesian, Iranian, Swiss and Portuguese legal LoRAs are the wrong jurisdiction or language.
- `JThomas-CoE/coe-qwen3.6-mmlu_pro-law-r25-19b-a3b-q4_k_m` is a GGUF-only expert-pruned Qwen3.6 (256→128 experts), with no training. Its expert mask was profiled on MMLU-Pro subsets, a contamination risk. It scores 67.27 vs 68.18 for the parent.

---

## Biology

### Reasoning-Medical0.1-E4B-sft
- **Tuned from:** `google/gemma-4-E4B-it` (via the `unsloth/gemma-4-E4B-it` mirror)
- **Repo:** `EpistemeAI/Reasoning-Medical0.1-E4B-sft` · base unsloth/gemma-4-E4B-it · Apache-2.0 · not gated · 31 dl / 5 likes · 2026-07-22 · `Gemma4ForConditionalGeneration` (yes), BF16 8.0B incl. vision/audio.
- **Training:** SFT on `EpistemeAI/medical-reasoning-mix-100k` (~100k blended public medical-reasoning CoT). The card does not list the source sets, so overlap with MMLU or MMLU-Pro is unknown (contamination risk).
- **Claims:** lm-eval `mmlu_pro_biology` (5-shot, custom-extract) 0.75 and `mmlu_proX_en_biology` 0.75. The identical round values suggest a small `--limit` run (not from card). MedQA-4opt 0.760, compared with *other* models (Huatuo-o1-RL-8B 0.726, MedGemma-4B 0.644). **No E4B base row.** BioSafeBench 87.5/100.
- **Serving notes:** Gemma-4 template; card suggests deterministic decoding for MCQ.
- **Use for:** Biology, adjacent (medicine to molecular biology and genetics). It is the only served-base tune with an MMLU-Pro biology number.
- **Credibility:** 1. No base comparison, opaque data mix, likely a subset eval.

### Qwen3.5-4B-Medical-Reasoning
- **Tuned from:** `Qwen/Qwen3.5-4B`
- **Repo:** `Kerassy/Qwen3.5-4B-Medical-Reasoning` · base Qwen3.5-4B · Apache-2.0 · 347 dl / 1 like · 2026-09-16 · `Qwen3_5ForConditionalGeneration` (yes); GGUFs included.
- **Training:** Unsloth LoRA SFT on `FreedomIntelligence/medical-o1-reasoning-SFT`. Per the HuatuoGPT-o1 paper that data was generated with GPT-4o (not from card), so the ToS caveat applies.
- **Claims:** MedQA-USMLE-4opt full test 68.34% (greedy, max_tokens 1024). No base row.
- **Serving notes:** thinking by default in `<think>`; prefill an empty think block for direct answers.
- **Use for:** Biology, adjacent (clinical, not molecular).
- **Credibility:** 1.

### MedForge Qwen3.5-4B-Distill
- **Tuned from:** `Qwen/Qwen3.5-4B`
- **Repo:** `fang04/medforge-qwen3.5-4b-distill` · base Qwen3.5-4B · Apache-2.0 · 60 dl / 0 likes · 2026-09-04 · `Qwen3_5ForConditionalGeneration` (yes).
- **Training:** LoRA r64 SFT on 4,490 examples: 3,817 Chinese medical-exam traces from DeepSeek V4-flash on de-contaminated CMExam-train, plus 673 general replay examples.
- **Claims (paired, official sampling, 32k budget):** CMExam 74.0→80.4 (in-domain), CMB-val 60.0→65.7, MedXpertQA (English) 25.1→26.4 (flat). Tokens per answer fall about 3×.
- **Use for:** Biology only marginally. It is Chinese clinical, and its English transfer (MedXpertQA) is flat. Of note: the card has the most careful evaluation in this group.
- **Credibility:** 2 for the claims; low relevance.

### Others screened (biology)
- `epicmajorman/Gemma4-Biomedical-E4B` (loads) is **abliterated** and advertises no refusals on chemical synthesis. No numbers. Avoid.
- `z8086486/GCCL-Medical-LLM-Qwen3.5-4B` is CausalLM (no), and its base row (MedQA 0.39%) shows a broken harness.
- `xtools-at/Ministral-3-3B-Medical-Reasoning` (Mistral3, loads) is a 4k-example medical-o1 LoRA, and its card says outright that it has no benchmarks.
- `lastmass/Qwen3.5-Medical-GSPO` is a Chinese medical-o1 SFT plus GSPO.
- `EpistemeAI/OpenMedResearch-Gemma-4E4N` is PaperSearchQA biomedical factoid QA, with no numbers.
- `omicverse/OmicAI-4B` is gated with an inaccessible card.
- `CodeXomics-ToolAgent-4B-v1` does genome-browser tool calling.
- `nvidia/NV-Reason-CT` (custom `VLM3D_ForConditionalGeneration`, no) and `JLKGroup/JOOMED` (brain-CT) are imaging models.
- MedJev, medx and biodecision are classifiers or extractors.
- `JThomas-CoE/coe-qwen3.5-biology-18b-a3b` is a pruned Qwen3.5-35B, not a served base.

---

## Engineering

### 0GM-1.0-35B-A3B (Preview-0427)
- **Tuned from:** `Qwen/Qwen3.6-35B-A3B`
- **Repo:** `0G-AI/0GM-1.0-35B-A3B-0427` · base Qwen/Qwen3.6-35B-A3B · Apache-2.0 · not gated · 105 dl / 16 likes · 2026-08-18 · `Qwen3_5MoeForConditionalGeneration` (yes), BF16 35.95B (~67 GB, same as base). Only GGUF quants exist (mradermacher); **no NVFP4 or FP8**.
- **Training:** card silent ("trained on 0G Compute's decentralized network").
- **Claims (vs Qwen3.6-35B-A3B, harness not named):**
  - MMLU-Pro at **4k context**: 75.75→77.62. Engineering **42.0→46.4**, physics 74.2→79.3, chemistry 72.1→74.6, biology 91.1→91.1, **law 67.5→66.0**.
  - AIME 2026: 70.0→83.3 (30 questions). MATH-500: 95.6→95.8.
  - The 4k budget truncates the base model's reasoning, so these gains may only reflect shorter chains (not from card).
- **Serving notes:** `--reasoning-parser qwen3`, tools `--tool-call-parser qwen3_coder`, keep ≥128K context, MTP supported.
- **Use for:** Engineering, adjacent (the only engineering base-vs-ft number on a served base); also a general slot. Does not help law.
- **Credibility:** 1–2. It has base rows and a subject breakdown, but the training is undisclosed and the 4k-context protocol is unusual.

### DeepSeek-V4-Pro-Qwen3.5-4B
- **Tuned from:** `Qwen/Qwen3.5-4B` (via the `unsloth/Qwen3.5-4B` mirror)
- **Repo:** `Jackrong/DeepSeek-V4-Pro-Qwen3.5-4B` · base unsloth/Qwen3.5-4B · Apache-2.0 · 2,792 dl / 21 likes · 2026-08-13 · `Qwen3_5ForConditionalGeneration` (yes).
- **Training:** Unsloth LoRA SFT, merged to BF16. Data: ~250K maths and STEM responses from DeepSeek-V4-Pro (Max effort). No code data.
- **Claims:** GSM8K 91.77% (two runs, MTP-Q8 GGUF, llama.cpp). MMLU-Pro on 500 Qs each: maths 80.4, physics 74.2, chemistry 74.8. **No 4B base row.** The sibling 9B distill gains only +0.9 over Qwen3.5-9B on the same samples (90.53 vs 89.60).
- **Serving notes:** thinking model; card tested at 64K context.
- **Use for:** Engineering, adjacent (numerical STEM). A different-teacher alternative to RL-MATH.
- **Credibility:** 1–2. The teacher is DeepSeek, whose licence permits distillation (not from card).

### Others screened (engineering)
- `JThomas-CoE/coe-qwen3.6-mmlu_pro-engineering-r25-19b-a3b-q4_k_m` is a GGUF-only expert-pruned Qwen3.6. Its mask was **profiled on MMLU-Pro**, and it needs thinking off. It scores 74.23 vs the parent's 64.95 at the "canonical" harness but 81.44 for the parent at full compute, so the parent wins. Not usable on vLLM and not a fine-tune.
- `UnipatAI/UniMath-35B-A3B` (loads) is olympiad proof-writing with a test-time self-evolution loop through its own scripts. No numbers on the card.
- `DavidOKB/MathThink-Qwen-3.5-4B` is a LoRA on Nemotron-SFT-Math-v3 with no numbers.
- `ben072292/Qwen3.5-4B-mixed-physics-v3` and `Laplaces-Red-Devils/physics-v07c` (CausalLM) are empty cards or contest-specific.
- `Rapnss/Volt-Circuit` is an empty repo with no weights.
- `covaga/gemma4-8b-electrical-vision` does electrical-symbol image recognition.

---

## Plancraft

### R2S ALFWorld student (Qwen3.5-4B)
- **Tuned from:** `Qwen/Qwen3.5-4B`
- **Repo:** `TreezzZ/r2s-alfworld-qwen3.5-4b-student` · base Qwen/Qwen3.5-4B · Apache-2.0 · not gated · 215 dl / 0 likes · 2026-10-21 · `Qwen3_5ForConditionalGeneration` (yes). The vision tower and MTP head are copied from the base.
- **Training:** on-policy distillation (sampled reverse KL, 150 updates, rollout T 0.4) from a Qwen3.5-9B teacher repaired by R2S (paper arXiv 2610.28586). Code: `Tree-Shu-Zhao/miles-agent@r2s`. The teacher is an open model, so there is no ToS issue.
- **Claims:** ALFWorld success (mean of seen and unseen, 3 decode seeds): student alone 43.4 ± 1.0, vanilla OPD 65.8 ± 2.6, **R2S 81.9 ± 2.4**.
- **Serving notes:** base chat template; trained with the paper's environment prompts and tool schemas. Sibling students exist for WebShop and ScienceWorld.
- **Use for:** Plancraft, adjacent. It is an embodied text agent doing multi-step household planning through tool calls. It has no crafting-grid or "impossible" training, and its gains may be tied to the ALFWorld prompt format (not from card).
- **Credibility:** 2. It has a paper, a base row, seeds and released code, but no adoption yet.

### SkillRise-ALFWorld Qwen3.5-4B (step 40)
- **Tuned from:** `Qwen/Qwen3.5-4B`
- **Repo:** `YuyangZhao/skillrise-alfworld-qwen3.5-4b-step40` · base Qwen3.5-4B · Apache-2.0 · 16 dl · 2026-10-05 · `Qwen3_5ForConditionalGeneration` (yes).
- **Training:** SkillRise cross-task meta-RL on ALFWorld (verl, 40 steps, thinking off). An independent reproduction on a new base.
- **Claims:** in-distribution validation, 128 tasks, pass@1 0.086→0.547 (step 45 reaches 0.727).
- **Serving notes:** `enable_thinking=False`; low temperature; SkillRise Solve/Curate prompt format.
- **Use for:** Plancraft, adjacent (weaker evidence than R2S).
- **Credibility:** 1–2.

### TextCraft GRPO LoRAs (CMU 11-768 A3)
- **Tuned from:** `Qwen/Qwen3.5-4B`
- **Repo:** `uilydna/scottyrl-a3-p5_textcraft_grpo` (and about 20 sibling student repos, e.g. `iversse/...`) · base Qwen/Qwen3.5-4B · no licence · ~10–25 dl · 2026-09/10 · PEFT LoRA r32, no config (vLLM `--enable-lora --max-lora-rank 32` per card).
- **Training:** GRPO in a course framework (scottyrl via Tinker) on TextCraft-Synth medium and hard. (not from card) TextCraft is the ADaPT Minecraft-recipe text environment, the closest domain to Plancraft's crafting task.
- **Claims:** none. The card gives only the grader commands.
- **Use for:** Plancraft, the closest domain, but untested. It has no grid-slot move or smelt actions and no "impossible" action.
- **Credibility:** 1.

### SkillGym-Qwen3.5-4B
- **Tuned from:** `Qwen/Qwen3.5-4B`
- **Repo:** `reasonwang/SkillGym-Qwen3.5-4B` · base Qwen3.5-4B · Apache-2.0 · 143 dl · 2026-10-01 · `Qwen3_5ForConditionalGeneration` (yes).
- **Training:** SFT for 2 epochs on 19k verified skill-use trajectories from three (unnamed) teacher models (paper arXiv 2609.37539).
- **Claims (vs Qwen3.5-4B):** SkillGym 33.8→47.0, SkillEval 62.8→70.8, SkillsBench 10.1→14.3, Skill-Use-Bench 8.8→48.7.
- **Serving notes:** `vllm serve` as the base; T 0.6, top_p 0.95, top_k 20.
- **Use for:** Plancraft, general agentic fallback (tool use plus reading instructions). It also suits a WorkBench backup.
- **Credibility:** 2.

### Others screened (Plancraft)
- `valentin0901/Planium-4B` (loads) does PDDL/STRIPS planning with tool calls. There is no training description and no numbers ("work in progress"). Cred 1, but it is the only explicit planning tune.
- `OrionLLM/GRM-2.5` (4B, loads) claims DeepPlanning 17.9 and TAU2 80.2 with no base row.
- `napalna/rby1-skill-planner-qwen3.5-4b` is a robot skill JSON planner: 82→99% on its own 170 cases, schema-forced output.
- `Andy-4.2-Air/Micro` (in model-cards.md) plays Mindcraft `!commands`.
- `Code-as-World-VL-4B` (video physics), `GameScaling` (keyboard/mouse), VLA/robotics tunes, `alearn2/rft_alfworld_*` (empty card), `saad1926q/8puzzle` (empty) and `JoshPurtell/craftax-cpt` (CPT on 0.8B-Base, card is the base's) are not relevant.
- `Zhanli-Li/DataWorld-35B-A3B` is RL on data-science environments.

---

## General-purpose

### Snowdon1.1-Small
- **Tuned from:** `Qwen/Qwen3.6-35B-A3B`
- **Repo:** `tri-fair-lab/Snowdon1.1-Small` · base Qwen/Qwen3.6-35B-A3B · Apache-2.0 · 179 dl / 15 likes · 2026-08-25 · `Qwen3_5MoeForConditionalGeneration` (yes), BF16 35.95B.
- **Training:** a rank-1 Fisher-routed directional ablation (merged), then one epoch of Constitutional DPO. Aimed at removing topic-conditioned political refusals (Imperial College tech report).
- **Claims (vs base):** GPQA-D 85.4 vs 85.2, HLE 13.3 vs 14.1, MMLU-Pro 85.2 vs 85.2, AIME26 93.3 vs 86.7, SWE-bench Pro 32.9 vs 34.3, Terminal-Bench 2.1 38.4 vs 45.2.
- **Serving notes:** identical to the base.
- **Use for:** a general slot only as a near-base variant. It has the most complete base rows in this list but adds little diversity.
- **Credibility:** 2.

### Qwen3.5-4B-Neo
- **Tuned from:** `Qwen/Qwen3.5-4B`
- **Repo:** `Jackrong/Qwen3.5-4B-Neo` · base Qwen3.5-4B · Apache-2.0 · 87 dl / 29 likes · 2026-03-23 · `Qwen3_5ForConditionalGeneration` (yes).
- **Training:** LoRA SFT (response-only). Data: `stepfun-ai/Step-3.5-Flash-SFT` and Nemotron-SFT competitive-programming, SWE and IF sets, plus `nohurry/Opus-4.6-Reasoning-3000x-filtered` and `Jackrong/Qwen3.5-reasoning-700x`. **Part of the data is Claude output (ToS caveat).**
- **Claims:** MMLU-Pro on a 250-Q subset over 5 categories: base 80.4 → 82.0. The card says it is better in biology, CS, maths and other sciences and worse in physics. Think length drops from 6,962 to 3,955 characters.
- **Use for:** Biology and general slot on 4B (adjacent). The gain is 4 questions out of 250, within noise (not from card).
- **Credibility:** 1–2.

### Qwen3.8-4B-Distill and Qwen3.8-35B-A3B-Distill
- **Tuned from:** 4B: `Qwen/Qwen3.5-4B`; 35B: `Qwen/Qwen3.6-35B-A3B` (by architecture; the cards name an unverifiable "Qwen3.8" teacher)
- **Repo:** `empero-ai/Qwen3.8-4B-Distill` (258,324 dl / 71 likes, 2026-08-15) and `empero-ai/Qwen3.8-35B-A3B-Distill` (4,215 dl / 79 likes, 2026-09-16) · Apache-2.0 · both ConditionalGeneration (yes). A community NVFP4 build exists: `qualitymaterial/Qwen3.8-35B-A3B-Distill-NVFP4-FreeToken`.
- **Training:** full SFT on ~45k "Qwen3.8 2.4T A95B" teacher traces from "internal" datasets. The teacher cannot be verified.
- **Claims:**
  - 4B (lm-eval, CoT): MMLU 0.354→0.553, GSM8K 0.850→0.785. The base MMLU of 0.354 shows a broken extraction.
  - 35B (loglik): MMLU 0.838→0.834, ARC-C 0.548→0.582.
- **Use for:** not recommended. The download count looks bot-inflated (not from card).
- **Credibility:** 1.

### Claude / Gemini reasoning distills (grouped)
- `hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled` (2,821 dl / 93 likes, loads): MMLU-Pro on 70 Qs, base 42.86 → 75.71. A base model at 42.9% means a broken harness.
- `lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled` (5,461 dl / 204 likes, loads; community FP8-Dynamic and AWQ builds): ~7.8k Opus 4.7 traces, attention-only LoRA. MMLU-Pro (500/subject) 74.9 with **engineering 54.8 and law 55.6**, no base row. GPQA and AIME are "pending".
- `Jackrong/Qwopus3.6-35B-A3B-v1` (61 likes, loads; a community NVFP4 build exists): 3-stage SFT. Only third-party "quality/reliability" scores, no standard benchmarks.
- `Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled(-v2)`: v2 trails the base on HumanEval (0.732 vs 0.768) and claims only shorter thinking.
- `TeichAI/Qwen3.5-4B-Claude-Opus-Reasoning-Distill`: GPQA-D 0.283 = base 0.283. Its base MMLU of 0.256 is broken.
- `Ayodele01/gemma-4-E4B-Gemini-3.1-Pro-Reasoning-Distill`: LoRA r8, 0.5 epoch, "matches base".
- `armand0e/Qwen3.6-35B-A3B-Fable-5-Distill`: empty card.
- **Use for:** none. **All are distilled from proprietary outputs (ToS caveat),** and none shows a credible gain on GPQA-D, MMLU-Pro or HLE. lordx64's own subject table suggests these distills are weak exactly on law and engineering.
- **Credibility:** 1.

### Ministral-3-3B and Gemma-E4B general tunes
- `ValiantLabs/Ministral-3-3B-Reasoning-2512-Esper3.1` (Mistral3, loads) is DeepSeek-V3.1/V3.2/R1-generated DevOps, code and AI-research SFT with no numbers. The Qwen3.6 Esper3.1 sibling is `Qwen3_5MoeForCausalLM` (no).
- `ertghiu256/ministral-3-3b-glm-distillation` (loads) is 919 GLM-5.2 logic-puzzle examples plus 50 GRPO steps, with no numbers.
- `Nitish-Garikoti/ministral3-3b-sft-openthoughts-v2` (loads) is a QLoRA on OpenThoughts-Agent with losses only.
- **Use for:** nothing beyond the official Ministral-3B-Reasoning (in model-cards.md).
- **Credibility:** 1.

### Others screened (general)
- `Edge0/Edge0-35B-A3B-preview` (1,080 likes) is int4 plus a recovery LoRA that scores *below* base (GPQA-D 79.8 vs 81.8, MMLU-Pro 81.0 vs 84.6). It is compression, not capability.
- `MergeAILab/Merge-35B-A3B-Reasoning-v5-b1` reports a private benchmark only.
- `GestaltLabs/Ornstein3.6-35B-A3B` has no numbers.
- `quimmedes/Deepwen-3.6` is GGUF only and focused on 3D and graphics.
- `FINAL-Bench/Darwin-36B-KR` has a near-empty card.
- `dnotitia/DNA3.0-*` is an uncensoring and persona tune.
- `sky7350/Mica-v0.1-4B` is CausalLM (no) and a decision model, with MMLU-Pro 53.0 vs base 45.2 on its harness.
- `Light-O1-Preview` is a human-motion model with a custom architecture.
- `nvidia/NV-Reason-CT` is CT imaging.

---

## Planned new bases (brief)

| subject | Llama-3.1-8B(-Instruct) | Llama-3.2-3B | Qwen2.5-7B-Instruct | gpt-oss-20b / granite-4.2-8b / Nemotron-Nano-9B-v2 |
|---|---|---|---|---|
| Law | `MaziyarPanahi/calme-2.3-legalkit-8b` (Instruct; SFT on synthetic legal data from Nous-Hermes-2-Mixtral; only OpenLLM numbers; cred 1). `ricdomolm/lawma-8b` is on Llama-3 (not 3.1; not fetched). | `Lexsi/llama32-3b-legal-dpo-recover` (1 dl, not fetched) | `ShengbinYue/LawLLM-7B` (Chinese law, not fetched) | none |
| Biology | `zou-lab/BioMed-R1-8B` (Instruct; paper 2505.11462; card has no table). `ncbi/Gene-R1-8B` (gene-set function analysis; NCBI). HuatuoGPT-o1-8B (in model-cards.md). `MegaScience/Llama3.1-8B-MegaScience` (base Llama; 1.25M open science SFT; paper). | none credible | HuatuoGPT-o1-7B (in model-cards.md) | gpt-oss-20b: medical-reasoning tunes only (`dousery/...`, cred 1) |
| Engineering / STEM | R1-Distill-Llama-8B (in model-cards.md), MegaScience-8B, `Skywork/Skywork-o1-Open-Llama-3.1-8B` (not fetched) | none | **`open-thoughts/OpenThinker3-7B`** (open OpenThoughts3-1.2M: 850k maths, 250k code, 100k science; GPQA-D 53.7, AIME24 69.0, JEEBench 72.4; cred 3) | none |
| Plancraft | Plancraft-author LoRA `gautierdag/oa-llama3-r64-a32` (in model-cards.md) | `docedit/sft-Llama-3.2-1B-plancraft-*` (1B, not 3B; not fetched) | **`osunlp/early-experience-textcraft`** (Agent Learning via Early Experience, arXiv 2510.08558; `il`/`sr`/`iwm` subfolders; minimal card; cred 2). Siblings exist for ALFWorld, TravelPlanner and ScienceWorld. `AgentFlow/agentflow-planner-7b` (empty card). | none |
| General | Hermes-3 / DeepHermes (in model-cards.md) | none | OpenThinker3-7B | granite-4.2-8b: none. Nemotron-Nano-9B-v2: only NVIDIA quants and a Japanese variant. |

## Caveats
- None of the "loads on 0.21" verdicts has been smoke-tested. LoRA-only repos (rajveer43, TextCraft LoRAs) need
  either vLLM 0.21 LoRA support for `Qwen3_5ForConditionalGeneration` / `Gemma4ForConditionalGeneration` or an
  offline merge (not from card).
- Contamination: the JThomas-CoE models were profiled on MMLU-Pro. EpistemeAI's 100k mix and the various
  Opus datasets do not state their sources, so MMLU-Pro or GPQA overlap cannot be excluded.
- 35B candidates without NVFP4/FP8 builds (0GM, Snowdon, UniMath) must be served in BF16 (~67 GB).
